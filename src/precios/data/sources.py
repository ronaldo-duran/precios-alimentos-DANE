"""Fuentes de datos de SIPSA.

Todas las implementaciones devuelven el mismo esquema diario canónico
(`COLUMNAS_CANONICAS`), de modo que el resto del pipeline no sabe ni le importa
de dónde vinieron los datos.
"""

from __future__ import annotations

import datetime as dt
import gzip
import logging
import shutil
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import requests

from precios.config import INCOMING_DIR, RAW_DIR

log = logging.getLogger(__name__)

#: Esquema diario canónico que toda fuente debe producir.
COLUMNAS_CANONICAS: tuple[str, ...] = (
    "fecha",
    "producto_sipsa",
    "plaza_sipsa",
    "precio_kg",
    "precio_kg_min",
    "precio_kg_max",
    "grupo",
    "departamento",
    "municipio",
    "fuente",
)

WSDL_URL = "https://appweb.dane.gov.co/sipsaWS/SrvSipsaUpraBeanService?WSDL"
ENDPOINT = "https://appweb.dane.gov.co/sipsaWS/SrvSipsaUpraBeanService"
NAMESPACE = "http://servicios.sipsa.co.gov.dane/"

_SOBRE_SOAP = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<env:Envelope xmlns:env="http://www.w3.org/2003/05/soap-envelope">'
    '<env:Body><{op} xmlns="{ns}"/></env:Body>'
    "</env:Envelope>"
)


class DataSource(ABC):
    """Contrato de una fuente de datos diarios de precios mayoristas."""

    #: Nombre corto que queda registrado en la columna `fuente`.
    name: str

    @abstractmethod
    def fetch(self, *, force: bool = False) -> pd.DataFrame:
        """Devuelve datos diarios en el esquema `COLUMNAS_CANONICAS`.

        Args:
            force: si es True ignora cualquier caché y vuelve a leer el origen.
        """

    @staticmethod
    def _conformar(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
        """Completa columnas faltantes y fija orden y tipos del esquema canónico."""
        df = df.copy()
        df["fuente"] = fuente
        for col in COLUMNAS_CANONICAS:
            if col not in df.columns:
                df[col] = pd.NA
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
        for col in ("precio_kg", "precio_kg_min", "precio_kg_max"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[list(COLUMNAS_CANONICAS)]


class SipsaSoapSource(DataSource):
    """Servicio web SOAP oficial del DANE.

    El servicio **no acepta parámetros**: cada llamada devuelve el histórico
    completo (~310 MB de XML para `promediosSipsaParcial`). Por eso se guarda un
    snapshot en parquet en `data/raw/` y las corridas posteriores del mismo día
    lo reutilizan en vez de volver a descargar.
    """

    name = "sipsa_soap"

    #: Mapea el nombre del campo XML -> columna canónica, por operación.
    _CAMPOS = {
        "promediosSipsaParcial": {
            "enmaFecha": "fecha",
            "artiNombre": "producto_sipsa",
            "fuenNombre": "plaza_sipsa",
            "promedioKg": "precio_kg",
            "minimoKg": "precio_kg_min",
            "maximoKg": "precio_kg_max",
            "grupNombre": "grupo",
            "deptNombre": "departamento",
            "muniNombre": "municipio",
        },
    }

    def __init__(
        self,
        operacion: str = "promediosSipsaParcial",
        *,
        raw_dir: Path = RAW_DIR,
        timeout: int = 600,
        conservar_xml: bool = False,
    ) -> None:
        if operacion not in self._CAMPOS:
            raise ValueError(
                f"Operación no soportada: {operacion!r}. Disponibles: {sorted(self._CAMPOS)}"
            )
        self.operacion = operacion
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.conservar_xml = conservar_xml

    def snapshot_path(self, fecha: dt.date | None = None) -> Path:
        """Ruta del snapshot parquet para una fecha de ingesta."""
        fecha = fecha or dt.date.today()
        return self.raw_dir / f"{self.operacion}_{fecha.isoformat()}.parquet"

    def fetch(self, *, force: bool = False) -> pd.DataFrame:
        destino = self.snapshot_path()
        if destino.exists() and not force:
            log.info("Reutilizando snapshot del día: %s", destino.name)
            return pd.read_parquet(destino)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        xml_path = destino.with_suffix(".xml")
        try:
            self._descargar(xml_path)
            df = self._parsear(xml_path)
        finally:
            if xml_path.exists():
                if self.conservar_xml:
                    self._comprimir(xml_path)
                else:
                    xml_path.unlink()

        df = self._conformar(df, self.name)
        df.to_parquet(destino, index=False)
        log.info("Snapshot guardado: %s (%d filas)", destino.name, len(df))
        return df

    def _descargar(self, destino: Path) -> None:
        """Descarga la respuesta SOAP a disco por streaming (la respuesta es enorme)."""
        cuerpo = _SOBRE_SOAP.format(op=self.operacion, ns=NAMESPACE)
        log.info("Llamando %s (la respuesta completa pesa cientos de MB)...", self.operacion)
        with requests.post(
            ENDPOINT,
            data=cuerpo.encode("utf-8"),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
            timeout=self.timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            with destino.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        log.info("Descargados %.1f MB", destino.stat().st_size / 1e6)

    def _parsear(self, xml_path: Path) -> pd.DataFrame:
        """Parsea incrementalmente: el XML no cabe cómodo en memoria."""
        campos = self._CAMPOS[self.operacion]
        filas: list[dict[str, str | None]] = []
        for _, elem in ET.iterparse(str(xml_path), events=("end",)):
            if elem.tag.rsplit("}", 1)[-1] != "return":
                continue
            fila = {}
            for hijo in elem:
                clave = campos.get(hijo.tag.rsplit("}", 1)[-1])
                if clave is not None:
                    fila[clave] = hijo.text
            filas.append(fila)
            elem.clear()
        if not filas:
            raise RuntimeError(
                f"{self.operacion} devolvió 0 registros. El servicio del DANE puede "
                "estar caído o haber cambiado su contrato."
            )
        df = pd.DataFrame(filas)
        # Las fechas llegan como '2026-09-29T00:00:00-05:00'; solo importa el día.
        df["fecha"] = df["fecha"].str.slice(0, 10)
        log.info("Parseados %d registros de %s", len(df), self.operacion)
        return df

    @staticmethod
    def _comprimir(xml_path: Path) -> None:
        with xml_path.open("rb") as origen, gzip.open(f"{xml_path}.gz", "wb") as destino:
            shutil.copyfileobj(origen, destino)
        xml_path.unlink()


class ManualExcelSource(DataSource):
    """Archivos del DANE depositados a mano en `data/incoming/`.

    Sirve de respaldo si el servicio SOAP se cae o se descontinúa, y para cargar
    los microdatos históricos (catálogo 776, 2013-2024). Ese histórico es
    **mensual**, así que no alimenta el modelo semanal directamente: se usa solo
    para contexto y comparación.
    """

    name = "manual_excel"

    #: Alias de encabezado observados -> columna canónica. Los microdatos del
    #: DANE cambian de nombres entre años (2013-2017 usa V1..V5).
    _ALIAS = {
        "fecha": "fecha",
        "fecha programada para la recoleccion": "fecha",
        "v1": "fecha",
        "producto": "producto_sipsa",
        "nombre articulo": "producto_sipsa",
        "v3": "producto_sipsa",
        "mercado": "plaza_sipsa",
        "nombre de la fuente": "plaza_sipsa",
        "v4": "plaza_sipsa",
        "precio_promedio_por_kilogramo_": "precio_kg",
        "precio promedio diario": "precio_kg",
        "v5": "precio_kg",
        "grupo": "grupo",
        "id grupo": "grupo",
        "v2": "grupo",
    }

    def __init__(self, incoming_dir: Path = INCOMING_DIR) -> None:
        self.incoming_dir = incoming_dir

    def fetch(self, *, force: bool = False) -> pd.DataFrame:
        archivos = sorted(
            p for p in self.incoming_dir.glob("*") if p.suffix.lower() in {".xlsx", ".xls", ".csv"}
        )
        if not archivos:
            log.warning(
                "No hay archivos en %s; la fuente manual no aporta filas.", self.incoming_dir
            )
            vacio = pd.DataFrame(columns=["fecha"])
            return self._conformar(vacio, self.name)

        partes = [self._leer(p) for p in archivos]
        df = pd.concat(partes, ignore_index=True)
        log.info("Leídas %d filas de %d archivo(s) manual(es)", len(df), len(archivos))
        return self._conformar(df, self.name)

    def _leer(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
        df.columns = [self._ALIAS.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        faltantes = {"fecha", "producto_sipsa", "plaza_sipsa", "precio_kg"} - set(df.columns)
        if faltantes:
            raise ValueError(
                f"{path.name}: faltan columnas {sorted(faltantes)} tras aplicar alias. "
                f"Encabezados leídos: {list(df.columns)}"
            )
        return df


class WebSource(DataSource):
    """Descarga directa de archivos publicados en el sitio del DANE.

    NO IMPLEMENTADA, y probablemente innecesaria: el DANE ya expone un servicio
    web oficial (`SipsaSoapSource`) que cubre este caso sin raspar HTML. Se deja
    el esqueleto solo por si el SOAP se descontinúa.

    Si alguna vez se implementa, lo verificado en la fase 0 fue:
      - `microdatos.dane.gov.co` no publica `robots.txt` (404), así que no hay
        reglas declaradas; `www.dane.gov.co/robots.txt` solo restringe rutas
        administrativas y el calendario, no las páginas de SIPSA.
      - Los datos son CC BY-SA 4.0: exigen atribución al DANE.
      - Aun así conviene revisar los términos de uso vigentes antes de
        automatizar descargas, y limitar la frecuencia.
    """

    name = "web"

    def fetch(self, *, force: bool = False) -> pd.DataFrame:
        raise NotImplementedError(
            "WebSource no está implementada a propósito: usa SipsaSoapSource, "
            "que consume la API oficial del DANE."
        )
