"""Etapa `ingest`: trae los datos crudos y los consolida.

Entrada: servicio SOAP del DANE (o `data/incoming/` con la fuente manual).
Salida:  `data/processed/diario.parquet` + `data/processed/ingest_status.json`

Es idempotente: si el origen no trae periodos nuevos respecto a lo ya
consolidado, no reescribe nada y deja constancia en el status para que las
etapas siguientes puedan terminar sin reentrenar.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from precios.config import PROCESSED_DIR
from precios.data.sources import DataSource, ManualExcelSource, SipsaSoapSource
from precios.logging_setup import setup_logging

log = logging.getLogger(__name__)

DIARIO_PATH = PROCESSED_DIR / "diario.parquet"
STATUS_PATH = PROCESSED_DIR / "ingest_status.json"

FUENTES: dict[str, type[DataSource]] = {
    "soap": SipsaSoapSource,
    "manual": ManualExcelSource,
}


def ingest(
    fuente: str = "soap",
    *,
    force: bool = False,
    destino: Path = DIARIO_PATH,
) -> dict:
    """Ejecuta la ingesta y devuelve el status.

    Args:
        fuente: clave en `FUENTES`.
        force: vuelve a descargar aunque exista snapshot del día.
        destino: parquet consolidado de salida.

    Returns:
        Diccionario de status, también escrito en `ingest_status.json`.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    source = FUENTES[fuente]()
    nuevo = source.fetch(force=force)

    fecha_max_previa = None
    if destino.exists():
        previo = pd.read_parquet(destino, columns=["fecha"])
        if not previo.empty:
            fecha_max_previa = previo["fecha"].max()

    fecha_max_nueva = nuevo["fecha"].max()
    hay_nuevos = fecha_max_previa is None or fecha_max_nueva > fecha_max_previa

    if hay_nuevos:
        nuevo.to_parquet(destino, index=False)
        log.info("Consolidado escrito: %s (%d filas)", destino.name, len(nuevo))
    else:
        log.warning(
            "Sin periodos nuevos: el origen llega a %s y lo consolidado ya está en %s. "
            "No se reescribe.",
            fecha_max_nueva.date(),
            fecha_max_previa.date(),
        )

    status = {
        "fuente": fuente,
        "ejecutado_en": dt.datetime.now().isoformat(timespec="seconds"),
        "n_filas": int(len(nuevo)),
        "fecha_min": str(nuevo["fecha"].min().date()),
        "fecha_max": str(fecha_max_nueva.date()),
        "fecha_max_previa": str(fecha_max_previa.date()) if fecha_max_previa is not None else None,
        "hay_datos_nuevos": bool(hay_nuevos),
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta de precios mayoristas SIPSA")
    parser.add_argument("--fuente", choices=sorted(FUENTES), default="soap")
    parser.add_argument(
        "--force", action="store_true", help="ignora el snapshot cacheado del día"
    )
    args = parser.parse_args()

    setup_logging()
    status = ingest(args.fuente, force=args.force)
    log.info("Status: %s", json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
