"""Registro de modelos entrenados.

Cada entrenamiento produce un directorio `models/<fecha>_<hash>/` con los
artefactos y un `metadata.json` que permite responder, meses después, la única
pregunta que importa cuando un pronóstico sale raro: *¿con qué datos, qué
código y qué semilla se produjo esto?*

El hash resume datos + configuración + código, así que dos entrenamientos con
las mismas entradas caen en el mismo directorio y el registro no se llena de
copias idénticas.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from precios.config import MODELS_DIR, PROJECT_ROOT

log = logging.getLogger(__name__)

LATEST_PATH = MODELS_DIR / "latest.json"
NOMBRE_MODELO = "modelo.joblib"
NOMBRE_METADATA = "metadata.json"


@dataclass
class Metadata:
    """Todo lo necesario para reproducir y auditar un entrenamiento."""

    version: str
    entrenado_en: str
    semana_inicio: str
    semana_fin: str
    n_series: int
    n_observaciones: int
    productos: list[str]
    plazas: list[str]
    horizontes: list[int]
    nivel_intervalo: float
    semilla: int
    params_lgbm: dict[str, Any]
    commit: str | None
    radios_conformales: dict[str, float] = field(default_factory=dict)
    metricas: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def commit_actual() -> str | None:
    """Hash corto del commit, o None si no estamos en un repo git."""
    try:
        salida = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return salida.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def calcular_hash(panel: pd.DataFrame, params: dict, semilla: int, commit: str | None) -> str:
    """Hash corto y estable de las entradas del entrenamiento.

    Incluye el rango y tamaño de los datos, los hiperparámetros, la semilla y
    el commit. No incluye la hora: reentrenar dos veces con lo mismo debe dar
    la misma versión.
    """
    piezas = [
        str(panel["semana"].min().date()),
        str(panel["semana"].max().date()),
        str(len(panel)),
        str(panel.groupby(["producto_id", "plaza_id"], observed=True).ngroups),
        json.dumps(params, sort_keys=True),
        str(semilla),
        commit or "sin-git",
    ]
    return hashlib.sha256("|".join(piezas).encode("utf-8")).hexdigest()[:8]


def guardar(
    modelo: Any,
    metadata: Metadata,
    *,
    models_dir: Path = MODELS_DIR,
    extras: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Guarda el modelo y su metadata, y actualiza el puntero `latest`.

    Args:
        modelo: objeto entrenado, serializable con joblib.
        metadata: descripción completa del entrenamiento.
        extras: tablas opcionales a guardar como CSV junto al modelo.

    Returns:
        Ruta del directorio de la versión.
    """
    destino = models_dir / metadata.version
    destino.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, destino / NOMBRE_MODELO)
    (destino / NOMBRE_METADATA).write_text(metadata.to_json(), encoding="utf-8")
    for nombre, tabla in (extras or {}).items():
        tabla.to_csv(destino / f"{nombre}.csv", index=False, encoding="utf-8")

    marcar_latest(metadata.version, models_dir=models_dir)
    log.info("Modelo registrado en %s", destino)
    return destino


def marcar_latest(version: str, *, models_dir: Path = MODELS_DIR) -> None:
    """Apunta `models/latest.json` a una versión.

    Se usa un archivo y no un symlink: en Windows los enlaces simbólicos
    requieren permisos especiales y romperían el pipeline sin avisar.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "latest.json").write_text(
        json.dumps({"version": version}, indent=2), encoding="utf-8"
    )


def version_latest(*, models_dir: Path = MODELS_DIR) -> str | None:
    """Versión apuntada por `latest`, o None si no hay ninguna registrada."""
    ruta = models_dir / "latest.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8")).get("version")


def cargar(version: str | None = None, *, models_dir: Path = MODELS_DIR) -> tuple[Any, Metadata]:
    """Carga un modelo registrado y su metadata.

    Args:
        version: versión concreta; si se omite, se usa la apuntada por `latest`.

    Raises:
        FileNotFoundError: si no hay modelo registrado.
    """
    version = version or version_latest(models_dir=models_dir)
    if version is None:
        raise FileNotFoundError(
            "No hay ningún modelo registrado. Ejecuta `make train` primero."
        )
    destino = models_dir / version
    if not (destino / NOMBRE_MODELO).exists():
        raise FileNotFoundError(f"La versión {version} no tiene {NOMBRE_MODELO}")

    modelo = joblib.load(destino / NOMBRE_MODELO)
    crudo = json.loads((destino / NOMBRE_METADATA).read_text(encoding="utf-8"))
    return modelo, Metadata(**crudo)


def listar(*, models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Inventario de versiones registradas, de la más reciente a la más antigua."""
    filas = []
    for ruta in sorted(models_dir.glob("*/")):
        meta = ruta / NOMBRE_METADATA
        if not meta.exists():
            continue
        d = json.loads(meta.read_text(encoding="utf-8"))
        filas.append(
            {
                "version": d.get("version"),
                "entrenado_en": d.get("entrenado_en"),
                "semana_fin": d.get("semana_fin"),
                "n_series": d.get("n_series"),
                "commit": d.get("commit"),
                "es_latest": d.get("version") == version_latest(models_dir=models_dir),
            }
        )
    if not filas:
        return pd.DataFrame(
            columns=["version", "entrenado_en", "semana_fin", "n_series", "commit", "es_latest"]
        )
    return pd.DataFrame(filas).sort_values("entrenado_en", ascending=False).reset_index(drop=True)
