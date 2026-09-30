"""Etapas `validate` y `clean`.

`validate` comprueba el esquema diario crudo y detiene el pipeline si algo no
cuadra. `clean` normaliza nombres, recorta al alcance, agrega a semanal, marca
calidad y escribe el panel consolidado.

Entrada: `data/processed/diario.parquet`
Salida:  `data/processed/semanal.parquet` y `data/processed/reporte_series.csv`
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from precios.cleaning.aggregate import agregar_semanal, reindexar_semanas
from precios.cleaning.normalize import (
    colapsar_duplicados_diarios,
    filtrar_alcance,
    normalizar_nombres,
)
from precios.config import PROCESSED_DIR, load_normalizacion, load_reglas, load_scope
from precios.data.validation import reporte_series, validar_diario, validar_semanal
from precios.logging_setup import setup_logging
from precios.pipeline.ingest import DIARIO_PATH

log = logging.getLogger(__name__)

SEMANAL_PATH = PROCESSED_DIR / "semanal.parquet"
REPORTE_PATH = PROCESSED_DIR / "reporte_series.csv"


def validate(origen: Path = DIARIO_PATH) -> pd.DataFrame:
    """Valida el esquema diario crudo.

    Se valida ANTES de normalizar nombres: así el chequeo de duplicados mide el
    estado real del origen y no los solapamientos que introduce un renombre.

    Raises:
        ValidationError: si alguna regla falla.
    """
    if not origen.exists():
        raise FileNotFoundError(f"No existe {origen}. Ejecuta `make ingest` primero.")
    df = pd.read_parquet(origen)
    validar_diario(df, load_reglas())
    return df


def clean(
    origen: Path = DIARIO_PATH,
    *,
    destino: Path = SEMANAL_PATH,
    reporte: Path = REPORTE_PATH,
) -> pd.DataFrame:
    """Normaliza, recorta al alcance y agrega a panel semanal."""
    reglas = load_reglas()
    scope = load_scope()
    norm = load_normalizacion()

    df = validate(origen)
    df = normalizar_nombres(df, norm)
    df = colapsar_duplicados_diarios(df)
    df = filtrar_alcance(df, scope)

    semanal = agregar_semanal(df, reglas)
    semanal = reindexar_semanas(semanal)
    validar_semanal(semanal.dropna(subset=["precio_kg"]), reglas)

    rep = reporte_series(semanal.dropna(subset=["precio_kg"]), reglas)
    destino.parent.mkdir(parents=True, exist_ok=True)
    semanal.to_parquet(destino, index=False)
    rep.to_csv(reporte, index=False, encoding="utf-8")

    log.info("Panel semanal escrito: %s (%d filas)", destino.name, len(semanal))
    log.info("Reporte de series escrito: %s (%d series)", reporte.name, len(rep))
    _resumen(rep, scope)
    return semanal


def _resumen(rep: pd.DataFrame, scope) -> None:
    """Deja en el log un resumen legible del estado de las series."""
    esperadas = scope.n_series_esperadas
    log.info("Series esperadas por configuración: %d | obtenidas: %d", esperadas, len(rep))
    if len(rep) != esperadas:
        faltan = esperadas - len(rep)
        log.warning("Diferencia de %d series frente a lo esperado", faltan)
    no_aptas = rep[~rep["apta"]]
    for _, fila in no_aptas.iterrows():
        log.warning(
            "NO apta: %s @ %s -> %s",
            fila["producto_id"],
            fila["plaza_id"],
            fila["motivo_exclusion"],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validación y limpieza del panel semanal")
    parser.add_argument(
        "--solo-validar",
        action="store_true",
        help="ejecuta únicamente la etapa validate",
    )
    args = parser.parse_args()

    setup_logging()
    if args.solo_validar:
        validate()
        log.info("Validación superada.")
    else:
        clean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
