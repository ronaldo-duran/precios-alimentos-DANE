"""Etapa `evaluate`: corre los baselines con validación walk-forward.

Entrada: `data/processed/semanal.parquet` + `reporte_series.csv`
Salida:  `data/processed/backtest_baselines.parquet` y las tablas de métricas
         `metricas_por_horizonte.csv` y `metricas_por_serie.csv`
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from precios.config import PROCESSED_DIR, load_evaluacion
from precios.evaluation.backtest import backtest_panel
from precios.evaluation.metrics import resumir
from precios.logging_setup import setup_logging
from precios.models.baselines import baselines_por_defecto
from precios.pipeline.clean import REPORTE_PATH, SEMANAL_PATH

log = logging.getLogger(__name__)

BACKTEST_PATH = PROCESSED_DIR / "backtest_baselines.parquet"
METRICAS_H_PATH = PROCESSED_DIR / "metricas_por_horizonte.csv"
METRICAS_SERIE_PATH = PROCESSED_DIR / "metricas_por_serie.csv"


def fijar_semillas(semilla: int) -> None:
    """Fija las semillas globales para que la corrida sea reproducible."""
    random.seed(semilla)
    np.random.seed(semilla)


def evaluate(
    panel_path: Path = SEMANAL_PATH,
    reporte_path: Path = REPORTE_PATH,
) -> pd.DataFrame:
    """Ejecuta el backtest de los baselines y escribe las tablas de métricas."""
    cfg = load_evaluacion()
    fijar_semillas(cfg.semilla)

    if not panel_path.exists():
        raise FileNotFoundError(f"No existe {panel_path}. Ejecuta `make clean` primero.")
    panel = pd.read_parquet(panel_path)
    reporte = pd.read_csv(reporte_path) if cfg.solo_series_aptas else None

    resultados = backtest_panel(
        panel,
        baselines_por_defecto(),
        min_train=cfg.min_train,
        horizonte_max=cfg.horizonte_max,
        paso=cfg.paso,
        max_origenes=cfg.max_origenes,
        solo_aptas=reporte,
    )

    por_h = resumir(resultados, por=["modelo", "h"])
    por_serie = resumir(resultados, por=["modelo", "producto_id", "plaza_id", "h"])

    resultados.to_parquet(BACKTEST_PATH, index=False)
    por_h.to_csv(METRICAS_H_PATH, index=False, encoding="utf-8")
    por_serie.to_csv(METRICAS_SERIE_PATH, index=False, encoding="utf-8")

    log.info("Backtest escrito: %s (%d filas)", BACKTEST_PATH.name, len(resultados))
    _resumen_honesto(por_h, por_serie)
    return resultados


def _resumen_honesto(por_h: pd.DataFrame, por_serie: pd.DataFrame) -> None:
    """Deja en el log lo que funcionó y, sobre todo, lo que no."""
    log.info("MASE por modelo y horizonte (1.0 = igual que el naive):")
    tabla = por_h.pivot(index="modelo", columns="h", values="mase")
    for modelo, fila in tabla.iterrows():
        valores = " ".join(f"h{h}={v:.3f}" for h, v in fila.items())
        log.info("  %-22s %s", modelo, valores)

    peores = por_serie[(por_serie["modelo"] != "naive") & (por_serie["mase"] > 1.0)]
    if not peores.empty:
        log.warning(
            "%d combinaciones (modelo, serie, horizonte) NO le ganan al naive",
            len(peores),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest walk-forward de los baselines")
    parser.parse_args()
    setup_logging()
    evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
