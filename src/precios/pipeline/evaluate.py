"""Etapa `evaluate`: compara todos los modelos con validación walk-forward.

Entrada: `data/processed/semanal.parquet` + `reporte_series.csv`
Salida:  `backtest.parquet`, `metricas_por_horizonte.csv`,
         `metricas_por_serie.csv`, `cobertura_intervalos.csv`,
         `importancia_features.csv`

Todos los modelos —univariados y globales— se miden sobre **los mismos
orígenes**, definidos una sola vez. Si cada familia derivara los suyos, las
tablas no serían comparables.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from precios.config import PROCESSED_DIR, load_evaluacion, load_lgbm_params
from precios.evaluation.backtest import backtest_global, backtest_panel
from precios.evaluation.choques import (
    cargar_episodios,
    degradacion_por_regimen,
    etiquetar_con_episodios,
    marcar_semanas_de_choque,
)
from precios.evaluation.metrics import resumir
from precios.evaluation.splits import describir_folds, rolling_origins
from precios.features.build import anadir_objetivos, construir_features, filas_a_predecir
from precios.logging_setup import setup_logging
from precios.models.baselines import baselines_por_defecto
from precios.models.conformal import ConformalGlobal
from precios.models.estadisticos import modelos_estadisticos
from precios.models.lgbm import LGBMGlobal
from precios.pipeline.clean import REPORTE_PATH, SEMANAL_PATH

log = logging.getLogger(__name__)

BACKTEST_PATH = PROCESSED_DIR / "backtest.parquet"
METRICAS_H_PATH = PROCESSED_DIR / "metricas_por_horizonte.csv"
METRICAS_SERIE_PATH = PROCESSED_DIR / "metricas_por_serie.csv"
COBERTURA_PATH = PROCESSED_DIR / "cobertura_intervalos.csv"
IMPORTANCIA_PATH = PROCESSED_DIR / "importancia_features.csv"
SEMANAS_CHOQUE_PATH = PROCESSED_DIR / "semanas_choque.csv"
DEGRADACION_PATH = PROCESSED_DIR / "degradacion_por_regimen.csv"

#: Nivel nominal de los intervalos de predicción.
NIVEL_INTERVALO = 0.8


def fijar_semillas(semilla: int) -> None:
    """Fija las semillas globales para que la corrida sea reproducible."""
    random.seed(semilla)
    np.random.seed(semilla)


def cargar_panel_apto(
    panel_path: Path = SEMANAL_PATH, reporte_path: Path = REPORTE_PATH, solo_aptas: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el panel semanal y el reporte, filtrando a las series aptas."""
    if not panel_path.exists():
        raise FileNotFoundError(f"No existe {panel_path}. Ejecuta `make clean` primero.")
    panel = pd.read_parquet(panel_path)
    reporte = pd.read_csv(reporte_path)
    if solo_aptas:
        aptas = set(
            zip(
                reporte.loc[reporte["apta"], "producto_id"],
                reporte.loc[reporte["apta"], "plaza_id"],
                strict=True,
            )
        )
        mascara = [
            par in aptas
            for par in zip(panel["producto_id"], panel["plaza_id"], strict=True)
        ]
        panel = panel[mascara].copy()
    return panel, reporte


def evaluate(*, solo_baselines: bool = False) -> pd.DataFrame:
    """Corre el backtest completo y escribe las tablas de métricas."""
    cfg = load_evaluacion()
    fijar_semillas(cfg.semilla)

    panel, reporte = cargar_panel_apto(solo_aptas=cfg.solo_series_aptas)
    n_semanas = int(panel.groupby(["producto_id", "plaza_id"], observed=True).size().max())
    folds = rolling_origins(
        n_semanas,
        min_train=cfg.min_train,
        horizonte_max=cfg.horizonte_max,
        paso=cfg.paso,
        max_origenes=cfg.max_origenes,
    )
    log.info("Orígenes comunes a todos los modelos: %s", describir_folds(folds))

    univariados = baselines_por_defecto()
    if not solo_baselines:
        univariados = univariados + modelos_estadisticos()

    log.info("Backtest univariado: %s", [m.name for m in univariados])
    resultados = [backtest_panel(panel, univariados, solo_aptas=reporte, folds=folds)]

    importancias = pd.DataFrame()
    if not solo_baselines:
        features = anadir_objetivos(construir_features(panel), cfg.horizontes)
        params = load_lgbm_params()
        constructores = {
            "lgbm_cuantil": lambda: LGBMGlobal(
                horizontes=cfg.horizontes, params=params, semilla=cfg.semilla
            ),
            "lgbm_conformal": lambda: ConformalGlobal(
                horizontes=cfg.horizontes,
                nivel=NIVEL_INTERVALO,
                params=params,
                semilla=cfg.semilla,
            ),
        }
        log.info("Backtest global: %s", list(constructores))
        resultados.append(backtest_global(features, constructores, folds))
        importancias = _importancias_ultimo_origen(features, folds[-1].origen, cfg)

    backtest = pd.concat(resultados, ignore_index=True)
    _analisis_choques(panel, backtest)
    _escribir(backtest, importancias)
    _resumen_honesto(backtest)
    return backtest


def _importancias_ultimo_origen(features: pd.DataFrame, origen: int, cfg) -> pd.DataFrame:
    """Importancia de features del modelo entrenado en el origen más reciente."""
    modelo = LGBMGlobal(
        horizontes=cfg.horizontes, params=load_lgbm_params(), semilla=cfg.semilla
    ).fit(features, origen)
    # Se comprueba de paso que el modelo produce algo en el origen final.
    modelo.predict(filas_a_predecir(features, origen))
    return modelo.importancias()


def _analisis_choques(panel: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    """Clasifica las semanas y mide cuánto se degrada cada modelo en los choques."""
    episodios, deteccion = cargar_episodios()
    semanas = marcar_semanas_de_choque(
        panel,
        cuantil=float(deteccion.get("cuantil_choque", 0.90)),
        min_series=int(deteccion.get("min_series", 5)),
    )
    semanas = etiquetar_con_episodios(semanas, episodios)
    semanas.to_csv(SEMANAS_CHOQUE_PATH, index=False, encoding="utf-8")

    degradacion = degradacion_por_regimen(backtest, semanas)
    degradacion.to_csv(DEGRADACION_PATH, index=False, encoding="utf-8")

    peores = (
        degradacion[degradacion["regimen"] == "choque"]
        .dropna(subset=["factor_degradacion"])
        .sort_values("factor_degradacion", ascending=False)
    )
    for _, fila in peores.head(5).iterrows():
        log.warning(
            "En semanas de choque, %s a h%d empeora x%.1f (MASE %.2f frente a normal)",
            fila["modelo"],
            fila["h"],
            fila["factor_degradacion"],
            fila["mase"],
        )
    return semanas


def _escribir(backtest: pd.DataFrame, importancias: pd.DataFrame) -> None:
    por_h = resumir(backtest, por=["modelo", "h"])
    por_serie = resumir(backtest, por=["modelo", "producto_id", "plaza_id", "h"])

    backtest.to_parquet(BACKTEST_PATH, index=False)
    por_h.to_csv(METRICAS_H_PATH, index=False, encoding="utf-8")
    por_serie.to_csv(METRICAS_SERIE_PATH, index=False, encoding="utf-8")

    con_intervalo = backtest.dropna(subset=["lo", "hi"]) if "lo" in backtest else pd.DataFrame()
    if not con_intervalo.empty:
        cobertura = resumir(con_intervalo, por=["modelo", "h"])
        cobertura["nivel_nominal_pct"] = 100 * NIVEL_INTERVALO
        cobertura.to_csv(COBERTURA_PATH, index=False, encoding="utf-8")

    if not importancias.empty:
        importancias.to_csv(IMPORTANCIA_PATH, index=False, encoding="utf-8")

    log.info("Backtest escrito: %s (%d filas)", BACKTEST_PATH.name, len(backtest))


def _resumen_honesto(backtest: pd.DataFrame) -> None:
    """Deja en el log lo que funcionó y, sobre todo, lo que no."""
    por_h = resumir(backtest, por=["modelo", "h"])
    tabla = por_h.pivot(index="modelo", columns="h", values="mase")
    referencia = tabla.loc["naive"] if "naive" in tabla.index else None

    log.info("MASE por modelo y horizonte (menor es mejor):")
    for modelo, fila in tabla.sort_values(1).iterrows():
        valores = " ".join(f"h{h}={v:.3f}" for h, v in fila.items())
        marca = ""
        if referencia is not None and modelo != "naive":
            gana = (fila < referencia).sum()
            marca = f"  <- le gana al naive en {gana}/{len(fila)} horizontes"
        log.info("  %-18s %s%s", modelo, valores, marca)

    if "lo" in backtest.columns:
        con_int = backtest.dropna(subset=["lo", "hi"])
        if not con_int.empty:
            cob = resumir(con_int, por=["modelo", "h"])
            log.info(
                "Cobertura empírica de los intervalos (nominal %.0f%%):",
                100 * NIVEL_INTERVALO,
            )
            for _, f in cob.iterrows():
                log.info(
                    "  %-18s h%d: %.1f%% (ancho medio %.1f%% del precio)",
                    f["modelo"],
                    f["h"],
                    f["cobertura_pct"],
                    f["amplitud_pct"],
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest walk-forward de todos los modelos")
    parser.add_argument(
        "--solo-baselines",
        action="store_true",
        help="omite ETS, ARIMA y LightGBM (corrida rápida)",
    )
    args = parser.parse_args()
    setup_logging()
    evaluate(solo_baselines=args.solo_baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
