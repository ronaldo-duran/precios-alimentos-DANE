"""Etapa `tune`: búsqueda limitada y registrada de hiperparámetros de LightGBM.

Entrada: `data/processed/semanal.parquet`
Salida:  `config/lgbm_params.yaml` (configuración elegida, versionada) y
         `data/processed/busqueda_hiperparametros.csv` (todas las probadas)

La búsqueda usa **solo la mitad antigua de los orígenes**. La mitad reciente se
reserva para la evaluación final, de modo que la comparación contra el naive no
esté contaminada por la propia selección.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd
import yaml

from precios.config import CONFIG_DIR, PROCESSED_DIR, load_evaluacion
from precios.evaluation.splits import rolling_origins
from precios.features.build import anadir_objetivos, construir_features
from precios.logging_setup import setup_logging
from precios.models.busqueda import REJILLA, buscar, partir_origenes
from precios.pipeline.evaluate import cargar_panel_apto, fijar_semillas

log = logging.getLogger(__name__)

PARAMS_PATH = CONFIG_DIR / "lgbm_params.yaml"
REGISTRO_PATH = PROCESSED_DIR / "busqueda_hiperparametros.csv"


def tune() -> tuple[dict, pd.DataFrame]:
    """Corre la rejilla sobre los orígenes de búsqueda y guarda el resultado."""
    cfg = load_evaluacion()
    fijar_semillas(cfg.semilla)

    panel, _ = cargar_panel_apto(solo_aptas=cfg.solo_series_aptas)
    features = anadir_objetivos(construir_features(panel), cfg.horizontes)

    n_semanas = int(panel.groupby(["producto_id", "plaza_id"], observed=True).size().max())
    folds = rolling_origins(
        n_semanas,
        min_train=cfg.min_train,
        horizonte_max=cfg.horizonte_max,
        paso=cfg.paso,
        max_origenes=cfg.max_origenes,
    )
    busqueda, evaluacion = partir_origenes(folds)
    log.info(
        "Búsqueda en %d orígenes (obs %d-%d); evaluación reservada: %d orígenes (obs %d-%d)",
        len(busqueda),
        busqueda[0].origen,
        busqueda[-1].origen,
        len(evaluacion),
        evaluacion[0].origen,
        evaluacion[-1].origen,
    )

    mejor, tabla = buscar(
        features, busqueda, horizontes=cfg.horizontes, rejilla=REJILLA, semilla=cfg.semilla
    )

    tabla.to_csv(REGISTRO_PATH, index=False, encoding="utf-8")
    _escribir_params(mejor, tabla, busqueda, evaluacion, cfg.semilla)
    return mejor, tabla


def _numero_limpio(valor: float) -> int | float:
    """Evita que YAML guarde `31.0` donde LightGBM espera `31`."""
    return int(valor) if float(valor).is_integer() else float(valor)


def _escribir_params(mejor: dict, tabla: pd.DataFrame, busqueda, evaluacion, semilla: int) -> None:
    """Guarda la configuración ganadora con la trazabilidad de cómo se eligió."""
    contenido = {
        "_procedencia": {
            "generado_por": "make tune (precios.pipeline.tune)",
            "fecha": dt.date.today().isoformat(),
            "semilla": semilla,
            "configuraciones_probadas": int(len(tabla)),
            "origenes_busqueda": f"{busqueda[0].origen}-{busqueda[-1].origen}",
            "origenes_evaluacion_reservados": f"{evaluacion[0].origen}-{evaluacion[-1].origen}",
            "mase_busqueda": round(float(tabla.loc[0, "mase"]), 4),
            "mase_peor_config": round(float(tabla["mase"].max()), 4),
            "nota": (
                "Elegida sobre la mitad antigua de los orígenes. El registro "
                "completo está en data/processed/busqueda_hiperparametros.csv. "
                "Editar a mano es válido; deja constancia aquí si lo haces."
            ),
        },
        "params": {k: _numero_limpio(v) for k, v in mejor.items()},
    }
    with PARAMS_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(contenido, fh, allow_unicode=True, sort_keys=False)
    log.info("Configuración elegida escrita en %s", PARAMS_PATH.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Búsqueda limitada de hiperparámetros")
    parser.parse_args()
    setup_logging()
    mejor, tabla = tune()
    log.info("Resumen de la búsqueda:\n%s", tabla.to_string(index=False))
    log.info("Mejor: %s", mejor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
