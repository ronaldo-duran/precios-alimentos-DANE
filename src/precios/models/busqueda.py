"""Búsqueda limitada y registrada de hiperparámetros.

Reglas que se respetan a propósito, porque son la diferencia entre medir y
engañarse:

1. **La búsqueda usa orígenes distintos de los de la evaluación final.** Los
   orígenes se parten en dos mitades: la primera sirve para elegir la
   configuración, la segunda para reportarla. Buscar y reportar sobre los mismos
   orígenes produce una mejora que no existe fuera de la muestra.
2. **La rejilla es pequeña y se fija de antemano** (`REJILLA`). No se amplía
   hasta que algún resultado guste.
3. **Se registran todas las configuraciones probadas**, no solo la ganadora, y
   el registro se escribe a disco para poder auditarlo.
4. **Semilla fija** en todas las corridas.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from precios.evaluation.metrics import escala_mase
from precios.evaluation.splits import Fold
from precios.features.build import filas_a_predecir
from precios.models.lgbm import LGBMGlobal

log = logging.getLogger(__name__)

#: Rejilla fija. Seis configuraciones que exploran capacidad (hojas, árboles)
#: y regularización (muestras mínimas, lambda). No se toca sin dejar registro.
REJILLA: tuple[dict, ...] = (
    {"num_leaves": 15, "n_estimators": 300, "min_child_samples": 60, "reg_lambda": 1.0},
    {"num_leaves": 31, "n_estimators": 400, "min_child_samples": 40, "reg_lambda": 1.0},
    {"num_leaves": 31, "n_estimators": 400, "min_child_samples": 20, "reg_lambda": 0.1},
    {"num_leaves": 63, "n_estimators": 400, "min_child_samples": 40, "reg_lambda": 5.0},
    {"num_leaves": 15, "n_estimators": 800, "min_child_samples": 40, "reg_lambda": 1.0},
    {"num_leaves": 31, "n_estimators": 200, "min_child_samples": 80, "reg_lambda": 10.0},
)


def partir_origenes(folds: Sequence[Fold]) -> tuple[list[Fold], list[Fold]]:
    """Divide los orígenes en mitad de búsqueda (antigua) y de evaluación (reciente)."""
    corte = len(folds) // 2
    return list(folds[:corte]), list(folds[corte:])


def evaluar_config(
    features: pd.DataFrame,
    params: dict,
    folds: Sequence[Fold],
    *,
    horizontes: Sequence[int],
    semilla: int = 42,
) -> pd.DataFrame:
    """Corre una configuración sobre los orígenes dados y devuelve sus errores.

    Solo entrena el cuantil 0,5: la búsqueda decide sobre el pronóstico puntual,
    no sobre los intervalos, que se calibran después.
    """
    reales = features.set_index(["producto_id", "plaza_id", "t"])["y"]
    filas = []

    for fold in folds:
        modelo = LGBMGlobal(
            horizontes=horizontes, cuantiles=(0.5,), params=params, semilla=semilla
        )
        try:
            modelo.fit(features, fold.origen)
            pred = modelo.predict(filas_a_predecir(features, fold.origen))
        except (ValueError, RuntimeError) as err:
            log.debug("config omitida en origen %d: %s", fold.origen, err)
            continue

        pred = pred.copy()
        pred["t_objetivo"] = fold.origen + pred["h"] - 1
        claves = list(
            zip(pred["producto_id"], pred["plaza_id"], pred["t_objetivo"], strict=True)
        )
        pred["y_true"] = [reales.get(k, np.nan) for k in claves]
        pred["origen"] = fold.origen
        filas.append(pred.dropna(subset=["y_true"]))

    if not filas:
        return pd.DataFrame()
    return pd.concat(filas, ignore_index=True)


def buscar(
    features: pd.DataFrame,
    folds_busqueda: Sequence[Fold],
    *,
    horizontes: Sequence[int],
    rejilla: Sequence[dict] = REJILLA,
    semilla: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Prueba la rejilla completa y devuelve (mejor config, registro de todas).

    El criterio es el MAE del log-cambio, promediado sobre horizontes: es la
    misma pérdida que optimiza el cuantil 0,5, así que la selección es coherente
    con el entrenamiento y no depende de la escala de cada producto.
    """
    escalas = _escalas(features, [f.origen for f in folds_busqueda])
    registro = []

    for i, params in enumerate(rejilla, 1):
        resultados = evaluar_config(
            features, params, folds_busqueda, horizontes=horizontes, semilla=semilla
        )
        if resultados.empty:
            log.warning("Config %d no produjo resultados: %s", i, params)
            continue

        error_abs = (resultados["y_true"] - resultados["y_pred"]).abs()
        escala = [
            escalas.get((prod, plaza, o), np.nan)
            for prod, plaza, o in zip(
                resultados["producto_id"],
                resultados["plaza_id"],
                resultados["origen"],
                strict=True,
            )
        ]
        with np.errstate(divide="ignore", invalid="ignore"):
            mase_punto = error_abs.to_numpy() / np.asarray(escala, dtype=float)

        fila = dict(params)
        fila["n"] = len(resultados)
        fila["mase"] = float(np.nanmean(mase_punto))
        fila["mae"] = float(error_abs.mean())
        registro.append(fila)
        log.info("Config %d/%d  MASE=%.4f  %s", i, len(rejilla), fila["mase"], params)

    if not registro:
        raise RuntimeError("La búsqueda no produjo ninguna configuración válida")

    tabla = pd.DataFrame(registro).sort_values("mase").reset_index(drop=True)
    mejor = {k: v for k, v in tabla.iloc[0].items() if k in REJILLA[0]}
    # LightGBM espera enteros donde la tabla dejó floats.
    for clave in ("num_leaves", "n_estimators", "min_child_samples"):
        if clave in mejor:
            mejor[clave] = int(mejor[clave])
    log.info("Mejor configuración: %s (MASE %.4f)", mejor, tabla.loc[0, "mase"])
    return mejor, tabla


def _escalas(
    features: pd.DataFrame, origenes: Sequence[int]
) -> dict[tuple[str, str, int], float]:
    """Escala del MASE por serie y origen (igual criterio que el backtest)."""
    from precios.evaluation.backtest import preparar_train

    salida: dict[tuple[str, str, int], float] = {}
    for (prod, plaza), g in features.groupby(["producto_id", "plaza_id"], observed=True):
        y = g.sort_values("t")["y"].to_numpy(dtype=float)
        for o in origenes:
            salida[(prod, plaza, o)] = escala_mase(preparar_train(y[:o]), periodo=1)
    return salida
