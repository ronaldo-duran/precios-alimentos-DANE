"""Búsqueda de hiperparámetros: que sea limitada, registrada y sin contaminar.

Lo que estos tests protegen no es la calidad de la configuración elegida, sino
la **higiene del procedimiento**: rejilla fija, orígenes de búsqueda separados
de los de evaluación, y registro completo de lo probado.
"""

from __future__ import annotations

import pandas as pd
import pytest

from precios.evaluation.splits import rolling_origins
from precios.features.build import anadir_objetivos, construir_features
from precios.models.busqueda import REJILLA, buscar, evaluar_config, partir_origenes

from .test_features import panel_sintetico


def test_la_rejilla_es_pequena_y_esta_fijada_de_antemano():
    assert 3 <= len(REJILLA) <= 12
    # Todas las configuraciones exploran las mismas claves: comparación limpia.
    claves = {frozenset(c) for c in REJILLA}
    assert len(claves) == 1


def test_los_origenes_de_busqueda_y_evaluacion_no_se_solapan():
    folds = rolling_origins(300, min_train=104, horizonte_max=4, paso=4)
    busqueda, evaluacion = partir_origenes(folds)
    assert busqueda and evaluacion
    assert max(f.origen for f in busqueda) < min(f.origen for f in evaluacion)


def test_la_busqueda_usa_los_origenes_antiguos_y_la_evaluacion_los_recientes():
    folds = rolling_origins(300, min_train=104, horizonte_max=4, paso=4)
    busqueda, evaluacion = partir_origenes(folds)
    assert len(busqueda) + len(evaluacion) == len(folds)
    assert busqueda[0].origen == folds[0].origen
    assert evaluacion[-1].origen == folds[-1].origen


@pytest.fixture
def features_pequenas() -> pd.DataFrame:
    panel = panel_sintetico(n=140, n_series=4, semilla=2)
    return anadir_objetivos(construir_features(panel), [1])


def test_evaluar_config_devuelve_una_fila_por_serie_y_origen(features_pequenas):
    folds = rolling_origins(140, min_train=110, horizonte_max=1, paso=10)
    out = evaluar_config(
        features_pequenas,
        {"num_leaves": 7, "n_estimators": 30, "min_child_samples": 20, "reg_lambda": 1.0},
        folds,
        horizontes=[1],
    )
    assert not out.empty
    assert set(out.columns) >= {"producto_id", "plaza_id", "h", "y_pred", "y_true", "origen"}
    assert out["y_true"].notna().all()


def test_buscar_registra_todas_las_configuraciones_probadas(features_pequenas):
    folds = rolling_origins(140, min_train=120, horizonte_max=1, paso=10)
    rejilla = (
        {"num_leaves": 7, "n_estimators": 30, "min_child_samples": 20, "reg_lambda": 1.0},
        {"num_leaves": 15, "n_estimators": 30, "min_child_samples": 40, "reg_lambda": 5.0},
    )
    mejor, tabla = buscar(features_pequenas, folds, horizontes=[1], rejilla=rejilla)

    assert len(tabla) == len(rejilla)  # se registran todas, no solo la ganadora
    assert set(mejor) == set(rejilla[0])
    # La tabla viene ordenada por MASE y la mejor es la primera.
    assert tabla["mase"].is_monotonic_increasing
    assert mejor["num_leaves"] == int(tabla.loc[0, "num_leaves"])


def test_buscar_devuelve_enteros_donde_lightgbm_los_exige(features_pequenas):
    folds = rolling_origins(140, min_train=120, horizonte_max=1, paso=10)
    rejilla = (
        {"num_leaves": 7, "n_estimators": 30, "min_child_samples": 20, "reg_lambda": 1.0},
    )
    mejor, _ = buscar(features_pequenas, folds, horizontes=[1], rejilla=rejilla)
    for clave in ("num_leaves", "n_estimators", "min_child_samples"):
        assert isinstance(mejor[clave], int)


def test_la_busqueda_es_reproducible_con_la_misma_semilla(features_pequenas):
    folds = rolling_origins(140, min_train=120, horizonte_max=1, paso=10)
    rejilla = (
        {"num_leaves": 7, "n_estimators": 30, "min_child_samples": 20, "reg_lambda": 1.0},
    )
    _, a = buscar(features_pequenas, folds, horizontes=[1], rejilla=rejilla, semilla=7)
    _, b = buscar(features_pequenas, folds, horizontes=[1], rejilla=rejilla, semilla=7)
    assert a.loc[0, "mase"] == pytest.approx(b.loc[0, "mase"])


def test_buscar_falla_si_ninguna_config_produce_resultados(features_pequenas):
    folds = rolling_origins(140, min_train=139, horizonte_max=1, paso=10)
    rejilla = ({"num_leaves": 7, "n_estimators": 10, "min_child_samples": 20, "reg_lambda": 1.0},)
    # Con un solo origen y muy pocas filas el modelo se omite por falta de datos.
    features_minimas = features_pequenas[features_pequenas["t"] > 130]
    with pytest.raises(RuntimeError, match="ninguna configuración"):
        buscar(features_minimas, folds, horizontes=[1], rejilla=rejilla)
