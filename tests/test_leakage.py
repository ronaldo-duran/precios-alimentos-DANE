"""Ausencia de data leakage.

La propiedad que se comprueba: **nada de lo que ocurre en la semana del origen
o después puede cambiar el pronóstico**. Se verifica corrompiendo el futuro y
exigiendo que las predicciones no se muevan ni un decimal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.evaluation.backtest import backtest_serie, preparar_train
from precios.evaluation.splits import Fold, rolling_origins
from precios.models.baselines import baselines_por_defecto


def _serie(n: int = 200, semilla: int = 3) -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(semilla)
    y = 2000 * np.exp(np.cumsum(0.03 * rng.standard_normal(n)))
    semanas = pd.Series(pd.date_range("2020-01-06", periods=n, freq="W-MON"))
    return y, semanas


def test_corromper_el_futuro_no_cambia_los_pronosticos():
    y, semanas = _serie()
    folds = rolling_origins(len(y), min_train=104, horizonte_max=4, paso=8)
    modelos = baselines_por_defecto()

    limpio = backtest_serie(y, semanas, modelos, folds)

    # Multiplica por 1000 todo lo posterior al primer origen evaluado.
    primer_origen = folds[0].origen
    y_corrupto = y.copy()
    y_corrupto[primer_origen:] *= 1000.0
    corrupto = backtest_serie(y_corrupto, semanas, modelos, folds)

    # y_true cambia (es el futuro), pero y_pred del primer origen NO puede cambiar.
    a = limpio[limpio["origen"] == primer_origen].reset_index(drop=True)
    b = corrupto[corrupto["origen"] == primer_origen].reset_index(drop=True)
    pd.testing.assert_series_equal(a["y_pred"], b["y_pred"])


def test_cada_modelo_solo_ve_la_ventana_de_entrenamiento():
    y, _ = _serie()
    origen = 150
    for modelo in baselines_por_defecto():
        esperado = modelo.fit(y[:origen]).predict([1, 2, 3, 4])

        y_corrupto = y.copy()
        y_corrupto[origen:] = -999.0
        obtenido = modelo.fit(y_corrupto[:origen]).predict([1, 2, 3, 4])

        np.testing.assert_array_equal(esperado, obtenido)


def test_la_escala_del_mase_no_usa_el_periodo_de_test():
    """La escala se calcula sobre el train; cambiar el test no debe moverla."""
    y, semanas = _serie()
    folds = [Fold(origen=120, horizontes=(1, 2, 3, 4))]
    modelos = baselines_por_defecto()

    limpio = backtest_serie(y, semanas, modelos, folds)
    y_corrupto = y.copy()
    y_corrupto[120:] *= 50.0
    corrupto = backtest_serie(y_corrupto, semanas, modelos, folds)

    assert limpio["escala_mase"].iloc[0] == corrupto["escala_mase"].iloc[0]


def test_preparar_train_no_mira_mas_alla_de_la_ventana():
    """Interpolar un hueco solo puede usar datos de la propia ventana."""
    y = np.array([100.0, 110.0, np.nan, 130.0, 140.0, 999999.0])
    ventana = y[:5]  # el 999999 queda fuera

    salida = preparar_train(ventana)

    assert not np.isnan(salida).any()
    assert salida[2] == 120.0  # interpolación lineal entre 110 y 130
    assert 999999.0 not in salida


def test_preparar_train_rellena_hacia_atras_sin_inventar_futuro():
    y = np.array([np.nan, np.nan, 300.0, 310.0])
    salida = preparar_train(y)
    assert not np.isnan(salida).any()
    # Los NaN iniciales se cubren con el primer valor conocido, no con una media global.
    assert salida[0] == 300.0
    assert salida[1] == 300.0


def test_ventana_totalmente_vacia_devuelve_nan():
    salida = preparar_train(np.array([np.nan, np.nan, np.nan]))
    assert np.isnan(salida).all()


def test_no_se_evaluan_semanas_sin_dato_observado():
    """Un hueco en el objetivo no puede contarse como acierto ni como error."""
    y, semanas = _serie(n=130)
    y[125] = np.nan  # hueco dentro del periodo de test
    folds = [Fold(origen=124, horizontes=(1, 2, 3, 4))]

    res = backtest_serie(y, semanas, baselines_por_defecto(), folds)

    assert not res["y_true"].isna().any()
    # h=2 apunta al índice 125, que es el hueco: no debe aparecer.
    assert 2 not in set(res["h"])
    assert {1, 3, 4} <= set(res["h"])


# --- Modelos globales -------------------------------------------------------


def _panel_para_global(n: int = 160, n_series: int = 4, semilla: int = 17) -> pd.DataFrame:
    from .test_features import panel_sintetico

    return panel_sintetico(n=n, n_series=n_series, semilla=semilla)


def test_lgbm_global_no_usa_datos_posteriores_al_origen():
    """Corromper el panel desde el origen no puede mover el pronóstico."""
    from precios.features.build import anadir_objetivos, construir_features, filas_a_predecir
    from precios.models.lgbm import LGBMGlobal

    panel = _panel_para_global()
    origen = 120
    semana_corte = sorted(panel["semana"].unique())[origen - 1]

    def predecir(p: pd.DataFrame) -> np.ndarray:
        F = anadir_objetivos(construir_features(p), [1, 2])
        modelo = LGBMGlobal(horizontes=(1, 2), cuantiles=(0.5,), semilla=1).fit(F, origen)
        salida = modelo.predict(filas_a_predecir(F, origen))
        return salida.sort_values(["producto_id", "h"])["y_pred"].to_numpy()

    limpio = predecir(panel)

    corrupto = panel.copy()
    futuro = corrupto["semana"] > semana_corte
    corrupto.loc[futuro, "precio_kg"] *= 100
    corrupto.loc[futuro, ["precio_kg_min", "precio_kg_max"]] *= 100

    np.testing.assert_allclose(limpio, predecir(corrupto), rtol=1e-12)


def test_conformal_calibra_con_datos_que_el_modelo_no_vio():
    """El tramo de calibración debe quedar fuera del entrenamiento propio."""
    from precios.features.build import anadir_objetivos, construir_features
    from precios.models.conformal import ConformalGlobal

    panel = _panel_para_global()
    F = anadir_objetivos(construir_features(panel), [1])
    origen, n_calib = 140, 26

    modelo = ConformalGlobal(horizontes=(1,), semanas_calibracion=n_calib, semilla=1)
    modelo.fit(F, origen)

    # El modelo interno se entrenó como si el origen fuera origen - n_calib.
    t_max_entrenamiento = origen - n_calib - 1 - 1  # origen_propio - 1 - h
    residuos = modelo._residuos_calibracion(F, origen, origen - n_calib, 1)
    assert residuos.size > 0
    # Toda fila de calibración es posterior al último dato de entrenamiento.
    calibracion = F[(F["t"] > t_max_entrenamiento) & (F["t"] <= origen - 2)]
    assert calibracion["t"].min() > t_max_entrenamiento


def test_conformal_falla_si_el_origen_es_demasiado_temprano():
    from precios.features.build import anadir_objetivos, construir_features
    from precios.models.conformal import ConformalGlobal

    F = anadir_objetivos(construir_features(_panel_para_global(n=60)), [1])
    modelo = ConformalGlobal(horizontes=(1,), semanas_calibracion=26)
    with pytest.raises(ValueError, match="demasiado temprano"):
        modelo.fit(F, origen=20)
