"""Ausencia de data leakage.

La propiedad que se comprueba: **nada de lo que ocurre en la semana del origen
o después puede cambiar el pronóstico**. Se verifica corrompiendo el futuro y
exigiendo que las predicciones no se muevan ni un decimal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
