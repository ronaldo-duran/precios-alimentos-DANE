"""Métricas de pronóstico y su comportamiento en los casos límite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.evaluation.metrics import (
    amplitud_relativa,
    cobertura,
    escala_mase,
    mae,
    mase,
    resumir,
    rmse,
    smape,
)


def test_mae_y_rmse_en_un_caso_a_mano():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    assert mae(y, p) == pytest.approx(7 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))


def test_la_escala_del_mase_es_el_mae_del_naive_en_el_train():
    y = np.array([1.0, 3.0, 6.0, 10.0])  # diferencias: 2, 3, 4
    assert escala_mase(y) == pytest.approx(3.0)


def test_la_escala_estacional_usa_el_periodo_indicado():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Con periodo 2: |3-1|, |4-2|, |5-3|, |6-4| = 2 en todos los casos.
    assert escala_mase(y, periodo=2) == pytest.approx(2.0)


def test_una_serie_constante_no_da_escala_cero():
    """Escala 0 volvería infinito cualquier MASE; se devuelve NaN."""
    assert np.isnan(escala_mase(np.array([5.0, 5.0, 5.0, 5.0])))


def test_serie_mas_corta_que_el_periodo_da_nan():
    assert np.isnan(escala_mase(np.array([1.0, 2.0]), periodo=5))


def test_mase_vale_uno_cuando_el_error_iguala_la_escala():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([13.0, 7.0])  # error absoluto medio = 3
    assert mase(y_true, y_pred, escala=3.0) == pytest.approx(1.0)


def test_mase_menor_que_uno_significa_mejor_que_el_naive():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([11.0, 9.0])  # error medio = 1
    assert mase(y_true, y_pred, escala=3.0) < 1.0


def test_mase_con_escala_invalida_es_nan():
    assert np.isnan(mase(np.array([1.0]), np.array([2.0]), escala=float("nan")))
    assert np.isnan(mase(np.array([1.0]), np.array([2.0]), escala=0.0))


def test_smape_es_simetrico():
    a = smape(np.array([100.0]), np.array([110.0]))
    b = smape(np.array([110.0]), np.array([100.0]))
    assert a == pytest.approx(b)


def test_smape_ignora_los_puntos_indefinidos():
    """Con y_true y y_pred ambos 0 el sMAPE es 0/0: se omite, no se cuenta como acierto."""
    y = np.array([0.0, 100.0])
    p = np.array([0.0, 110.0])
    solo_valido = smape(np.array([100.0]), np.array([110.0]))
    assert smape(y, p) == pytest.approx(solo_valido)


def test_cobertura_cuenta_los_valores_dentro_del_intervalo():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([1.5, 2.5, 2.5, 3.5])  # cubre los dos primeros
    assert cobertura(y, lo, hi) == pytest.approx(50.0)


def test_cobertura_incluye_los_bordes():
    assert cobertura(np.array([5.0]), np.array([5.0]), np.array([5.0])) == 100.0


def test_amplitud_relativa_se_expresa_como_porcentaje():
    y = np.array([100.0, 200.0])
    lo = np.array([90.0, 180.0])
    hi = np.array([110.0, 220.0])  # anchos 20 y 40 -> 20% en ambos
    assert amplitud_relativa(y, lo, hi) == pytest.approx(20.0)


def _resultados() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "modelo": ["naive"] * 4 + ["otro"] * 4,
            "h": [1, 1, 2, 2] * 2,
            "y_true": [100.0, 100.0, 100.0, 100.0] * 2,
            "y_pred": [110.0, 90.0, 120.0, 80.0, 105.0, 95.0, 110.0, 90.0],
            "escala_mase": [10.0] * 8,
        }
    )


def test_resumir_agrupa_por_modelo_y_horizonte():
    out = resumir(_resultados(), por=["modelo", "h"])
    assert len(out) == 4
    assert set(out.columns) >= {"modelo", "h", "n", "mae", "rmse", "mase", "smape"}


def test_resumir_calcula_el_mase_esperado():
    out = resumir(_resultados(), por=["modelo", "h"])
    fila = out[(out["modelo"] == "naive") & (out["h"] == 1)].iloc[0]
    assert fila["mase"] == pytest.approx(1.0)  # error 10 / escala 10


def test_resumir_anade_cobertura_si_hay_intervalos():
    df = _resultados()
    df["lo"] = df["y_pred"] - 50
    df["hi"] = df["y_pred"] + 50
    out = resumir(df, por=["modelo", "h"])
    assert "cobertura_pct" in out.columns
    assert out["cobertura_pct"].iloc[0] == pytest.approx(100.0)


def test_resumir_exige_las_columnas_minimas():
    with pytest.raises(ValueError, match="Faltan columnas"):
        resumir(pd.DataFrame({"modelo": ["a"], "h": [1]}))
