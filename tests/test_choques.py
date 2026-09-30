"""Detección de episodios de choque y medición de la degradación del error."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.evaluation.choques import (
    degradacion_por_regimen,
    etiquetar_con_episodios,
    marcar_semanas_de_choque,
)


def panel_con_choque(n: int = 100, n_series: int = 10, semana_choque: int = 50) -> pd.DataFrame:
    """Panel plano salvo una semana en la que TODAS las series saltan a la vez."""
    rng = np.random.default_rng(9)
    semanas = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    filas = []
    for i in range(n_series):
        precios = 2000 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
        precios[semana_choque:] *= 1.6  # salto del 60% que persiste
        filas.append(
            pd.DataFrame(
                {
                    "semana": semanas,
                    "producto_id": f"p{i}",
                    "plaza_id": "z",
                    "precio_kg": precios,
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


def test_detecta_la_semana_en_la_que_salta_todo_el_panel():
    panel = panel_con_choque()
    semanas = marcar_semanas_de_choque(panel, cuantil=0.95, min_series=5)
    choques = semanas[semanas["es_choque"]]
    semana_esperada = sorted(panel["semana"].unique())[50]
    assert semana_esperada in set(choques["semana"])


def test_una_sola_serie_enloquecida_no_marca_la_semana():
    """Se usa la mediana justamente para esto: un choque mueve el panel entero."""
    panel = panel_con_choque(semana_choque=10_000)  # sin choque real
    objetivo = sorted(panel["semana"].unique())[50]
    una = (panel["producto_id"] == "p0") & (panel["semana"] >= objetivo)
    panel.loc[una, "precio_kg"] *= 100

    semanas = marcar_semanas_de_choque(panel, cuantil=0.95, min_series=5)
    fila = semanas[semanas["semana"] == objetivo].iloc[0]
    assert not fila["es_choque"]


def test_las_semanas_con_pocas_series_no_se_clasifican():
    panel = panel_con_choque(n_series=3)
    semanas = marcar_semanas_de_choque(panel, cuantil=0.90, min_series=5)
    assert not semanas["es_choque"].any()


def test_el_umbral_sale_del_cuantil_pedido():
    panel = panel_con_choque()
    laxo = marcar_semanas_de_choque(panel, cuantil=0.50, min_series=5)
    estricto = marcar_semanas_de_choque(panel, cuantil=0.99, min_series=5)
    assert laxo["es_choque"].sum() > estricto["es_choque"].sum()


def test_etiquetar_asigna_el_episodio_por_rango_de_fechas():
    semanas = pd.DataFrame({"semana": pd.date_range("2021-04-01", periods=10, freq="W-MON")})
    episodios = pd.DataFrame(
        [
            {
                "id": "paro",
                "nombre": "Paro nacional",
                "desde": pd.Timestamp("2021-04-28"),
                "hasta": pd.Timestamp("2021-05-20"),
                "nota": "",
            }
        ]
    )
    out = etiquetar_con_episodios(semanas, episodios)
    dentro = out[out["semana"].between("2021-04-28", "2021-05-20")]
    assert (dentro["episodio"] == "Paro nacional").all()
    fuera = out[~out["semana"].between("2021-04-28", "2021-05-20")]
    assert fuera["episodio"].isna().all()


def _backtest_sintetico() -> tuple[pd.DataFrame, pd.DataFrame]:
    semanas_lista = pd.date_range("2024-01-01", periods=20, freq="W-MON")
    choque = semanas_lista[10]
    bt = pd.DataFrame(
        {
            "modelo": "m",
            "h": 1,
            "semana_objetivo": semanas_lista,
            # Error de 10 en todas, salvo 100 en la semana de choque.
            "y_true": [100.0] * 20,
            "y_pred": [110.0 if s != choque else 200.0 for s in semanas_lista],
            "escala_mase": 10.0,
        }
    )
    semanas = pd.DataFrame({"semana": semanas_lista, "es_choque": semanas_lista == choque})
    return bt, semanas


def test_la_degradacion_separa_choque_de_normal():
    bt, semanas = _backtest_sintetico()
    out = degradacion_por_regimen(bt, semanas)
    assert set(out["regimen"]) == {"choque", "normal"}
    normal = out[out["regimen"] == "normal"].iloc[0]
    choque = out[out["regimen"] == "choque"].iloc[0]
    assert normal["mase"] == pytest.approx(1.0)  # error 10 / escala 10
    assert choque["mase"] == pytest.approx(10.0)  # error 100 / escala 10


def test_el_factor_de_degradacion_es_el_cociente():
    bt, semanas = _backtest_sintetico()
    out = degradacion_por_regimen(bt, semanas)
    assert out["factor_degradacion"].dropna().unique()[0] == pytest.approx(10.0)


def test_la_degradacion_mira_la_semana_objetivo_no_la_del_origen():
    """Lo que importa es si falla cuando ocurre el choque, no cuándo se pronosticó."""
    bt, semanas = _backtest_sintetico()
    bt["semana_origen"] = bt["semana_objetivo"] - pd.Timedelta(7, unit="D")
    out = degradacion_por_regimen(bt, semanas)
    assert out[out["regimen"] == "choque"]["n"].iloc[0] == 1


def test_la_cobertura_se_reporta_por_regimen_si_hay_intervalos():
    bt, semanas = _backtest_sintetico()
    bt["lo"] = bt["y_pred"] - 5
    bt["hi"] = bt["y_pred"] + 5
    out = degradacion_por_regimen(bt, semanas)
    assert "cobertura_pct" in out.columns
    # Ningún intervalo cubre: el pronóstico está a 10 (o 100) del valor real.
    assert out["cobertura_pct"].max() == 0.0
