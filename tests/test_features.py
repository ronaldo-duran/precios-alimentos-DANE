"""Construcción de features: corrección y, sobre todo, ausencia de leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.features.build import (
    CATEGORICAS,
    anadir_objetivos,
    columnas_features,
    construir_features,
    filas_a_predecir,
    filas_entrenables,
    objetivo_log_cambio,
)


def panel_sintetico(n: int = 60, n_series: int = 2, semilla: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    semanas = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    filas = []
    for i in range(n_series):
        precios = 2000 * np.exp(np.cumsum(0.03 * rng.standard_normal(n)))
        filas.append(
            pd.DataFrame(
                {
                    "semana": semanas,
                    "producto_id": f"prod{i}",
                    "plaza_id": "plaza0",
                    "producto": f"Prod {i}",
                    "plaza": "Plaza",
                    "ciudad": "C",
                    "grupo": "G",
                    "precio_kg": precios,
                    "precio_kg_min": precios * 0.95,
                    "precio_kg_max": precios * 1.05,
                    "n_dias": 6,
                    "flag_outlier": False,
                    "flag_pocos_dias": False,
                    "flag_faltante": False,
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


def test_los_identificadores_son_features_categoricas():
    F = construir_features(panel_sintetico())
    cols = columnas_features(F)
    for c in CATEGORICAS:
        assert c in cols
        assert isinstance(F[c].dtype, pd.CategoricalDtype)


def test_las_columnas_objetivo_no_entran_como_features():
    F = anadir_objetivos(construir_features(panel_sintetico()), [1, 2])
    cols = columnas_features(F)
    assert not [c for c in cols if c.startswith("target_")]
    assert "y" not in cols and "semana" not in cols and "t" not in cols


def test_el_indice_t_es_de_calendario_y_comparable_entre_series():
    F = construir_features(panel_sintetico(n=30, n_series=2))
    por_serie = F.groupby(list(CATEGORICAS), observed=True)["t"].agg(["min", "max"])
    assert por_serie["min"].nunique() == 1
    assert por_serie["max"].nunique() == 1


def test_una_serie_que_empieza_mas_tarde_no_desalinea_el_indice():
    panel = panel_sintetico(n=30, n_series=2)
    tardia = panel["producto_id"] == "prod1"
    panel = panel[~tardia | (panel["semana"] >= panel["semana"].min() + pd.Timedelta(70, unit="D"))]

    F = construir_features(panel)
    # Para una misma semana calendario, t debe coincidir en ambas series.
    semana = F["semana"].max()
    ts = F[F["semana"] == semana]["t"].unique()
    assert len(ts) == 1


def test_el_objetivo_es_el_log_cambio_a_h_semanas():
    F = construir_features(panel_sintetico(n=20, n_series=1))
    obj = objetivo_log_cambio(F, 3)
    y = F["y"].to_numpy()
    esperado = np.log(y[3] / y[0])
    assert obj.iloc[0] == pytest.approx(esperado)
    # Las últimas h filas no tienen futuro observable.
    assert obj.tail(3).isna().all()


def test_el_objetivo_no_cruza_series():
    F = construir_features(panel_sintetico(n=20, n_series=2))
    obj = objetivo_log_cambio(F, 1)
    fin_primera = F[F["producto_id"] == "prod0"].index[-1]
    assert pd.isna(obj.loc[fin_primera])


def test_toda_feature_en_t_solo_usa_informacion_hasta_t():
    """Corromper el futuro no puede alterar ninguna feature de la fila t."""
    panel = panel_sintetico(n=60, n_series=1)
    corte = 40

    F_limpio = construir_features(panel)
    panel_corrupto = panel.copy()
    futuro = panel_corrupto.index >= corte
    panel_corrupto.loc[futuro, "precio_kg"] *= 1000
    panel_corrupto.loc[futuro, "precio_kg_min"] *= 1000
    panel_corrupto.loc[futuro, "precio_kg_max"] *= 1000
    F_corrupto = construir_features(panel_corrupto)

    cols = columnas_features(F_limpio)
    a = F_limpio[F_limpio["t"] < corte][cols].reset_index(drop=True)
    b = F_corrupto[F_corrupto["t"] < corte][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_filas_entrenables_excluye_objetivos_posteriores_al_origen():
    F = anadir_objetivos(construir_features(panel_sintetico(n=60, n_series=1)), [1, 4])
    origen = 30
    for h in (1, 4):
        train = filas_entrenables(F, origen, h)
        # La fila t predice t+h; ese objetivo debe estar dentro de lo conocido.
        assert (train["t"] + h <= origen - 1).all()
        assert train["t"].max() == origen - 1 - h


def test_filas_a_predecir_toma_la_ultima_semana_conocida():
    F = construir_features(panel_sintetico(n=60, n_series=3))
    origen = 30
    pred = filas_a_predecir(F, origen)
    assert len(pred) == 3  # una fila por serie
    assert set(pred["t"]) == {origen - 1}


def test_las_semanas_faltantes_quedan_como_nan_sin_interpolar():
    panel = panel_sintetico(n=40, n_series=1)
    panel.loc[20, "precio_kg"] = np.nan

    F = construir_features(panel)
    assert pd.isna(F.loc[20, "y"])
    assert pd.isna(F.loc[20, "log_precio"])


def test_semana_sin_y_cos_hacen_contiguas_la_52_y_la_1():
    F = construir_features(panel_sintetico(n=110, n_series=1))
    s52 = F[F["semana_anio"] == 52].iloc[0]
    s1 = F[F["semana_anio"] == 1].iloc[0]
    distancia = np.hypot(s52["semana_sin"] - s1["semana_sin"], s52["semana_cos"] - s1["semana_cos"])
    assert distancia < 0.3
