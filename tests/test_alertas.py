"""Alertas de alza: umbral por producto y probabilidad calibrada."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.models.alertas import derivar_umbrales, probabilidad_alza, umbral_de


def test_el_umbral_sale_del_config_por_producto():
    cfg = {"umbrales": {"tomate": 0.40, "yuca": 0.10}, "umbral_por_defecto": 0.15}
    assert umbral_de("tomate", cfg) == 0.40
    assert umbral_de("yuca", cfg) == 0.10


def test_un_producto_sin_umbral_propio_usa_el_por_defecto():
    cfg = {"umbrales": {"tomate": 0.40}, "umbral_por_defecto": 0.15}
    assert umbral_de("lechuga", cfg) == 0.15


def test_la_probabilidad_crece_cuando_el_pronostico_apunta_al_alza():
    residuos = {1: np.random.default_rng(1).normal(0, 0.1, 2000)}
    baja = probabilidad_alza({1: 0.0}, residuos, 0.10, [1])
    alta = probabilidad_alza({1: 0.20}, residuos, 0.10, [1])
    assert alta > baja


def test_la_probabilidad_cae_cuando_sube_el_umbral():
    residuos = {1: np.random.default_rng(2).normal(0, 0.15, 2000)}
    p10 = probabilidad_alza({1: 0.0}, residuos, 0.10, [1])
    p40 = probabilidad_alza({1: 0.0}, residuos, 0.40, [1])
    assert p10 > p40


def test_con_residuos_simetricos_y_pronostico_plano_el_umbral_cero_da_la_mitad():
    residuos = {1: np.random.default_rng(3).normal(0, 0.1, 20000)}
    p = probabilidad_alza({1: 0.0}, residuos, 0.0, [1])
    assert p == pytest.approx(0.5, abs=0.02)


def test_la_probabilidad_esta_entre_cero_y_uno():
    residuos = {1: np.random.default_rng(4).normal(0, 0.1, 1000)}
    for umbral in (0.0, 0.05, 0.5, 5.0):
        p = probabilidad_alza({1: 0.1}, residuos, umbral, [1])
        assert 0.0 <= p <= 1.0


def test_dos_horizontes_no_dan_menos_probabilidad_que_uno():
    """Dos oportunidades de superar el umbral no pueden ser menos que una."""
    rng = np.random.default_rng(5)
    residuos = {1: rng.normal(0, 0.1, 4000), 2: rng.normal(0, 0.15, 4000)}
    solo_h1 = probabilidad_alza({1: 0.05, 2: 0.05}, residuos, 0.10, [1])
    ambos = probabilidad_alza({1: 0.05, 2: 0.05}, residuos, 0.10, [1, 2])
    assert ambos >= solo_h1


def test_el_acople_comonotonico_no_infla_como_la_independencia():
    """Suponer independencia sobreestimaría: los errores están correlacionados."""
    rng = np.random.default_rng(6)
    residuos = {1: rng.normal(0, 0.1, 8000), 2: rng.normal(0, 0.1, 8000)}
    pred = {1: 0.0, 2: 0.0}
    p = probabilidad_alza(pred, residuos, 0.10, [1, 2])

    p1 = probabilidad_alza(pred, {1: residuos[1]}, 0.10, [1])
    independiente = 1 - (1 - p1) ** 2
    assert p < independiente


def test_sin_residuos_la_probabilidad_es_nan():
    assert np.isnan(probabilidad_alza({1: 0.1}, {1: np.array([])}, 0.10, [1]))


def test_sin_pronostico_para_el_horizonte_la_probabilidad_es_nan():
    residuos = {1: np.random.default_rng(7).normal(0, 0.1, 100)}
    assert np.isnan(probabilidad_alza({}, residuos, 0.10, [1]))


def _panel_con_volatilidades_distintas() -> pd.DataFrame:
    """Dos productos: uno tranquilo y otro mucho más volátil.

    El contraste tiene que ser grande para que el redondeo al 5% y el piso del
    10% no los aplasten en el mismo umbral, que es justo lo que pasa con la
    volatilidad real de yuca frente a tomate.
    """
    rng = np.random.default_rng(11)
    semanas = pd.date_range("2024-01-01", periods=300, freq="W-MON")
    filas = []
    for prod, sigma in (("tranquilo", 0.02), ("volatil", 0.18)):
        precios = 2000 * np.exp(np.cumsum(sigma * rng.standard_normal(300)))
        filas.append(
            pd.DataFrame(
                {
                    "semana": semanas,
                    "producto_id": prod,
                    "plaza_id": "z",
                    "precio_kg": precios,
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


def test_el_producto_volatil_recibe_un_umbral_mas_alto():
    tabla = derivar_umbrales(
        _panel_con_volatilidades_distintas(),
        {"horizonte_semanas": 2, "derivacion": {"cuantil": 0.75, "redondeo": 0.05, "piso": 0.10}},
    )
    umbrales = tabla.set_index("producto_id")["umbral"]
    assert umbrales["volatil"] > umbrales["tranquilo"]


def test_los_umbrales_igualan_la_tasa_base_entre_productos():
    """Ese es el punto de tener umbral por producto."""
    tabla = derivar_umbrales(
        _panel_con_volatilidades_distintas(),
        {"horizonte_semanas": 2, "derivacion": {"cuantil": 0.75, "redondeo": 0.05, "piso": 0.10}},
    )
    tasas = tabla["tasa_base_pct"]
    assert tasas.max() - tasas.min() < 15


def test_el_umbral_nunca_baja_del_piso():
    panel = _panel_con_volatilidades_distintas()
    panel.loc[panel["producto_id"] == "tranquilo", "precio_kg"] = 2000.0  # serie plana
    tabla = derivar_umbrales(
        panel,
        {"horizonte_semanas": 2, "derivacion": {"cuantil": 0.75, "redondeo": 0.05, "piso": 0.10}},
    )
    assert (tabla["umbral"] >= 0.10).all()
