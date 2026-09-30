"""Intervalos de predicción: cobertura en un caso sintético con respuesta conocida.

El punto de estos tests no es que el modelo acierte, sino que el **mecanismo de
calibración** haga lo que promete. Se construye un proceso con ruido conocido,
de modo que la cobertura correcta sea calculable de antemano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.evaluation.metrics import amplitud_relativa, cobertura
from precios.models.conformal import _cuantil_conformal


def test_el_cuantil_conformal_usa_la_correccion_de_muestra_finita():
    """Con n=9 y nivel 0.8, el índice es ceil(10*0.8)=8: el 8.º residuo ordenado."""
    residuos = np.arange(1.0, 10.0)  # 1..9
    assert _cuantil_conformal(residuos, 0.8) == pytest.approx(8.0)


def test_el_cuantil_conformal_es_conservador_frente_al_empirico():
    residuos = np.arange(1.0, 11.0)  # 1..10
    empirico = np.quantile(residuos, 0.8)
    conformal = _cuantil_conformal(residuos, 0.8)
    assert conformal >= empirico


def test_con_muy_pocos_residuos_se_usa_el_maximo():
    residuos = np.array([2.0, 5.0, 7.0])
    # ceil(4 * 0.95) = 4 > n=3 -> no se puede garantizar, se toma el máximo.
    assert _cuantil_conformal(residuos, 0.95) == 7.0


def test_sin_residuos_devuelve_nan():
    assert np.isnan(_cuantil_conformal(np.array([]), 0.8))


def test_un_intervalo_bien_calibrado_cubre_el_nivel_nominal():
    """Ruido gaussiano conocido: ±1,2816 sigma debe cubrir el 80%."""
    rng = np.random.default_rng(11)
    n = 40_000
    y_pred = np.full(n, 100.0)
    y_true = y_pred + rng.normal(0, 10, n)
    radio = 1.2816 * 10
    cob = cobertura(y_true, y_pred - radio, y_pred + radio)
    assert cob == pytest.approx(80.0, abs=1.0)


def test_la_calibracion_conformal_alcanza_la_cobertura_pedida():
    """Calibrar con una mitad y medir en la otra debe dar ~80% de cobertura."""
    rng = np.random.default_rng(23)
    n = 20_000
    # Residuos con una distribución fea (mezcla), donde un sigma gaussiano fallaría.
    residuos = np.where(
        rng.random(n) < 0.9, rng.normal(0, 1, n), rng.normal(0, 6, n)
    )
    calibracion, prueba = residuos[: n // 2], residuos[n // 2 :]

    radio = _cuantil_conformal(np.abs(calibracion), 0.8)
    cob = 100 * np.mean(np.abs(prueba) <= radio)
    assert cob == pytest.approx(80.0, abs=1.5)


def test_un_intervalo_demasiado_estrecho_se_detecta():
    """La regresión cuantílica mal calibrada debe delatarse en la cobertura."""
    rng = np.random.default_rng(31)
    n = 20_000
    y_pred = np.full(n, 100.0)
    y_true = y_pred + rng.normal(0, 10, n)
    radio_corto = 0.5 * 10  # solo ~38% de cobertura
    cob = cobertura(y_true, y_pred - radio_corto, y_pred + radio_corto)
    assert cob < 50.0


def test_la_amplitud_acompana_a_la_cobertura():
    """Un intervalo puede cubrir el 100% siendo inútil de ancho."""
    y_true = np.array([100.0, 100.0, 100.0])
    ancho = np.array([1000.0, 1000.0, 1000.0])
    lo, hi = y_true - ancho, y_true + ancho
    assert cobertura(y_true, lo, hi) == 100.0
    assert amplitud_relativa(y_true, lo, hi) == pytest.approx(2000.0)


def test_cobertura_ignora_filas_sin_intervalo():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, np.nan, 0.0])
    hi = np.array([5.0, np.nan, 5.0])
    assert cobertura(y, lo, hi) == 100.0


def test_resumen_de_cobertura_por_horizonte():
    from precios.evaluation.metrics import resumir

    df = pd.DataFrame(
        {
            "modelo": ["m"] * 4,
            "h": [1, 1, 2, 2],
            "y_true": [100.0, 100.0, 100.0, 100.0],
            "y_pred": [100.0] * 4,
            "lo": [95.0, 95.0, 80.0, 80.0],
            "hi": [105.0, 105.0, 120.0, 120.0],
        }
    )
    out = resumir(df, por=["modelo", "h"])
    # El intervalo de h=2 es cuatro veces más ancho.
    anchos = out.sort_values("h")["amplitud_pct"].tolist()
    assert anchos[1] == pytest.approx(4 * anchos[0])
