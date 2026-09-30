"""Detección de cambios de unidad de medida.

Un cambio de unidad es el fallo más peligroso porque no se parece a un error:
el pipeline lo leería como una caída real de precios. El detector se prueba en
las dos direcciones — que dispare cuando debe y que calle cuando no — porque
las dos primeras versiones fallaron precisamente en una de ellas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.cleaning.unidades import (
    TOLERANCIA,
    _factor_sospechoso,
    detectar_cambio_unidad,
    marcar_cambio_unidad,
)


def panel_volatil(n: int = 120, n_series: int = 12, semilla: int = 4) -> pd.DataFrame:
    """Panel con la volatilidad real de estos productos (CV alto)."""
    rng = np.random.default_rng(semilla)
    semanas = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    filas = []
    for i in range(n_series):
        precios = 2000 * np.exp(np.cumsum(0.06 * rng.standard_normal(n)))
        filas.append(
            pd.DataFrame(
                {
                    "semana": semanas,
                    "producto_id": f"p{i % 4}",
                    "plaza_id": f"z{i // 4}",
                    "precio_kg": precios,
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


def _aplicar(panel: pd.DataFrame, factor: float, desde: str, solo=None) -> pd.DataFrame:
    out = panel.copy()
    m = out["semana"] >= pd.Timestamp(desde)
    if solo is not None:
        m &= out["producto_id"].isin(solo)
    out.loc[m, "precio_kg"] *= factor
    return out


# --- El factor --------------------------------------------------------------


def test_reconoce_los_factores_redondos():
    assert _factor_sospechoso(2.0) == 2.0
    assert _factor_sospechoso(0.5) == 2.0 or _factor_sospechoso(0.5) == pytest.approx(0.5)
    assert _factor_sospechoso(1000.0) == 1000.0


def test_un_ratio_de_mercado_no_es_un_factor():
    assert _factor_sospechoso(1.0) is None
    assert _factor_sospechoso(1.3) is None
    assert _factor_sospechoso(0.75) is None


def test_ignora_ratios_imposibles():
    assert _factor_sospechoso(0.0) is None
    assert _factor_sospechoso(-2.0) is None
    assert _factor_sospechoso(float("nan")) is None


def test_la_tolerancia_no_alcanza_la_volatilidad_normal():
    """La banda de x0,5 llega a 0,575; el mínimo real observado es 0,707."""
    assert 0.5 * (1 + TOLERANCIA) < 0.707


# --- No dispara cuando no debe ---------------------------------------------


def test_no_dispara_con_volatilidad_normal():
    """La primera versión daba más de cien falsos positivos aquí."""
    assert detectar_cambio_unidad(panel_volatil()).empty


def test_no_dispara_si_solo_cambia_un_producto():
    """Que un producto se desplome a la mitad es mercado, no unidad."""
    panel = _aplicar(panel_volatil(), 0.5, "2024-09-01", solo=["p0"])
    assert detectar_cambio_unidad(panel).empty


def test_no_dispara_con_una_tendencia_suave():
    panel = panel_volatil()
    factor = np.linspace(1.0, 1.4, len(panel))
    panel["precio_kg"] = panel["precio_kg"] * factor
    assert detectar_cambio_unidad(panel).empty


def test_con_pocas_series_no_concluye():
    """La mediana transversal no significa nada con dos series."""
    panel = _aplicar(panel_volatil(n_series=2), 0.5, "2024-09-01")
    assert detectar_cambio_unidad(panel, min_series=5).empty


def test_un_panel_vacio_no_revienta():
    vacio = panel_volatil().iloc[0:0]
    assert detectar_cambio_unidad(vacio).empty


# --- Sí dispara cuando debe -------------------------------------------------


def test_detecta_el_panel_partido_a_la_mitad():
    """kg -> 500 g: el caso que motivó todo el módulo."""
    panel = _aplicar(panel_volatil(), 0.5, "2024-09-02")
    sospechas = detectar_cambio_unidad(panel)
    assert not sospechas.empty
    assert (sospechas["ratio_transversal"] < 0.6).all()


def test_detecta_un_cambio_de_escala_de_moneda():
    panel = _aplicar(panel_volatil(), 1000.0, "2024-09-02")
    sospechas = detectar_cambio_unidad(panel)
    assert not sospechas.empty
    assert sospechas["factor"].max() >= 100


def test_detecta_el_paso_de_kilos_a_libras():
    panel = _aplicar(panel_volatil(), 1 / 2.2046, "2024-09-02")
    assert not detectar_cambio_unidad(panel).empty


def test_la_sospecha_apunta_cerca_de_la_semana_real():
    corte = pd.Timestamp("2024-09-02")
    panel = _aplicar(panel_volatil(), 0.5, str(corte.date()))
    sospechas = detectar_cambio_unidad(panel, ventana=8)
    # La ventana de 8 semanas hace que varias semanas contiguas al corte
    # disparen; basta con que el corte real quede dentro del rango señalado.
    assert sospechas["semana"].min() <= corte <= sospechas["semana"].max()


def test_reporta_cuantas_series_acompanan():
    panel = _aplicar(panel_volatil(), 0.5, "2024-09-02")
    sospechas = detectar_cambio_unidad(panel)
    assert (sospechas["pct_en_banda"] > 50).any()


# --- El marcado -------------------------------------------------------------


def test_marca_desde_la_primera_semana_sospechosa():
    panel = _aplicar(panel_volatil(), 0.5, "2024-09-02")
    sospechas = detectar_cambio_unidad(panel)
    marcado = marcar_cambio_unidad(panel, sospechas)
    desde = sospechas["semana"].min()
    assert marcado.loc[marcado["semana"] >= desde, "flag_cambio_unidad"].all()
    assert not marcado.loc[marcado["semana"] < desde, "flag_cambio_unidad"].any()


def test_sin_sospechas_la_columna_existe_y_es_falsa():
    """El esquema del panel no debe depender de si hubo sospechas."""
    panel = panel_volatil()
    marcado = marcar_cambio_unidad(panel, detectar_cambio_unidad(panel))
    assert "flag_cambio_unidad" in marcado.columns
    assert not marcado["flag_cambio_unidad"].any()


def test_marcar_no_altera_los_precios():
    panel = _aplicar(panel_volatil(), 0.5, "2024-09-02")
    marcado = marcar_cambio_unidad(panel, detectar_cambio_unidad(panel))
    pd.testing.assert_series_equal(marcado["precio_kg"], panel["precio_kg"])
