"""Variables exógenas: calendario y ENSO.

El riesgo aquí no es que las features estén mal calculadas, sino que metan
información del futuro. El calendario puede mirar hacia adelante —los festivos
de 2027 ya se conocen— pero el ONI no, y esa asimetría es lo que se prueba.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from precios.features.build import columnas_features, construir_features
from precios.features.exogenas import (
    REZAGO_ONI_MESES,
    _parsear_oni,
    features_calendario,
    features_enso,
)

from .test_features import panel_sintetico

ONI_CRUDO = """ SEAS  YR   TOTAL   ANOM
  DJF 2024  26.50  -0.50
  JFM 2024  26.80  -0.20
  FMA 2024  27.10   0.10
  MAM 2024  27.40   0.40
  AMJ 2024  27.90   0.90
  MJJ 2024  28.30   1.30
  JJA 2024  28.60   1.60
  JAS 2024  28.80   1.80
  ASO 2024  28.70   1.70
  SON 2024  28.40   1.40
  OND 2024  28.00   1.00
  NDJ 2024  27.60   0.60
  DJF 2025  27.20   0.20
  JFM 2025  26.90  -0.10
  FMA 2025  26.60  -0.40
  MAM 2025  26.30  -0.70
"""


# --- Calendario -------------------------------------------------------------


def test_detecta_la_semana_santa_de_2026():
    """Jueves y Viernes Santo de 2026 caen el 2 y 3 de abril."""
    cal = features_calendario(pd.Series([pd.Timestamp("2026-03-30")]))
    assert cal.loc[pd.Timestamp("2026-03-30"), "semana_santa"] == 1
    assert cal.loc[pd.Timestamp("2026-03-30"), "festivos"] == 2


def test_una_semana_corriente_no_es_semana_santa():
    cal = features_calendario(pd.Series([pd.Timestamp("2026-04-13")]))
    assert cal.loc[pd.Timestamp("2026-04-13"), "semana_santa"] == 0
    assert cal.loc[pd.Timestamp("2026-04-13"), "festivos"] == 0


def test_aplica_la_ley_emiliani():
    """San José es el 19 de marzo pero se traslada al lunes 23 en 2026."""
    cal = features_calendario(pd.Series([pd.Timestamp("2026-03-16"), pd.Timestamp("2026-03-23")]))
    assert cal.loc[pd.Timestamp("2026-03-16"), "festivos"] == 0
    assert cal.loc[pd.Timestamp("2026-03-23"), "festivo_lunes"] == 1


def test_los_dias_de_cotizacion_descuentan_los_festivos():
    """SIPSA cotiza de lunes a sábado: seis días menos los festivos."""
    cal = features_calendario(pd.Series([pd.Timestamp("2026-03-30")]))
    fila = cal.loc[pd.Timestamp("2026-03-30")]
    assert fila["dias_cotizacion"] == 6 - fila["festivos"]


def test_un_festivo_en_domingo_no_resta_dias_de_cotizacion():
    """El domingo no se cotiza, así que un festivo ahí no cambia nada."""
    cal = features_calendario(pd.Series([pd.Timestamp("2026-11-30")]))
    assert cal.loc[pd.Timestamp("2026-11-30"), "dias_cotizacion"] == 6


def test_detecta_la_semana_de_fin_de_anio():
    cal = features_calendario(pd.Series([pd.Timestamp("2026-12-21")]))
    assert cal.loc[pd.Timestamp("2026-12-21"), "fin_de_anio"] == 1


def test_anade_el_calendario_de_la_semana_objetivo():
    """Los festivos futuros SÍ se conocen: es la única feature que puede mirar
    hacia adelante sin ser leakage."""
    cal = features_calendario(pd.Series([pd.Timestamp("2026-03-23")]), horizontes=(1, 2))
    fila = cal.loc[pd.Timestamp("2026-03-23")]
    # A una semana está la Semana Santa.
    assert fila["semana_santa_h1"] == 1
    assert fila["festivos_h1"] == 2
    # A dos semanas ya no.
    assert fila["semana_santa_h2"] == 0


def test_el_calendario_es_determinista():
    a = features_calendario(pd.Series([pd.Timestamp("2026-06-01")]))
    b = features_calendario(pd.Series([pd.Timestamp("2026-06-01")]))
    pd.testing.assert_frame_equal(a, b)


# --- ENSO -------------------------------------------------------------------


def test_parsea_el_formato_de_la_noaa():
    oni = _parsear_oni(ONI_CRUDO)
    assert len(oni) == 16
    assert list(oni.columns) == ["mes", "oni"]


def test_asigna_cada_trimestre_a_su_mes_central():
    """'DJF' de 2024 es enero de 2024; 'JJA' es julio."""
    oni = _parsear_oni(ONI_CRUDO).set_index("mes")["oni"]
    assert oni[pd.Timestamp("2024-01-01")] == pytest.approx(-0.50)
    assert oni[pd.Timestamp("2024-07-01")] == pytest.approx(1.60)


def test_un_archivo_ilegible_falla_en_vez_de_devolver_vacio():
    with pytest.raises(RuntimeError, match="ninguna fila legible"):
        _parsear_oni("cabecera\nbasura sin formato\n")


def test_el_oni_se_rezaga_para_no_usar_lo_no_publicado():
    """La NOAA publica con retraso: el ONI de una semana debe ser el de dos
    meses antes, no el de su propio mes."""
    oni = _parsear_oni(ONI_CRUDO)
    e = features_enso(pd.Series([pd.Timestamp("2024-09-02")]), oni, rezago_meses=2)
    # Septiembre - 2 meses = julio 2024 -> 1.60
    assert e.loc[pd.Timestamp("2024-09-02"), "oni"] == pytest.approx(1.60)


def test_el_rezago_por_defecto_son_dos_meses():
    assert REZAGO_ONI_MESES == 2


def test_marca_la_fase_nino_con_el_umbral_de_la_noaa():
    oni = _parsear_oni(ONI_CRUDO)
    e = features_enso(pd.Series([pd.Timestamp("2024-09-02")]), oni)
    assert e.loc[pd.Timestamp("2024-09-02"), "enso_nino"] == 1
    assert e.loc[pd.Timestamp("2024-09-02"), "enso_nina"] == 0


def test_marca_la_fase_nina():
    oni = _parsear_oni(ONI_CRUDO)
    # Junio 2025 - 2 meses = abril 2025, que en el fixture es MAM 2025 = -0.70.
    e = features_enso(pd.Series([pd.Timestamp("2025-06-02")]), oni)
    assert e.loc[pd.Timestamp("2025-06-02"), "oni"] == pytest.approx(-0.70)
    assert e.loc[pd.Timestamp("2025-06-02"), "enso_nina"] == 1
    assert e.loc[pd.Timestamp("2025-06-02"), "enso_nino"] == 0


def test_la_tendencia_es_la_diferencia_con_tres_meses_antes():
    oni = _parsear_oni(ONI_CRUDO)
    e = features_enso(pd.Series([pd.Timestamp("2024-09-02")]), oni)
    # julio 2024 (1.60) menos abril 2024 (0.40) = 1.20
    assert e.loc[pd.Timestamp("2024-09-02"), "oni_tendencia_3m"] == pytest.approx(1.20)


def test_sin_dato_para_el_mes_devuelve_nan_en_vez_de_inventar():
    oni = _parsear_oni(ONI_CRUDO)
    e = features_enso(pd.Series([pd.Timestamp("2030-01-07")]), oni)
    assert np.isnan(e.loc[pd.Timestamp("2030-01-07"), "oni"])


def test_el_oni_es_constante_dentro_del_mes():
    """Es una limitación real del índice, no un error: conviene que se vea."""
    oni = _parsear_oni(ONI_CRUDO)
    semanas = pd.Series(pd.date_range("2024-09-02", periods=4, freq="W-MON"))
    e = features_enso(semanas, oni)
    assert e["oni"].nunique() == 1


# --- Integración con el constructor de features -----------------------------


def test_sin_exogenas_el_conjunto_de_features_no_cambia():
    panel = panel_sintetico(n=60, n_series=2)
    base = columnas_features(construir_features(panel))
    assert not [c for c in base if c.startswith(("festivos", "oni", "enso"))]


def test_el_bloque_calendario_anade_sus_columnas():
    panel = panel_sintetico(n=60, n_series=2)
    cols = columnas_features(construir_features(panel, exogenas=("calendario",)))
    assert "festivos" in cols and "semana_santa" in cols and "festivos_h1" in cols


def test_un_bloque_desconocido_falla_pronto():
    panel = panel_sintetico(n=30, n_series=1)
    with pytest.raises(ValueError, match="desconocidos"):
        construir_features(panel, exogenas=("clima",))


def test_las_exogenas_son_iguales_para_todas_las_series_de_una_semana():
    """Son features de calendario: no dependen del producto ni de la plaza."""
    panel = panel_sintetico(n=60, n_series=3)
    F = construir_features(panel, exogenas=("calendario",))
    por_semana = F.groupby("semana")["festivos"].nunique()
    assert (por_semana == 1).all()


def test_el_calendario_no_depende_de_los_precios():
    """Corromper los precios no puede mover una feature de calendario."""
    panel = panel_sintetico(n=60, n_series=1)
    a = construir_features(panel, exogenas=("calendario",))["festivos"]
    panel2 = panel.copy()
    panel2["precio_kg"] *= 1000
    b = construir_features(panel2, exogenas=("calendario",))["festivos"]
    pd.testing.assert_series_equal(a, b)


def test_los_festivos_son_los_de_colombia_no_los_de_otro_pais():
    """El 7 de agosto (Batalla de Boyacá) solo es festivo en Colombia."""
    cal = features_calendario(pd.Series([pd.Timestamp("2026-08-03")]))
    assert cal.loc[pd.Timestamp("2026-08-03"), "festivos"] >= 1
    assert dt.date(2026, 8, 7).weekday() == 4  # viernes
