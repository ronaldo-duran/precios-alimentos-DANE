"""La validación debe DETENER el pipeline ante datos que no son de fiar."""

from __future__ import annotations

import pandas as pd
import pytest

from precios.data.validation import ValidationError, reporte_series, validar_diario, validar_semanal


def test_acepta_datos_limpios(diario_limpio, reglas):
    validar_diario(diario_limpio, reglas)  # no debe lanzar


def test_rechaza_esquema_inesperado(diario_limpio, reglas):
    df = diario_limpio.drop(columns=["precio_kg_min"])
    with pytest.raises(ValidationError, match="Esquema inesperado"):
        validar_diario(df, reglas)


def test_rechaza_dataframe_vacio(diario_limpio, reglas):
    with pytest.raises(ValidationError, match="vacío"):
        validar_diario(diario_limpio.iloc[0:0], reglas)


def test_rechaza_precio_no_positivo(diario_limpio, reglas):
    df = diario_limpio.copy()
    df.loc[0, "precio_kg"] = 0.0
    with pytest.raises(ValidationError) as exc:
        validar_diario(df, reglas)
    assert any("<= 0" in f for f in exc.value.fallos)


def test_rechaza_precio_absurdo(diario_limpio, reglas):
    df = diario_limpio.copy()
    df.loc[0, "precio_kg"] = 9_000_000.0
    with pytest.raises(ValidationError) as exc:
        validar_diario(df, reglas)
    assert any("fuera del rango" in f for f in exc.value.fallos)


def test_rechaza_duplicados_por_fecha_producto_plaza(diario_limpio, reglas):
    df = pd.concat([diario_limpio, diario_limpio.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError) as exc:
        validar_diario(df, reglas)
    assert any("duplicadas" in f for f in exc.value.fallos)


def test_rechaza_nulos_en_columna_clave(diario_limpio, reglas):
    df = diario_limpio.copy()
    df.loc[0, "producto_sipsa"] = None
    with pytest.raises(ValidationError) as exc:
        validar_diario(df, reglas)
    assert any("nulos" in f for f in exc.value.fallos)


def test_rechaza_fechas_futuras(diario_limpio, reglas):
    df = diario_limpio.copy()
    df.loc[0, "fecha"] = pd.Timestamp.today().normalize() + pd.Timedelta(30, unit="D")
    with pytest.raises(ValidationError) as exc:
        validar_diario(df, reglas)
    assert any("futura" in f for f in exc.value.fallos)


def _panel_semanal_minimo() -> pd.DataFrame:
    semanas = pd.date_range("2024-01-01", periods=6, freq="W-MON")
    return pd.DataFrame(
        {
            "semana": semanas,
            "producto_id": "papa",
            "plaza_id": "bogota",
            "precio_kg": [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0],
            "flag_outlier": False,
        }
    )


def test_semanal_rechaza_semana_que_no_es_lunes(reglas):
    df = _panel_semanal_minimo()
    df.loc[0, "semana"] = df.loc[0, "semana"] + pd.Timedelta(2, unit="D")
    with pytest.raises(ValidationError) as exc:
        validar_semanal(df, reglas)
    assert any("lunes" in f for f in exc.value.fallos)


def test_semanal_rechaza_fechas_fuera_de_orden(reglas):
    df = _panel_semanal_minimo()
    df = pd.concat([df.iloc[[3]], df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError) as exc:
        validar_semanal(df, reglas)
    assert any("fuera de orden" in f for f in exc.value.fallos)


def test_reporte_marca_serie_con_demasiados_faltantes(reglas):
    df = _panel_semanal_minimo()
    # 6 semanas < min_semanas (104) -> no apta.
    rep = reporte_series(df, reglas)
    assert len(rep) == 1
    assert not rep.loc[0, "apta"]
    assert "historia" in rep.loc[0, "motivo_exclusion"]


def test_reporte_detecta_huecos(reglas):
    df = _panel_semanal_minimo()
    df = df.drop(index=[2, 3]).reset_index(drop=True)  # 4 de 6 semanas
    rep = reporte_series(df, reglas)
    assert rep.loc[0, "n_semanas"] == 4
    assert rep.loc[0, "span_semanas"] == 6
    assert rep.loc[0, "pct_faltantes"] == pytest.approx(33.33, abs=0.01)
