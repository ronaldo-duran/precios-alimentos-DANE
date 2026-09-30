"""Agregación diaria -> semanal, flags de calidad y reindexado."""

from __future__ import annotations

import pandas as pd
import pytest

from precios.cleaning.aggregate import (
    _lunes,
    agregar_semanal,
    descartar_semana_parcial,
    marcar_outliers,
    reindexar_semanas,
)
from precios.cleaning.normalize import filtrar_alcance

from .conftest import construir_diario


def _preparar(scope, **kwargs) -> pd.DataFrame:
    df = construir_diario(
        ["Papa negra*", "Tomate*"], ["Bogotá, D.C., Corabastos"], **kwargs
    )
    scope_bogota = type(scope)(
        productos=scope.productos, plazas=(scope.plazas[0],), exclusiones=()
    )
    return filtrar_alcance(df, scope_bogota)


def test_la_semana_se_etiqueta_con_su_lunes():
    fechas = pd.Series(pd.to_datetime(["2024-03-06", "2024-03-09", "2024-03-04"]))
    assert set(_lunes(fechas)) == {pd.Timestamp("2024-03-04")}


def test_agrega_con_mediana_y_no_con_promedio(scope, reglas):
    """Un error de digitación de un día no debe arrastrar la semana."""
    df = _preparar(scope, n_semanas=4, dias_por_semana=5)
    semanas = df["semana"] if "semana" in df else _lunes(df["fecha"])
    mascara = semanas == pd.Timestamp("2024-01-01")
    idx = df[mascara & (df["producto_id"] == "papa")].index[0]
    df.loc[idx, "precio_kg"] = 250.0  # cotización imposible, como la de Popayán

    out = agregar_semanal(df, reglas)
    fila = out[(out["producto_id"] == "papa") & (out["semana"] == pd.Timestamp("2024-01-01"))]
    # La mediana se queda cerca del nivel real (~2000), el promedio caería mucho.
    assert fila["precio_kg"].iloc[0] > 1800


def test_cuenta_los_dias_de_cada_semana(scope, reglas):
    df = _preparar(scope, n_semanas=4, dias_por_semana=5)
    out = agregar_semanal(df, reglas)
    assert set(out["n_dias"]) == {5}


def test_descarta_la_semana_en_curso(scope, reglas):
    """La última semana con menos días de lo habitual está incompleta."""
    completo = _preparar(scope, n_semanas=10, dias_por_semana=6)
    completo["semana"] = _lunes(completo["fecha"])
    ultima = completo["semana"].max()
    # Deja solo 1 día en la última semana.
    dia1 = completo[completo["semana"] == ultima]["fecha"].min()
    parcial = completo[(completo["semana"] != ultima) | (completo["fecha"] == dia1)]

    out = descartar_semana_parcial(parcial)
    assert ultima not in set(out["semana"])
    assert len(set(out["semana"])) == 9


def test_no_descarta_una_semana_completa(scope, reglas):
    df = _preparar(scope, n_semanas=10, dias_por_semana=6)
    df["semana"] = _lunes(df["fecha"])
    out = descartar_semana_parcial(df)
    assert len(out) == len(df)


def test_marca_outliers_sin_borrar_filas(reglas):
    semanas = pd.date_range("2024-01-01", periods=30, freq="W-MON")
    precios = [2000.0] * 30
    precios[15] = 12000.0  # salto de +500%
    df = pd.DataFrame(
        {
            "semana": semanas,
            "producto_id": "papa",
            "plaza_id": "bogota",
            "precio_kg": precios,
        }
    )
    out = marcar_outliers(df, reglas)
    assert len(out) == len(df)  # nada se borra
    assert bool(out.loc[15, "flag_outlier"])
    assert out["flag_outlier"].sum() >= 1


def test_no_marca_outliers_en_una_serie_con_ruido_normal(reglas):
    import numpy as np

    rng = np.random.default_rng(7)
    semanas = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    precios = 2000 * np.exp(np.cumsum(0.03 * rng.standard_normal(60)))
    df = pd.DataFrame(
        {
            "semana": semanas,
            "producto_id": "papa",
            "plaza_id": "bogota",
            "precio_kg": precios,
        }
    )
    out = marcar_outliers(df, reglas)
    assert not out["flag_outlier"].any()


def test_una_serie_casi_plana_no_dispara_outliers_triviales(reglas):
    """Con MAD ~ 0, el z-score se dispara solo; el piso de relevancia lo frena.

    Serie en diente de sierra 2000/2010/2020: la variación máxima es del 1%,
    pero la MAD de los log-retornos es casi cero, así que el z-score llega a
    cientos. Sin `salto_relativo_min` esto se marcaría como outlier.
    """
    semanas = pd.date_range("2024-01-01", periods=30, freq="W-MON")
    df = pd.DataFrame(
        {
            "semana": semanas,
            "producto_id": "papa",
            "plaza_id": "bogota",
            "precio_kg": [2000.0 + 10 * (i % 3) for i in range(30)],
        }
    )
    out = marcar_outliers(df, reglas)
    assert out["z_robusto"].abs().max() > reglas.outliers["umbral_z"]
    assert not out["flag_outlier"].any()


def test_reindexa_e_identifica_semanas_faltantes():
    semanas = list(pd.date_range("2024-01-01", periods=6, freq="W-MON"))
    del semanas[2]  # hueco deliberado
    df = pd.DataFrame(
        {
            "semana": semanas,
            "producto_id": "papa",
            "plaza_id": "bogota",
            "producto": "Papa negra",
            "plaza": "Corabastos",
            "ciudad": "Bogotá",
            "grupo": "TUBERCULOS",
            "precio_kg": [1000.0, 1010.0, 1030.0, 1040.0, 1050.0],
        }
    )
    out = reindexar_semanas(df)
    assert len(out) == 6
    assert int(out["flag_faltante"].sum()) == 1
    hueco = out[out["flag_faltante"]]
    assert hueco["semana"].iloc[0] == pd.Timestamp("2024-01-15")
    assert pd.isna(hueco["precio_kg"].iloc[0])  # se marca, no se imputa


def test_el_reindexado_conserva_los_metadatos_de_la_serie():
    semanas = list(pd.date_range("2024-01-01", periods=4, freq="W-MON"))
    del semanas[1]
    df = pd.DataFrame(
        {
            "semana": semanas,
            "producto_id": "papa",
            "plaza_id": "bogota",
            "producto": "Papa negra",
            "plaza": "Corabastos",
            "ciudad": "Bogotá",
            "grupo": "TUBERCULOS",
            "precio_kg": [1000.0, 1030.0, 1040.0],
        }
    )
    out = reindexar_semanas(df)
    assert set(out["producto"]) == {"Papa negra"}
    assert set(out["ciudad"]) == {"Bogotá"}


def test_rechaza_estadistico_desconocido(scope, reglas):
    df = _preparar(scope, n_semanas=3)
    reglas.agregacion["estadistico"] = "moda"
    with pytest.raises(ValueError, match="estadistico"):
        agregar_semanal(df, reglas)
