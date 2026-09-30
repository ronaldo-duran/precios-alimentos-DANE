"""Normalización de nombres y recorte al alcance."""

from __future__ import annotations

import pandas as pd
import pytest

from precios.cleaning.normalize import (
    colapsar_duplicados_diarios,
    filtrar_alcance,
    normalizar_nombres,
)

from .conftest import construir_diario


def test_aplica_alias_de_plaza(normalizacion):
    df = construir_diario(["Papa negra*"], ["Cali, Santa Helena"], n_semanas=2)
    out = normalizar_nombres(df, normalizacion)
    assert set(out["plaza_sipsa"]) == {"Cali, Santa Elena"}


def test_aplica_alias_de_producto(normalizacion):
    df = construir_diario(["Piña *"], ["Bogotá, D.C., Corabastos"], n_semanas=2)
    out = normalizar_nombres(df, normalizacion)
    assert set(out["producto_sipsa"]) == {"Piña*"}


def test_recorta_espacios_sobrantes(normalizacion):
    df = construir_diario(["  Tomate*  "], ["Bogotá, D.C., Corabastos"], n_semanas=2)
    out = normalizar_nombres(df, normalizacion)
    assert set(out["producto_sipsa"]) == {"Tomate*"}


def test_no_toca_nombres_ya_canonicos(normalizacion):
    df = construir_diario(["Tomate*"], ["Bogotá, D.C., Corabastos"], n_semanas=2)
    out = normalizar_nombres(df, normalizacion)
    pd.testing.assert_series_equal(
        out["producto_sipsa"].astype(str), df["producto_sipsa"].astype(str)
    )


def test_colapsa_duplicados_de_un_renombre(normalizacion):
    """Un renombre con solapamiento crea dos filas del mismo mercado el mismo día."""
    vieja = construir_diario(["Tomate*"], ["Cali, Santa Helena"], n_semanas=1, dias_por_semana=1)
    nueva = construir_diario(["Tomate*"], ["Cali, Santa Elena"], n_semanas=1, dias_por_semana=1)
    vieja["precio_kg"] = 1000.0
    nueva["precio_kg"] = 2000.0

    df = normalizar_nombres(pd.concat([vieja, nueva], ignore_index=True), normalizacion)
    assert df.duplicated(subset=["fecha", "producto_sipsa", "plaza_sipsa"]).any()

    out = colapsar_duplicados_diarios(df)
    assert len(out) == 1
    assert out.loc[0, "precio_kg"] == pytest.approx(1500.0)
    assert list(out.columns) == list(df.columns)


def test_colapsar_es_inocuo_sin_duplicados(diario_limpio):
    out = colapsar_duplicados_diarios(diario_limpio)
    assert len(out) == len(diario_limpio)


def test_filtrar_asigna_ids_y_etiquetas(scope):
    df = construir_diario(
        ["Papa negra*", "Tomate*"], ["Bogotá, D.C., Corabastos", "Cali, Santa Elena"], n_semanas=2
    )
    out = filtrar_alcance(df, scope)
    assert set(out["producto_id"]) == {"papa", "tomate"}
    assert set(out["plaza_id"]) == {"bogota", "cali"}
    assert set(out["ciudad"]) == {"Bogotá", "Cali"}


def test_filtrar_descarta_pares_excluidos(scope):
    df = construir_diario(
        ["Papa negra*", "Tomate*"], ["Bogotá, D.C., Corabastos", "Cali, Santa Elena"], n_semanas=2
    )
    out = filtrar_alcance(df, scope)
    pares = set(zip(out["producto_id"], out["plaza_id"], strict=True))
    assert ("papa", "cali") not in pares
    assert len(pares) == scope.n_series_esperadas == 3


def test_filtrar_descarta_lo_que_no_esta_en_el_alcance(scope):
    df = construir_diario(
        ["Papa negra*", "Tomate*", "Yuca*"],
        ["Bogotá, D.C., Corabastos", "Cali, Santa Elena"],
        n_semanas=2,
    )
    out = filtrar_alcance(df, scope)
    assert "Yuca*" not in set(out["producto_sipsa"])


def test_filtrar_falla_si_un_producto_configurado_desaparece(scope):
    """Si SIPSA renombra un producto, el pipeline debe gritar, no seguir en silencio."""
    df = construir_diario(
        ["Papa negra*"], ["Bogotá, D.C., Corabastos", "Cali, Santa Elena"], n_semanas=2
    )
    with pytest.raises(ValueError, match="Tomate"):
        filtrar_alcance(df, scope)


def test_filtrar_falla_si_una_plaza_configurada_desaparece(scope):
    df = construir_diario(["Papa negra*", "Tomate*"], ["Bogotá, D.C., Corabastos"], n_semanas=2)
    with pytest.raises(ValueError, match="Santa Elena"):
        filtrar_alcance(df, scope)
