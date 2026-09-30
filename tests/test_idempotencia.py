"""El pipeline debe ser idempotente: correrlo dos veces da lo mismo.

Es la propiedad que hace seguro el cron semanal: si el job se reintenta o corre
dos veces el mismo día, no debe producir un estado distinto.
"""

from __future__ import annotations

import pandas as pd

from precios.cleaning.aggregate import agregar_semanal, reindexar_semanas
from precios.cleaning.normalize import (
    colapsar_duplicados_diarios,
    filtrar_alcance,
    normalizar_nombres,
)

from .conftest import construir_diario


def _panel(df, scope, normalizacion, reglas) -> pd.DataFrame:
    salida = normalizar_nombres(df, normalizacion)
    salida = colapsar_duplicados_diarios(salida)
    salida = filtrar_alcance(salida, scope)
    return reindexar_semanas(agregar_semanal(salida, reglas))


def test_limpiar_dos_veces_da_el_mismo_panel(scope, normalizacion, reglas):
    df = construir_diario(
        ["Papa negra*", "Tomate*"],
        ["Bogotá, D.C., Corabastos", "Cali, Santa Elena"],
        n_semanas=20,
    )
    primera = _panel(df, scope, normalizacion, reglas)
    segunda = _panel(df, scope, normalizacion, reglas)
    pd.testing.assert_frame_equal(primera, segunda)


def test_normalizar_es_idempotente(normalizacion):
    df = construir_diario(["Piña *"], ["Cali, Santa Helena"], n_semanas=2)
    una = normalizar_nombres(df, normalizacion)
    dos = normalizar_nombres(una, normalizacion)
    pd.testing.assert_frame_equal(una, dos)


def test_reingerir_los_mismos_dias_no_duplica_filas(scope, normalizacion, reglas):
    """Reprocesar un solapamiento de días no debe inflar el panel."""
    df = construir_diario(
        ["Papa negra*", "Tomate*"], ["Bogotá, D.C., Corabastos"], n_semanas=10
    )
    scope_bogota = type(scope)(productos=scope.productos, plazas=(scope.plazas[0],), exclusiones=())

    solo_una_vez = _panel(df, scope_bogota, normalizacion, reglas)

    # Simula una reingesta que vuelve a traer las últimas 3 semanas.
    corte = df["fecha"].max() - pd.Timedelta(21, unit="D")
    repetido = pd.concat([df, df[df["fecha"] > corte]], ignore_index=True)
    repetido = colapsar_duplicados_diarios(normalizar_nombres(repetido, normalizacion))
    con_repeticion = reindexar_semanas(
        agregar_semanal(filtrar_alcance(repetido, scope_bogota), reglas)
    )

    assert len(con_repeticion) == len(solo_una_vez)
    pd.testing.assert_series_equal(
        con_repeticion["precio_kg"], solo_una_vez["precio_kg"], check_exact=False
    )
