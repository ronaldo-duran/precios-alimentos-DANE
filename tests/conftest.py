"""Fixtures compartidas: datos sintéticos con problemas conocidos.

Los tests no tocan la red ni `data/`. Cada fixture construye un caso pequeño
donde la respuesta correcta se conoce de antemano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.config import Exclusion, Normalizacion, Plaza, Producto, ReglasLimpieza, Scope
from precios.data.sources import COLUMNAS_CANONICAS


@pytest.fixture
def scope() -> Scope:
    return Scope(
        productos=(
            Producto(id="papa", sipsa="Papa negra*", etiqueta="Papa negra"),
            Producto(id="tomate", sipsa="Tomate*", etiqueta="Tomate"),
        ),
        plazas=(
            Plaza(
                id="bogota",
                sipsa="Bogotá, D.C., Corabastos",
                ciudad="Bogotá",
                etiqueta="Corabastos",
            ),
            Plaza(id="cali", sipsa="Cali, Santa Elena", ciudad="Cali", etiqueta="Santa Elena"),
        ),
        exclusiones=(Exclusion(producto="papa", plaza="cali", motivo="serie cerrada"),),
    )


@pytest.fixture
def normalizacion() -> Normalizacion:
    return Normalizacion(
        plazas={"Cali, Santa Helena": "Cali, Santa Elena"},
        productos={"Piña *": "Piña*"},
    )


@pytest.fixture
def reglas() -> ReglasLimpieza:
    return ReglasLimpieza(
        agregacion={
            "estadistico": "mediana",
            "inicio_semana": "lunes",
            "min_dias_por_semana": 1,
            "descartar_semana_parcial": True,
        },
        limites={"precio_kg_min": 100.0, "precio_kg_max": 500000.0},
        outliers={
            "metodo": "mad_log_retorno",
            "umbral_z": 5.0,
            "salto_relativo_min": 0.15,
            "salto_relativo_max": 1.0,
        },
        series={"max_pct_faltantes": 10.0, "min_semanas": 104},
    )


def construir_diario(
    productos: list[str],
    plazas: list[str],
    n_semanas: int = 12,
    dias_por_semana: int = 5,
    precio_base: float = 2000.0,
    inicio: str = "2024-01-01",
) -> pd.DataFrame:
    """Genera un panel diario sintético y limpio (lunes = inicio de semana)."""
    rng = np.random.default_rng(42)
    filas = []
    lunes0 = pd.Timestamp(inicio)
    for s in range(n_semanas):
        for d in range(dias_por_semana):
            fecha = lunes0 + pd.Timedelta(7 * s + d, unit="D")
            for prod in productos:
                for plaza in plazas:
                    precio = precio_base * (1 + 0.02 * rng.standard_normal())
                    filas.append(
                        {
                            "fecha": fecha,
                            "producto_sipsa": prod,
                            "plaza_sipsa": plaza,
                            "precio_kg": round(precio, 1),
                            "precio_kg_min": round(precio * 0.95, 1),
                            "precio_kg_max": round(precio * 1.05, 1),
                            "grupo": "VERDURAS",
                            "departamento": "X",
                            "municipio": "Y",
                            "fuente": "test",
                        }
                    )
    return pd.DataFrame(filas)[list(COLUMNAS_CANONICAS)]


@pytest.fixture
def diario_limpio() -> pd.DataFrame:
    return construir_diario(
        ["Papa negra*", "Tomate*"],
        ["Bogotá, D.C., Corabastos", "Cali, Santa Elena"],
    )
