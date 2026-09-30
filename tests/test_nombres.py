"""Canonización de nombres propios colombianos.

El caso de prueba es la lista real de formas en que aparece "Bogotá" en fuentes
públicas colombianas. Si esta suite pasa, un cambio de grafía en el origen no
hace desaparecer una serie en silencio.
"""

from __future__ import annotations

import pandas as pd
import pytest

from precios.cleaning.nombres import Canonizador, clave, normalizar_serie
from precios.cleaning.normalize import canonizar_contra_alcance

# Las 35 formas que el usuario reportó, más las que añade el propio DANE.
VARIANTES_BOGOTA = [
    "Bogotá", "bogotá", "BOGOTÁ", "Bogota", "bogota", "BOGOTA",
    "Bogotá D.C.", "Bogotá D.C", "Bogotá dc", "Bogotá DC", "bogotá dc",
    "bogotá DC", "Bogota DC", "Bogota dc", "bogota dc", "BOGOTA DC",
    "BOGOTÁ DC", "Bogotá, D.C.", "Bogotá, D.C", "Bogota, D.C.", "Bogota, D.C",
    "Bogotá, DC", "Bogota, DC", "Bogotá D. C.", "Bogotá D. C", "Bogota D. C.",
    "Bogota D. C", "Bogotá Distrito Capital", "Bogota Distrito Capital",
    "Bogotá, Distrito Capital", "Bogota, Distrito Capital",
    "Distrito Capital de Bogotá", "Distrito Capital de Bogota",
    "Bogotá Capital", "Bogota Capital", "  BOGOTÁ,  D. C.  ",
]

VARIANTES_CUCUTA = [
    "Cúcuta", "cucuta", "CÚCUTA", "CUCUTA", "Cucuta",
    "San José de Cúcuta", "SAN JOSE DE CUCUTA", "san josé de cucuta",
    "San Jose De Cúcuta", "SAN JOSÉ DE CÚCUTA",
]


def _canonizador() -> Canonizador:
    return Canonizador(
        canonicos={
            "Bogotá, D.C.": [
                "Bogotá", "Bogotá D.C.", "Bogotá Distrito Capital",
                "Distrito Capital de Bogotá", "Bogotá Capital", "Distrito Capital",
            ],
            "Cúcuta": ["San José de Cúcuta"],
            "Medellín": [],
        },
        etiqueta="ciudad",
    )


# --- La clave --------------------------------------------------------------


@pytest.mark.parametrize("texto", VARIANTES_BOGOTA)
def test_toda_variante_de_bogota_da_una_de_cinco_claves(texto):
    assert clave(texto) in {
        "bogota", "bogotadc", "bogotadistritocapital",
        "distritocapitaldebogota", "bogotacapital",
    }


def test_la_clave_quita_tildes():
    assert clave("Medellín") == clave("MEDELLIN") == "medellin"


def test_la_clave_quita_puntuacion_y_espacios():
    assert clave("Bogotá, D. C.") == clave("BogotaDC") == "bogotadc"


def test_la_clave_ignora_espacios_sobrantes():
    assert clave("  Cúcuta  ") == "cucuta"


def test_la_clave_de_vacio_o_none_es_vacia():
    assert clave("") == ""
    assert clave(None) == ""


def test_la_clave_distingue_nombres_realmente_distintos():
    """Quitar ruido no puede fusionar dos lugares diferentes."""
    assert clave("San José de Cúcuta") != clave("San José del Guaviare")
    assert clave("Santa Elena") != clave("Santa Helena")


def test_la_clave_conserva_los_numeros():
    assert clave("Pereira, La 41") == "pereirala41"


# --- El canonizador --------------------------------------------------------


@pytest.mark.parametrize("texto", VARIANTES_BOGOTA)
def test_todas_las_variantes_de_bogota_resuelven_a_la_canonica(texto):
    assert _canonizador().resolver(texto) == "Bogotá, D.C."


@pytest.mark.parametrize("texto", VARIANTES_CUCUTA)
def test_cucuta_y_san_jose_de_cucuta_son_el_mismo_lugar(texto):
    assert _canonizador().resolver(texto) == "Cúcuta"


def test_un_nombre_desconocido_devuelve_none():
    assert _canonizador().resolver("Leticia") is None


def test_el_modo_estricto_falla_con_instrucciones():
    with pytest.raises(ValueError, match="config/nombres.yaml"):
        _canonizador().resolver_estricto("Leticia")


def test_el_error_incluye_la_clave_para_poder_diagnosticar():
    with pytest.raises(ValueError, match="leticia"):
        _canonizador().resolver_estricto("Leticia")


def test_una_canonica_sin_variantes_se_resuelve_a_si_misma():
    c = _canonizador()
    assert c.resolver("medellin") == "Medellín"
    assert c.resolver("MEDELLÍN") == "Medellín"


def test_rechaza_una_variante_asignada_a_dos_canonicas():
    """Un diccionario ambiguo debe fallar al construirse, no al usarse."""
    with pytest.raises(ValueError, match="dos formas canónicas"):
        Canonizador(canonicos={"Cali": ["Santiago"], "Santiago de Chile": ["Santiago"]})


def test_desconocidos_reporta_el_nombre_y_su_clave():
    fuera = _canonizador().desconocidos(["Bogotá", "Leticia", "Leticia"])
    assert fuera == {"Leticia": "leticia"}


def test_normalizar_serie_conserva_el_original_si_no_resuelve():
    """Convertir un nombre nuevo en None lo volvería un hueco invisible."""
    salida = normalizar_serie(["BOGOTA DC", "Leticia"], _canonizador())
    assert salida == ["Bogotá, D.C.", "Leticia"]


def test_normalizar_serie_en_estricto_falla():
    with pytest.raises(ValueError):
        normalizar_serie(["Leticia"], _canonizador(), estricto=True)


# --- Integración con el alcance del pipeline -------------------------------


def test_canonizar_contra_alcance_recupera_una_grafia_cambiada(scope):
    """El caso IDEAM: la fuente pasa de MAYÚSCULAS a Capitalizado sin avisar."""
    df = pd.DataFrame(
        {
            "producto_sipsa": ["PAPA NEGRA*", "tomate*"],
            "plaza_sipsa": ["BOGOTÁ, D.C., CORABASTOS", "bogota, d.c., corabastos"],
        }
    )
    out = canonizar_contra_alcance(df, scope)
    assert set(out["producto_sipsa"]) == {"Papa negra*", "Tomate*"}
    assert set(out["plaza_sipsa"]) == {"Bogotá, D.C., Corabastos"}


def test_canonizar_contra_alcance_no_toca_lo_que_ya_coincide(scope):
    df = pd.DataFrame(
        {
            "producto_sipsa": ["Papa negra*"],
            "plaza_sipsa": ["Bogotá, D.C., Corabastos"],
        }
    )
    out = canonizar_contra_alcance(df, scope)
    pd.testing.assert_frame_equal(out, df)


def test_canonizar_contra_alcance_deja_pasar_lo_que_no_es_del_alcance(scope):
    """Lo de fuera del alcance se descarta después; aquí no se inventa nada."""
    df = pd.DataFrame(
        {"producto_sipsa": ["Uchuva"], "plaza_sipsa": ["Leticia, La Plaza"]}
    )
    out = canonizar_contra_alcance(df, scope)
    assert out["producto_sipsa"].iloc[0] == "Uchuva"
