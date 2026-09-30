"""Utilidades de la app: color y rutas.

Las páginas de Streamlit llaman a `st.*` al importarse, así que no se pueden
importar en un test. Lo que sí se puede —y conviene— es blindar las funciones
puras de color, porque un fallo ahí produce texto ilegible sin romper nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))

from _comun import (  # noqa: E402
    PALETA_CLARA,
    PALETA_OSCURA,
    PROCESADO,
    _mezclar,
    escala_divergente,
    rgba,
)


def _luminancia(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    canales = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    canales = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in canales]
    return 0.2126 * canales[0] + 0.7152 * canales[1] + 0.0722 * canales[2]


def _contraste(a: str, b: str) -> float:
    l1, l2 = sorted([_luminancia(a), _luminancia(b)], reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def test_rgba_convierte_correctamente():
    assert rgba("#2a78d6", 0.15) == "rgba(42,120,214,0.15)"


def test_mezclar_acerca_el_color_a_la_superficie():
    original = _luminancia("#2a78d6")
    mezclado = _luminancia(_mezclar("#2a78d6", 0.62))
    superficie = _luminancia(PALETA_CLARA["superficie"])
    assert original < mezclado < superficie


def test_mezclar_con_alfa_uno_no_cambia_el_color():
    assert _mezclar("#2a78d6", 1.0) == "#2a78d6"


def test_la_escala_divergente_tiene_tres_paradas_y_gris_al_centro():
    escala = escala_divergente()
    assert len(escala) == 3
    assert [pos for pos, _ in escala] == [0.0, 0.5, 1.0]
    # El punto medio debe leerse como "nada": un gris, no un hue.
    r, g, b = (int(escala[1][1].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    assert max(r, g, b) - min(r, g, b) < 12


def test_los_polos_de_la_escala_son_calido_y_frio():
    """Dos hues que se leen como opuestos, no dos fríos."""
    frio = escala_divergente()[2][1]
    calido = escala_divergente()[0][1]
    r_f, _, b_f = (int(frio.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    r_c, _, b_c = (int(calido.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    assert b_f > r_f  # el polo positivo es azul
    assert r_c > b_c  # el polo negativo es rojo


@pytest.mark.parametrize("alfa", [0.5, 0.62, 0.7])
def test_el_texto_de_las_celdas_es_legible_en_toda_la_escala(alfa):
    """Los mapas de calor llevan el valor dentro de la celda: debe leerse."""
    tinta = PALETA_CLARA["tinta"]
    for _, color in escala_divergente(alfa):
        assert _contraste(tinta, color) >= 4.5


def test_las_dos_paletas_definen_las_mismas_claves():
    assert set(PALETA_CLARA) == set(PALETA_OSCURA)


def test_la_ruta_de_datos_se_deriva_del_repo_no_es_absoluta_del_autor():
    """En Streamlit Cloud el repo se clona en otra ruta: nada puede estar fijo."""
    assert PROCESADO == RAIZ / "data" / "processed"
    assert PROCESADO.is_relative_to(RAIZ)
