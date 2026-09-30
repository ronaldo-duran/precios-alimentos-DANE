"""Punto de entrada de la app.

Se usa `st.navigation` en vez del descubrimiento automático de `pages/` para
poder escribir los títulos del menú con acentos y en un orden elegido: con el
descubrimiento automático, Streamlit los deriva del nombre del archivo y
saldrían "Pronostico" y "Choques y limites".

Ejecutar con:  streamlit run app/Inicio.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Precios mayoristas SIPSA–DANE",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGINAS = [
    st.Page("paginas/inicio.py", title="Inicio", icon=":material/home:", default=True),
    st.Page("paginas/pronostico.py", title="Pronóstico", icon=":material/trending_up:"),
    st.Page("paginas/desempeno.py", title="Desempeño", icon=":material/insights:"),
    st.Page("paginas/choques.py", title="Choques y límites", icon=":material/bolt:"),
    st.Page("paginas/alertas.py", title="Alertas", icon=":material/notifications:"),
]

st.navigation(PAGINAS).run()
