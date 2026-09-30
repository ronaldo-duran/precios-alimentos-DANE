"""Página de entrada de la app.

Lo que se ve primero no es un pronóstico: es qué son estos datos y qué no son.
"""

from __future__ import annotations

import streamlit as st
from _comun import (
    cargar_metricas_horizonte,
    cargar_panel,
    cargar_reporte_series,
    encabezado,
)

encabezado(
    "Pronóstico de precios mayoristas de alimentos",
    "Colombia · SIPSA–DANE · horizontes de 1 a 4 semanas",
)

panel = cargar_panel()
reporte = cargar_reporte_series()
metricas = cargar_metricas_horizonte()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Series", f"{panel.groupby(['producto_id', 'plaza_id'], observed=True).ngroups}")
col2.metric("Semanas de historia", f"{panel['semana'].nunique()}")
col3.metric("Productos", f"{panel['producto_id'].nunique()}")
col4.metric("Plazas", f"{panel['plaza_id'].nunique()}")

st.markdown(
    """
### Qué hay aquí

Un pipeline reproducible que pronostica **precios mayoristas semanales** de
alimentos en Colombia, con intervalos de incertidumbre y validación
walk-forward. Es un proyecto personal de aprendizaje: su objetivo no es acertar
el precio, sino **medir con honestidad cuándo acierta y cuándo no**.

### Lo primero que deberías saber

Estos son precios **mayoristas**: los que se cotizan en centrales de abastos
como Corabastos o la Central Mayorista de Antioquia. **No son lo que pagas en la
tienda de la esquina.** El margen minorista varía por ciudad, canal y producto,
y este proyecto no lo modela.

### Lo segundo: a una semana, el modelo no le gana al método trivial

Predecir *"el precio de la próxima semana será el de esta semana"* —el baseline
**naive**— es sorprendentemente difícil de batir. Nuestro mejor modelo lo
consigue por un margen del 2% a una semana, y solo gracias a un par de
productos. A 2–4 semanas el margen sube al 3–7%.

Lo dice la página **Desempeño**, con los números.

### Lo tercero: los intervalos fallan justo cuando importan

Las bandas de incertidumbre están calibradas para cubrir el 80% de las veces, y
en semanas normales cumplen. **En semanas de choque —un paro, una helada— la
cobertura cae al 45%.** La página **Choques y límites** explica por qué y
muestra dónde.
"""
)

if metricas is not None:
    mejor = metricas[metricas["h"] == 1].nsmallest(1, "mase")
    naive = metricas[(metricas["h"] == 1) & (metricas["modelo"] == "naive")]
    if not mejor.empty and not naive.empty:
        st.info(
            f"A 1 semana, el mejor modelo (`{mejor['modelo'].iloc[0]}`) tiene "
            f"MASE **{mejor['mase'].iloc[0]:.3f}** frente a **{naive['mase'].iloc[0]:.3f}** "
            f"del naive. Un MASE menor que 1 significaría ganarle al naive dentro "
            f"del propio periodo de entrenamiento; estos valores son mayores porque "
            f"el periodo evaluado es más volátil que el histórico.",
            icon="📊",
        )

if reporte is not None:
    no_aptas = reporte[~reporte["apta"]]
    if not no_aptas.empty:
        detalle = ", ".join(
            f"{r['producto_id']} @ {r['plaza_id']} ({r['motivo_exclusion']})"
            for _, r in no_aptas.iterrows()
        )
        st.caption(f"Series excluidas del modelado por calidad de datos: {detalle}")

st.divider()
st.markdown(
    """
**Fuente**: servicio web SOAP del DANE
(`appweb.dane.gov.co/sipsaWS`), actualizado a diario.
Datos bajo licencia Creative Commons BY-SA 4.0.
Este sitio no está afiliado al DANE.
"""
)
