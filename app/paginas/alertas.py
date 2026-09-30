"""Página 4: probabilidad de alza en las próximas 2 semanas, por serie."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _comun import (
    PROCESADO,
    cargar_alertas,
    encabezado,
    estilo_figura,
    etiquetas,
    falta_artefacto,
    paleta,
)

encabezado("Alertas de alza", "Probabilidad de que el precio suba más del umbral en 2 semanas")

p = paleta()
prod_map, plaza_map = etiquetas()
alertas = cargar_alertas()

if alertas is None or alertas.empty:
    falta_artefacto("alertas_log.csv", "make forecast")
    st.stop()

ultimo = alertas["semana_origen"].max()
vigentes = alertas[alertas["semana_origen"] == ultimo].copy()
vigentes["producto"] = vigentes["producto_id"].map(lambda k: prod_map.get(k, k))
vigentes["plaza"] = vigentes["plaza_id"].map(lambda k: plaza_map.get(k, k))
vigentes = vigentes.sort_values("probabilidad", ascending=False)

activas = vigentes[vigentes["alerta_activa"]]
c1, c2, c3 = st.columns(3)
c1.metric("Alertas activas", f"{len(activas)} de {len(vigentes)}")
c2.metric("Probabilidad mediana", f"{100 * vigentes['probabilidad'].median():.0f}%")
c3.metric("Ventana", f"{ultimo.date():%d/%m} → {vigentes['semana_limite'].iloc[0].date():%d/%m}")

st.markdown(
    """
### El umbral es distinto para cada producto, y eso es lo importante

Un umbral plano del 10% no significa lo mismo en yuca que en tomate: con 10%
fijo la alerta se dispararía el **13,7% de las semanas en yuca y el 36,6% en
tomate**. Una alerta que suena un tercio del tiempo no informa de nada.

Cada umbral es el **percentil 75 de las alzas históricas a 2 semanas de ese
producto**. Así la tasa base queda entre el 10% y el 17% en todos, y "alerta
activa" significa lo mismo en todas partes.
"""
)

# --- Gráfica: probabilidad por serie, con emphasis en las activas ----------
etiquetas_y = vigentes["producto"] + " · " + vigentes["plaza"]
colores = [
    p["critico"] if activa else p["tenue"]
    for activa in vigentes["alerta_activa"]
]

fig = go.Figure(
    go.Bar(
        x=100 * vigentes["probabilidad"],
        y=etiquetas_y,
        orientation="h",
        marker={"color": colores, "line": {"width": 2, "color": p["superficie"]}},
        customdata=pd.DataFrame(
            {
                "umbral": 100 * vigentes["umbral_alza"],
                "precio": vigentes["precio_actual"],
                "objetivo": vigentes["precio_umbral"],
            }
        ),
        hovertemplate=(
            "<b>%{y}</b><br>Probabilidad: %{x:.0f}%<br>"
            "Umbral: +%{customdata[0]:.0f}%<br>"
            "De %{customdata[1]:,.0f} a %{customdata[2]:,.0f} COP/kg<extra></extra>"
        ),
    )
)
umbral_prob = 30
fig.add_vline(
    x=umbral_prob,
    line_width=2,
    line_color=p["tinta_secundaria"],
    annotation_text=f"Umbral de alerta {umbral_prob}%",
    annotation_position="top right",
    annotation_font_color=p["tinta_secundaria"],
)
fig.update_layout(
    title="Probabilidad de alza por encima del umbral del producto",
    hovermode="closest",
)
estilo_figura(fig, alto=max(420, 26 * len(vigentes)))
fig.update_xaxes(showgrid=True, gridcolor=p["rejilla"], ticksuffix="%", range=[0, 100])
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "En rojo, las series cuya probabilidad supera el umbral de alerta. En gris, "
    "el resto. El color marca estado, no identidad: no es una serie más."
)

# --- Tabla detallada --------------------------------------------------------
st.subheader("Detalle")
tabla = vigentes[
    [
        "producto", "plaza", "precio_actual", "umbral_alza",
        "precio_umbral", "probabilidad", "alerta_activa",
    ]
].copy()
tabla["umbral_alza"] = 100 * tabla["umbral_alza"]
tabla["probabilidad"] = 100 * tabla["probabilidad"]
tabla.columns = [
    "Producto", "Plaza", "Precio actual", "Umbral (+%)",
    "Precio de alerta", "Probabilidad (%)", "Activa",
]
st.dataframe(
    tabla.style.format(
        {
            "Precio actual": "{:,.0f}",
            "Precio de alerta": "{:,.0f}",
            "Umbral (+%)": "{:.0f}",
            "Probabilidad (%)": "{:.0f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# --- Historial de aciertos --------------------------------------------------
st.divider()
st.subheader("¿Han acertado estas alertas?")

desempeno = PROCESADO / "desempeno_alertas.csv"
reconciliadas = PROCESADO / "alertas_reconciliadas.csv"

if not reconciliadas.exists():
    falta_artefacto("alertas_reconciliadas.csv", "make reconcile")
else:
    recon = pd.read_csv(reconciliadas)
    resueltas = (
        recon[recon["estado"] == "resuelto"]
        if "estado" in recon.columns
        else pd.DataFrame()
    )

    if resueltas.empty:
        st.info(
            "Ninguna alerta se ha resuelto todavía: hacen falta 2 semanas desde su "
            "emisión. **Una alerta sin historial de aciertos es una opinión, no una "
            "medición**, así que esta sección seguirá vacía hasta entonces.",
            icon="⏳",
        )
    elif desempeno.exists():
        tabla_d = pd.read_csv(desempeno)
        if tabla_d.empty:
            st.info("Aún no hay suficientes alertas resueltas para resumir.", icon="⏳")
        else:
            st.dataframe(tabla_d, hide_index=True, use_container_width=True)
            st.caption(
                "Si el grupo 'alerta activa' no tiene una tasa de ocurrencia "
                "claramente mayor que 'sin alerta', la alerta no está aportando "
                "nada y conviene decirlo."
            )

st.divider()
st.warning(
    "**Estas probabilidades heredan el límite de los intervalos.** Se derivan de "
    "la calibración conformal, cuya cobertura cae del 82% al 45% en semanas de "
    "choque. La alerta es informativa en régimen normal y **no es de fiar durante "
    "un choque** — justo cuando un alza es más probable. Ver *Choques y límites*.",
    icon="⚠️",
)
