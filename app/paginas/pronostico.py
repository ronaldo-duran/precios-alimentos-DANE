"""Página 1: el pronóstico vigente de una serie, con su banda y el naive."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _comun import (
    cargar_panel,
    cargar_pronosticos,
    encabezado,
    estilo_figura,
    etiquetas,
    falta_artefacto,
    paleta,
    rgba,
    selector_serie,
)

encabezado("Pronóstico", "Histórico reciente, pronóstico a 1–4 semanas y banda de incertidumbre")

panel = cargar_panel()
pron = cargar_pronosticos()
prod_map, plaza_map = etiquetas()

producto, plaza = selector_serie("pronostico")
semanas_historia = st.slider("Semanas de histórico a mostrar", 26, 208, 78, step=26)

serie = panel[
    (panel["producto_id"] == producto) & (panel["plaza_id"] == plaza)
].sort_values("semana")
historia = serie.dropna(subset=["precio_kg"]).tail(semanas_historia)

if pron is None:
    falta_artefacto("forecasts_log.csv", "make forecast")
    st.stop()

vivos = pron[pron["procedencia"] == "vivo"] if "procedencia" in pron.columns else pron
ultimo_origen = vivos["semana_origen"].max()
actual = vivos[
    (vivos["producto_id"] == producto)
    & (vivos["plaza_id"] == plaza)
    & (vivos["semana_origen"] == ultimo_origen)
]

if actual.empty:
    st.info("No hay pronóstico vigente para esta serie.", icon="ℹ️")
    st.stop()

modelo = actual[actual["modelo"] != "naive"].sort_values("h")
naive = actual[actual["modelo"] == "naive"].sort_values("h")
p = paleta()

# --- Tarjetas: el número es la noticia, no un gráfico de una barra ----------
precio_actual = float(historia["precio_kg"].iloc[-1])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio actual (COP/kg)", f"{precio_actual:,.0f}".replace(",", "."))
if not modelo.empty:
    h4 = modelo[modelo["h"] == modelo["h"].max()].iloc[0]
    cambio = 100 * (h4["y_pred"] / precio_actual - 1)
    c2.metric(
        f"Pronóstico a {int(h4['h'])} semanas",
        f"{h4['y_pred']:,.0f}".replace(",", "."),
        f"{cambio:+.1f}%",
    )
    c3.metric(
        "Banda del 80%",
        f"{h4['lo']:,.0f} – {h4['hi']:,.0f}".replace(",", "."),
    )
c4.metric("Semana de origen", f"{ultimo_origen.date():%d/%m/%Y}")

# --- Gráfica ----------------------------------------------------------------
fig = go.Figure()

# Banda primero, para que quede por detrás de las líneas.
if not modelo.empty:
    puente = pd.concat([
        pd.DataFrame({
            "semana_objetivo": [ultimo_origen],
            "lo": [precio_actual],
            "hi": [precio_actual],
        }),
        modelo[["semana_objetivo", "lo", "hi"]],
    ])
    fig.add_trace(
        go.Scatter(
            x=list(puente["semana_objetivo"]) + list(puente["semana_objetivo"])[::-1],
            y=list(puente["hi"]) + list(puente["lo"])[::-1],
            fill="toself",
            fillcolor=rgba(p["serie_1"], 0.15),
            line={"width": 0},
            hoverinfo="skip",
            name="Banda del 80%",
            showlegend=True,
        )
    )

fig.add_trace(
    go.Scatter(
        x=historia["semana"],
        y=historia["precio_kg"],
        mode="lines",
        name="Histórico observado",
        line={"color": p["serie_1"], "width": 2},
        hovertemplate="%{y:,.0f} COP/kg<extra>Observado</extra>",
    )
)

if not modelo.empty:
    fig.add_trace(
        go.Scatter(
            x=[ultimo_origen, *modelo["semana_objetivo"]],
            y=[precio_actual, *modelo["y_pred"]],
            mode="lines+markers",
            name="Pronóstico del modelo",
            line={"color": p["serie_1"], "width": 2, "dash": "dot"},
            marker={"size": 8, "line": {"width": 2, "color": p["superficie"]}},
            hovertemplate="%{y:,.0f} COP/kg<extra>Modelo</extra>",
        )
    )

if not naive.empty:
    fig.add_trace(
        go.Scatter(
            x=[ultimo_origen, *naive["semana_objetivo"]],
            y=[precio_actual, *naive["y_pred"]],
            mode="lines",
            name="Baseline naive (último precio)",
            line={"color": p["serie_2"], "width": 2, "dash": "dash"},
            hovertemplate="%{y:,.0f} COP/kg<extra>Naive</extra>",
        )
    )

fig.add_vline(x=ultimo_origen.timestamp() * 1000, line_width=1, line_color=p["eje"])
fig.update_layout(
    title=f"{prod_map.get(producto, producto)} · {plaza_map.get(plaza, plaza)}"
)
estilo_figura(fig, alto=460, titulo_y="COP por kilogramo")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "La línea punteada azul es el pronóstico; la naranja discontinua es el naive, "
    "que simplemente repite el último precio observado. **Si ambas casi coinciden, "
    "el modelo está diciendo que no sabe más que el método trivial.** "
    "La banda cubre el 80% de las veces en semanas normales; durante un choque, "
    "mucho menos (ver *Choques y límites*)."
)

with st.expander("Ver la tabla de pronósticos"):
    tabla = modelo[["h", "semana_objetivo", "y_pred", "lo", "hi"]].copy()
    tabla["semana_objetivo"] = tabla["semana_objetivo"].dt.date
    tabla.columns = [
        "Horizonte (sem)", "Semana objetivo", "Pronóstico",
        "Límite inferior", "Límite superior",
    ]
    st.dataframe(
        tabla.style.format({
            "Pronóstico": "{:,.0f}",
            "Límite inferior": "{:,.0f}",
            "Límite superior": "{:,.0f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

