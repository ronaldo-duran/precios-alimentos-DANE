"""Página 2: qué tan bien funciona, medido contra el naive y contra la realidad."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _comun import (
    cargar_ablacion,
    cargar_cobertura,
    cargar_desempeno_vivo,
    cargar_metricas_horizonte,
    cargar_metricas_serie,
    cargar_pronosticos,
    encabezado,
    escala_divergente,
    estilo_figura,
    etiquetas,
    falta_artefacto,
    paleta,
)

encabezado("Desempeño", "MASE por producto y horizonte, cobertura de intervalos y registro en vivo")

p = paleta()
prod_map, _ = etiquetas()
por_h = cargar_metricas_horizonte()
por_serie = cargar_metricas_serie()
cobertura = cargar_cobertura()

MODELO = "lgbm_cuantil"


def _formatear_desempeno(tabla: pd.DataFrame) -> pd.DataFrame:
    """Prepara la tabla de desempeño en vivo para mostrarla."""
    cols = [
        "modelo", "h", "n", "mase", "mae", "sesgo",
        "cobertura_pct", "n_versiones", "primera", "ultima",
    ]
    existentes = [c for c in cols if c in tabla.columns]
    salida = tabla[existentes].copy()
    salida = salida.rename(
        columns={
            "modelo": "Modelo",
            "h": "Horizonte",
            "n": "n",
            "mase": "MASE",
            "mae": "MAE (COP/kg)",
            "sesgo": "Sesgo (COP/kg)",
            "cobertura_pct": "Cobertura (%)",
            "n_versiones": "Versiones",
            "primera": "Desde",
            "ultima": "Hasta",
        }
    )
    formatos = {
        c: f
        for c, f in {
            "MASE": "{:.3f}",
            "MAE (COP/kg)": "{:,.0f}",
            "Sesgo (COP/kg)": "{:+,.0f}",
            "Cobertura (%)": "{:.1f}",
        }.items()
        if c in salida.columns
    }
    return salida.style.format(formatos)

st.markdown(
    """
El **MASE** compara el error del modelo contra el del naive dentro de su propio
periodo de entrenamiento. Lo que importa aquí no es su valor absoluto sino la
comparación entre modelos sobre los mismos orígenes.
"""
)

# --- 1. Mejora frente al naive, por horizonte -------------------------------
if por_h is None:
    falta_artefacto("metricas_por_horizonte.csv", "make evaluate")
    st.stop()

tabla = por_h.pivot(index="modelo", columns="h", values="mase")
naive = tabla.loc["naive"]
mejora = (100 * (1 - tabla.div(naive, axis=1))).round(1)
mejora = mejora.drop(index="naive").sort_values(1, ascending=True)

st.subheader("¿Alguien le gana al naive?")

# El naive estacional pierde por ~290% y los promedios móviles por 35-78%.
# Dejarlos en la gráfica comprime a cero el rango donde está la historia
# (±8%), así que se grafican los competitivos y los demás van a la tabla.
UMBRAL_LEGIBLE = -15.0
competitivos = mejora[mejora.min(axis=1) >= UMBRAL_LEGIBLE]
descartados = mejora[mejora.min(axis=1) < UMBRAL_LEGIBLE]

fig = go.Figure(
    go.Heatmap(
        z=competitivos.values,
        x=[f"h={h}" for h in competitivos.columns],
        y=list(competitivos.index),
        colorscale=escala_divergente(),
        zmid=0,
        text=[[f"{v:+.1f}%" for v in fila] for fila in competitivos.values],
        texttemplate="%{text}",
        textfont={"size": 13, "color": p["tinta"]},
        xgap=2,
        ygap=2,
        hovertemplate="%{y} · %{x}: %{z:+.1f}%<extra></extra>",
        colorbar={
            "title": {"text": "% vs naive", "font": {"size": 11}},
            "thickness": 12,
            "outlinewidth": 0,
            "tickfont": {"size": 11},
        },
    )
)
fig.update_layout(
    title="Diferencia de MASE frente al naive (azul = mejor que el naive)",
    hovermode="closest",
)
estilo_figura(fig, alto=340)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Azul = mejor que el naive, rojo = peor. Fíjate en los números: **la "
    "diferencia entre el mejor modelo y el naive cabe en unos pocos puntos "
    "porcentuales.**"
)

if not descartados.empty:
    peores = ", ".join(
        f"`{m}` ({descartados.loc[m].min():.0f}% a {descartados.loc[m].max():.0f}%)"
        for m in descartados.index
    )
    st.caption(
        f"Fuera de la gráfica por escala, porque pierden por demasiado: {peores}. "
        "Están en la tabla de abajo."
    )

with st.expander("Ver el MASE de todos los modelos"):
    completa = tabla.copy()
    completa.columns = [f"h={h}" for h in completa.columns]
    st.dataframe(
        completa.sort_values("h=1").style.format("{:.3f}"),
        use_container_width=True,
    )

# --- 2. El promedio esconde la verdad: desglose por producto ----------------
st.subheader("La mejora no está repartida")

if por_serie is not None:
    sel = por_serie[por_serie["modelo"].isin([MODELO, "naive"])]
    agg = (
        sel.groupby(["modelo", "producto_id", "h"], as_index=False)["mase"]
        .mean()
        .pivot_table(index=["producto_id", "h"], columns="modelo", values="mase")
        .reset_index()
    )
    agg["mejora"] = 100 * (1 - agg[MODELO] / agg["naive"])
    pivote = agg.pivot(index="producto_id", columns="h", values="mejora")
    pivote.index = [prod_map.get(i, i) for i in pivote.index]
    pivote = pivote.sort_values(1, ascending=True)

    fig2 = go.Figure(
        go.Heatmap(
            z=pivote.values,
            x=[f"h={h}" for h in pivote.columns],
            y=list(pivote.index),
            colorscale=escala_divergente(),
            zmid=0,
            text=[[f"{v:+.1f}%" for v in fila] for fila in pivote.values],
            texttemplate="%{text}",
            textfont={"size": 13, "color": p["tinta"]},
            xgap=2,
            ygap=2,
            hovertemplate="%{y} · %{x}: %{z:+.1f}%<extra></extra>",
            colorbar={
                "title": {"text": "% vs naive", "font": {"size": 11}},
                "thickness": 12,
                "outlinewidth": 0,
                "tickfont": {"size": 11},
            },
        )
    )
    fig2.update_layout(
        title=f"Mejora de {MODELO} sobre el naive, por producto",
        hovermode="closest",
    )
    estilo_figura(fig2, alto=400)
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

    ganadores = pivote[pivote[1] > 0].index.tolist()
    st.caption(
        f"Solo **{len(ganadores)} de {len(pivote)} productos** mejoran a una semana "
        f"({', '.join(ganadores) if ganadores else 'ninguno'}). "
        "En el resto, el modelo empata o pierde contra repetir el último precio. "
        "El promedio global es positivo porque un par de productos arrastran al resto."
    )

# --- 3. Cobertura de los intervalos -----------------------------------------
st.subheader("¿Los intervalos cubren lo que prometen?")

if cobertura is None:
    falta_artefacto("cobertura_intervalos.csv", "make evaluate")
else:
    nominal = float(cobertura["nivel_nominal_pct"].iloc[0])
    fig3 = go.Figure()
    for modelo, g in cobertura.groupby("modelo"):
        color = p["serie_1"] if "conformal" in modelo else p["serie_2"]
        fig3.add_trace(
            go.Bar(
                x=g["h"],
                y=g["cobertura_pct"],
                name=modelo,
                marker={"color": color, "line": {"width": 2, "color": p["superficie"]}},
                hovertemplate="h=%{x}: %{y:.1f}%<extra>" + modelo + "</extra>",
            )
        )
    fig3.add_hline(
        y=nominal,
        line_width=2,
        line_color=p["tinta_secundaria"],
        annotation_text=f"Nivel nominal {nominal:.0f}%",
        annotation_position="top left",
        annotation_font_color=p["tinta_secundaria"],
    )
    fig3.update_layout(
        barmode="group",
        title="Cobertura empírica de los intervalos",
        hovermode="closest",
    )
    estilo_figura(fig3, alto=400, titulo_y="% de valores reales dentro del intervalo")
    fig3.update_xaxes(
        title={"text": "Horizonte (semanas)", "font": {"color": p["tenue"], "size": 12}},
        dtick=1,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "Un intervalo que promete el 80% y cubre el 65% está mintiendo. La "
        "calibración conformal se mantiene pegada a la línea; la regresión "
        "cuantílica se queda corta y empeora con el horizonte. Por eso el modelo "
        "en producción es el conformal, aunque su pronóstico puntual sea algo peor."
    )

    with st.expander("Ver el ancho de los intervalos"):
        anchos = cobertura[["modelo", "h", "cobertura_pct", "amplitud_pct"]].copy()
        anchos.columns = [
            "Modelo", "Horizonte", "Cobertura (%)", "Ancho medio (% del precio)",
        ]
        st.dataframe(
            anchos.style.format(
                {"Cobertura (%)": "{:.1f}", "Ancho medio (% del precio)": "{:.1f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "La honestidad cuesta ancho: el intervalo conformal es más ancho que el "
            "cuantílico en todos los horizontes. Un intervalo estrecho que no cubre "
            "no es mejor, es solo más cómodo de mirar."
        )

# --- 4. ¿Aportan las variables exógenas? ------------------------------------
st.divider()
st.subheader("¿Aportan las variables exógenas?")

ablacion = cargar_ablacion()
if ablacion is None:
    falta_artefacto("ablacion.csv", "make ablacion")
else:
    st.markdown(
        """
Mismo walk-forward, mismos orígenes, mismos hiperparámetros y misma semilla;
lo único que cambia es el bloque de features. Cualquier diferencia es
atribuible al bloque.

La columna que manda es **reservados**: son los orígenes que no se usaron para
elegir hiperparámetros. Si el signo no coincide entre las dos columnas, la
mejora no es de fiar.
"""
    )
    vista = ablacion[ablacion["variante"] != "base"].copy()
    tabla_abl = vista[
        ["variante", "h", "n_features", "delta_todos_pct", "delta_reservados_pct"]
    ].rename(
        columns={
            "variante": "Variante",
            "h": "Horizonte",
            "n_features": "Features",
            "delta_todos_pct": "Δ todos (%)",
            "delta_reservados_pct": "Δ reservados (%)",
        }
    )
    st.dataframe(
        tabla_abl.style.format({"Δ todos (%)": "{:+.2f}", "Δ reservados (%)": "{:+.2f}"}),
        hide_index=True,
        use_container_width=True,
    )
    mejor = vista.loc[vista["delta_reservados_pct"].idxmax()]
    st.caption(
        f"Mejor caso: `{mejor['variante']}` a h={int(mejor['h'])}, "
        f"{mejor['delta_reservados_pct']:+.2f}% sobre los orígenes reservados. "
        "Positivo significa mejor que el modelo sin exógenas."
    )


# --- 5. El registro en vivo -------------------------------------------------
st.divider()
st.subheader("Registro en vivo")

st.markdown(
    """
Un backtest siempre se puede repetir hasta que salga bien. Un pronóstico
**escrito antes de que ocurriera la semana**, no. Esta tabla acumula el error
real de los pronósticos registrados en `forecasts_log.csv`.

Si una misma semana recibió pronósticos de varias versiones del modelo, solo
cuenta el más reciente: el que la app estaba mostrando cuando llegó el dato
real. Los demás siguen en el log, pero promediarlos le daría doble peso a esa
semana.
"""
)

vivo = cargar_desempeno_vivo()
pron = cargar_pronosticos()

if pron is not None and "procedencia" in pron.columns:
    n_vivo = int((pron["procedencia"] == "vivo").sum())
    st.caption(f"Pronósticos en vivo registrados: **{n_vivo}**")

if vivo is None or vivo.empty:
    st.info(
        "Todavía no hay ningún pronóstico en vivo resuelto. El registro empieza "
        "vacío por definición: la primera fila no se puede evaluar hasta que pase "
        "la semana objetivo.",
        icon="⏳",
    )
else:
    en_vivo = vivo[vivo["procedencia"] == "vivo"] if "procedencia" in vivo.columns else vivo
    backfill = (
        vivo[vivo["procedencia"] == "backfill"]
        if "procedencia" in vivo.columns
        else pd.DataFrame()
    )

    if en_vivo.empty:
        st.info(
            "Aún no hay pronósticos **en vivo** resueltos; los primeros se "
            "resolverán cuando el DANE publique la semana objetivo.",
            icon="⏳",
        )
    else:
        st.dataframe(_formatear_desempeno(en_vivo), hide_index=True, use_container_width=True)

    if not backfill.empty:
        with st.expander("Desempeño del sembrado (walk-forward, valor descriptivo)"):
            st.caption(
                "Estos pronósticos son igual de out-of-sample —ningún modelo vio "
                "datos posteriores a su origen— pero **no se escribieron antes de "
                "los hechos**, así que no tienen el mismo valor probatorio. Nunca "
                "se mezclan con los de arriba."
            )
            st.dataframe(
                _formatear_desempeno(backfill), hide_index=True, use_container_width=True
            )
