"""Página 3: dónde se rompe el modelo, por qué, y cómo está hecho."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from _comun import (
    cargar_degradacion,
    cargar_importancias,
    cargar_semanas_choque,
    encabezado,
    estilo_figura,
    falta_artefacto,
    paleta,
    rgba,
)

encabezado("Choques y límites", "Dónde falla el modelo, por qué, y cómo está construido")

p = paleta()
semanas = cargar_semanas_choque()
degradacion = cargar_degradacion()

st.markdown(
    """
Un error promedio bajo puede esconder un modelo inútil justo cuando más se
necesita. Esta página separa las semanas **normales** de las **de choque** y
mide el daño por separado.
"""
)

# --- 1. Cuándo se movió todo el mercado a la vez ---------------------------
st.subheader("Semanas de choque")

if semanas is None:
    falta_artefacto("semanas_choque.csv", "make evaluate")
    st.stop()

st.markdown(
    """
La detección es **automática**, no una lista escrita a mano: una semana es de
choque si el movimiento **mediano de todo el panel** supera el percentil 90 de
su propia distribución histórica. Se usa la mediana para que una sola serie
enloquecida no marque la semana entera: un choque de verdad mueve todo.
"""
)

fig = go.Figure()
choques = semanas[semanas["es_choque"]]

fig.add_trace(
    go.Scatter(
        x=semanas["semana"],
        y=semanas["mov_mediano_pct"],
        mode="lines",
        name="Movimiento mediano del panel",
        line={"color": rgba(p["tenue"], 0.9), "width": 2},
        hovertemplate="%{x|%d/%m/%Y}: %{y:.1f}%<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=choques["semana"],
        y=choques["mov_mediano_pct"],
        mode="markers",
        name="Semana de choque",
        marker={"color": p["critico"], "size": 9, "line": {"width": 2, "color": p["superficie"]}},
        hovertemplate="%{x|%d/%m/%Y}: %{y:.1f}%<extra>Choque</extra>",
    )
)
if "umbral" in semanas.columns:
    # El umbral se guarda en log; aquí se muestra en % para poder leerlo.
    umbral_pct = float(100 * np.expm1(semanas["umbral"].iloc[0]))
    fig.add_hline(
        y=umbral_pct,
        line_width=1,
        line_color=p["eje"],
        annotation_text=f"Umbral de choque: {umbral_pct:.1f}%",
        annotation_position="top left",
        annotation_font_color=p["tenue"],
    )
fig.update_layout(title="Volatilidad semanal del panel completo", hovermode="x unified")
estilo_figura(fig, alto=380, titulo_y="Cambio mediano absoluto (%)")
st.plotly_chart(fig, use_container_width=True)

top = choques.nlargest(6, "mov_mediano_pct")[
    ["semana", "mov_mediano_pct", "episodio"]
].copy()
top["semana"] = top["semana"].dt.date
top.columns = ["Semana", "Movimiento mediano (%)", "Episodio"]
top["Episodio"] = top["Episodio"].fillna("— sin etiquetar —")
st.dataframe(
    top.style.format({"Movimiento mediano (%)": "{:.1f}"}),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    f"**{len(choques)} semanas de choque de {len(semanas)}.** Las más violentas "
    "coinciden con episodios conocidos que la detección no conocía: el paro "
    "nacional de 2021, el inicio de la cuarentena de 2020 y el pico de 2024-Q3."
)

# --- 2. Cuánto se degrada el error -----------------------------------------
st.subheader("Todos los modelos se rompen igual")

if degradacion is None:
    falta_artefacto("degradacion_por_regimen.csv", "make evaluate")
else:
    # Se comparan solo los modelos competitivos: el naive estacional (MASE ~5)
    # y los promedios móviles estirarían el eje y aplastarían la comparación
    # que importa, que es entre modelos que sí están cerca del naive.
    COMPETITIVOS = ["naive", "auto_arima", "ets", "lgbm_cuantil", "lgbm_conformal"]
    h1 = degradacion[(degradacion["h"] == 1) & degradacion["modelo"].isin(COMPETITIVOS)]
    piv = h1.pivot_table(index="modelo", columns="regimen", values="mase")
    piv = piv.dropna().sort_values("normal")

    fig2 = go.Figure()
    for regimen, color in (("normal", p["serie_1"]), ("choque", p["serie_2"])):
        if regimen not in piv.columns:
            continue
        fig2.add_trace(
            go.Bar(
                x=piv.index,
                y=piv[regimen],
                name=f"Semanas {regimen}es" if regimen == "normal" else "Semanas de choque",
                marker={"color": color, "line": {"width": 2, "color": p["superficie"]}},
                hovertemplate="%{x}: MASE %{y:.2f}<extra>" + regimen + "</extra>",
            )
        )
    fig2.update_layout(barmode="group", title="MASE a 1 semana por régimen", hovermode="closest")
    estilo_figura(fig2, alto=400, titulo_y="MASE")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Solo se muestran los modelos competitivos; el naive estacional y los "
        "promedios móviles quedan fuera por escala. "
        "El error se triplica para todos, y las diferencias entre modelos se "
        "vuelven ruido. **Si un paro bloquea las vías, el precio de la semana "
        "siguiente no está en la historia.** Ningún modelo de este proyecto "
        "puede anticipar eso, y ninguno pretende hacerlo."
    )

    # --- 3. El hallazgo incómodo: los intervalos también se rompen ---------
    st.subheader("Y los intervalos también se rompen")

    conf = degradacion[
        (degradacion["modelo"] == "lgbm_conformal") & degradacion["cobertura_pct"].notna()
    ]
    if not conf.empty:
        cob = conf.pivot_table(index="h", columns="regimen", values="cobertura_pct")
        fig3 = go.Figure()
        for regimen, color in (("normal", p["serie_1"]), ("choque", p["critico"])):
            if regimen not in cob.columns:
                continue
            fig3.add_trace(
                go.Bar(
                    x=cob.index,
                    y=cob[regimen],
                    name=f"Semanas {regimen}es" if regimen == "normal" else "Semanas de choque",
                    marker={"color": color, "line": {"width": 2, "color": p["superficie"]}},
                    hovertemplate="h=%{x}: %{y:.1f}%<extra>" + regimen + "</extra>",
                )
            )
        fig3.add_hline(
            y=80,
            line_width=2,
            line_color=p["tinta_secundaria"],
            annotation_text="Nivel nominal 80%",
            annotation_position="top left",
            annotation_font_color=p["tinta_secundaria"],
        )
        fig3.update_layout(
            barmode="group",
            title="Cobertura del intervalo del 80%, por régimen",
            hovermode="closest",
        )
        estilo_figura(fig3, alto=400, titulo_y="Cobertura empírica (%)")
        fig3.update_xaxes(
            title={"text": "Horizonte (semanas)", "font": {"color": p["tenue"], "size": 12}},
            dtick=1,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.error(
            "**La cobertura global del 81% es un promedio que esconde el fallo.** "
            "En semanas normales el intervalo cumple; en semanas de choque, a un "
            "horizonte de una semana, falla más de la mitad de las veces.\n\n"
            "No es un error de implementación: la predicción conformal garantiza "
            "cobertura **marginal**, no condicional. Promete acertar el 80% de las "
            "veces en promedio, no el 80% en cada régimen. Con series de tiempo, "
            "donde los choques rompen la intercambiabilidad, esa distinción deja "
            "de ser teórica.\n\n"
            "**Lectura práctica: estos intervalos son útiles en condiciones "
            "normales y no son de fiar durante un choque, que es exactamente "
            "cuando alguien querría consultarlos.**",
            icon="🚨",
        )

# --- 4. Cómo está hecho ----------------------------------------------------
st.divider()
st.subheader("Cómo funciona el método")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
**Los datos.** Servicio web SOAP oficial del DANE, actualizado a diario.
Precios ya expresados en COP por kilogramo, agregados de diario a semanal con
la **mediana** (no el promedio: el histórico tiene errores de digitación de un
solo día que el promedio propagaría a toda la semana).

**La validación.** Rolling-origin con ventana expansiva: 61 orígenes, primer
origen tras 104 semanas, horizontes 1–4. Nunca un split aleatorio, que
entrenaría con semanas posteriores a las que predice.

**Contra qué se compara.** El baseline **naive** —repetir el último precio— no
es un hombre de paja: es muy difícil de batir y define la escala del MASE.
"""
    )
with col2:
    st.markdown(
        """
**El modelo.** LightGBM global entrenado sobre todas las series a la vez, con
`producto` y `plaza` como categóricas. Predice el **log-cambio**
`log(y[t+h]/y[t])`, no el nivel: así un aguacate a 7.000 COP/kg y una zanahoria
a 1.000 hablan el mismo idioma.

**Los intervalos.** Conformal split: el ancho se calibra con residuos **fuera
de muestra** de 26 semanas que el modelo nunca vio.

**Sin fuga de información.** Toda feature en la semana `t` usa solo datos hasta
`t`. Incluso el relleno de semanas faltantes ocurre dentro de la ventana de
entrenamiento de cada fold. Hay tests que corrompen el futuro y exigen que los
pronósticos no se muevan.
"""
    )

importancias = cargar_importancias()
if importancias is not None and not importancias.empty:
    with st.expander("¿En qué se fija el modelo?"):
        top_f = importancias.head(10).iloc[::-1]
        fig4 = go.Figure(
            go.Bar(
                x=top_f["ganancia"],
                y=top_f["feature"],
                orientation="h",
                marker={"color": p["serie_1"], "line": {"width": 2, "color": p["superficie"]}},
                hovertemplate="%{y}: %{x:.0f}<extra></extra>",
            )
        )
        fig4.update_layout(title="Importancia media de las features", hovermode="closest")
        estilo_figura(fig4, alto=380)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            "Dominan `ret_52`, `ret_26` y `rel_ma52`: el modelo sí encuentra "
            "estructura anual, pero **relativa** (dónde está el precio frente a su "
            "propio nivel de hace un año), no de nivel absoluto. Por eso el naive "
            "estacional, que copia el precio del año pasado tal cual, fracasa."
        )

st.divider()
st.subheader("Limitaciones que conviene tener presentes")
st.markdown(
    """
- **Son precios mayoristas.** No son lo que paga un hogar en la tienda.
- **La historia empieza en 2020-02.** Incluye la pandemia y el paro de 2021: dos
  rupturas de régimen en seis años de datos.
- **No hay arroz, fríjol, huevo, panela ni maíz.** El endpoint con historia
  suficiente solo cubre frutas, verduras y tubérculos.
- **A una semana, el modelo apenas mejora al naive**, y en la mayoría de
  productos individuales no lo mejora en absoluto.
- **Ninguna variable exógena todavía**: ni clima, ni ENSO, ni festivos, ni
  volumen de abastecimiento.
"""
)
