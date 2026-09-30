"""Piezas compartidas por las páginas de la app.

Dos reglas que gobiernan este módulo:

1. **La app nunca entrena ni descarga nada.** Lee artefactos ya calculados por
   el pipeline. Eso la hace instantánea, desplegable en Streamlit Community
   Cloud sin secretos, y hace imposible que la app y el backtest discrepen.
2. **Las rutas se derivan de `__file__`**, nunca absolutas, para que funcione
   igual en local que en el contenedor de Streamlit Cloud.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
PROCESADO = RAIZ / "data" / "processed"

AVISO = (
    "**Precios mayoristas (SIPSA–DANE), no precios al consumidor.** "
    "Los pronósticos son estimaciones con incertidumbre, no garantías."
)

# --- Paleta -----------------------------------------------------------------
# Slots categóricos en orden fijo. El orden es el mecanismo de seguridad para
# daltonismo, no una preferencia estética: no se reordena ni se cicla.

PALETA_CLARA = {
    "superficie": "#fcfcfb",
    "tinta": "#0b0b0b",
    "tinta_secundaria": "#52514e",
    "tenue": "#898781",
    "rejilla": "#e1e0d9",
    "eje": "#c3c2b7",
    "serie_1": "#2a78d6",  # azul: el modelo
    "serie_2": "#eb6834",  # naranja: el naive
    "serie_3": "#1baf7a",
    "bueno": "#0ca30c",
    "advertencia": "#fab219",
    "serio": "#ec835a",
    "critico": "#d03b3b",
}

PALETA_OSCURA = {
    "superficie": "#1a1a19",
    "tinta": "#ffffff",
    "tinta_secundaria": "#c3c2b7",
    "tenue": "#898781",
    "rejilla": "#2c2c2a",
    "eje": "#383835",
    "serie_1": "#3987e5",
    "serie_2": "#d95926",
    "serie_3": "#199e70",
    "bueno": "#0ca30c",
    "advertencia": "#fab219",
    "serio": "#ec835a",
    "critico": "#d03b3b",
}


def rgba(hex_color: str, alfa: float) -> str:
    """Convierte un hex a `rgba()` con la opacidad pedida.

    Se usa para las bandas de incertidumbre: un relleno translúcido del mismo
    hue que su línea, no un color distinto.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alfa})"


def paleta() -> dict[str, str]:
    """Paleta del tema con el que la página se está pintando realmente.

    El orden importa: manda `.streamlit/config.toml`. `st.context.theme.type`
    refleja la preferencia del navegador, que puede decir "dark" mientras la
    página se pinta clara porque el config la fija — y entonces saldrían
    gráficas oscuras sobre fondo claro, con el texto en blanco sobre blanco.

    Los pasos oscuros son una selección propia para la superficie oscura, no un
    volteo automático de los claros.
    """
    base = None
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = None
    if base in ("light", "dark"):
        return PALETA_OSCURA if base == "dark" else PALETA_CLARA

    try:
        tipo = st.context.theme.type  # Streamlit >= 1.46
    except Exception:
        tipo = "light"
    return PALETA_OSCURA if tipo == "dark" else PALETA_CLARA


def _mezclar(hex_color: str, alfa: float) -> str:
    """Mezcla un color con la superficie activa.

    Los mapas de calor llevan el valor escrito dentro de cada celda, así que la
    celda no puede llegar al paso más saturado del ramp: el texto perdería
    contraste justo en los extremos, que son las celdas que más se miran.
    Mezclar con la superficie mantiene el hue y deja la tinta legible en todas.
    """
    superficie = paleta()["superficie"].lstrip("#")
    base = [int(superficie[i : i + 2], 16) for i in (0, 2, 4)]
    h = hex_color.lstrip("#")
    color = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    mezcla = [round(alfa * c + (1 - alfa) * b) for c, b in zip(color, base, strict=True)]
    return "#{:02x}{:02x}{:02x}".format(*mezcla)


def escala_divergente(alfa: float = 0.62) -> list[list]:
    """Escala azul <-> rojo con gris neutro en el centro.

    Dos hues que se leen como opuestos y un punto medio que se lee como "nada".
    Nunca un hue en el centro, nunca dos polos fríos.
    """
    p = paleta()
    neutro = "#f0efec" if p["superficie"] == PALETA_CLARA["superficie"] else "#383835"
    return [
        [0.0, _mezclar(p["critico"], alfa)],
        [0.5, neutro],
        [1.0, _mezclar(p["serie_1"], alfa)],
    ]


def estilo_figura(fig, *, alto: int = 420, titulo_y: str = "") -> None:
    """Aplica la cromía y el cromo recesivo comunes a todas las gráficas."""
    p = paleta()
    fig.update_layout(
        height=alto,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "color": p["tinta_secundaria"],
            "size": 13,
        },
        # Margen superior generoso: el título ocupa la primera línea y la
        # leyenda la segunda; con menos se pisan.
        margin={"l": 8, "r": 8, "t": 78, "b": 8},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "title_text": "",
        },
        title={"font": {"color": p["tinta"], "size": 16}, "y": 0.97, "yanchor": "top"},
    )
    # Rejilla y ejes en hairline sólido, un tono por encima de la superficie.
    fig.update_xaxes(
        showgrid=False, linecolor=p["eje"], linewidth=1, ticks="outside",
        tickcolor=p["eje"], tickfont={"color": p["tenue"]},
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=p["rejilla"], gridwidth=1, zeroline=False,
        linecolor="rgba(0,0,0,0)", tickfont={"color": p["tenue"]},
        title={"text": titulo_y, "font": {"color": p["tenue"], "size": 12}},
    )


# --- Carga de datos ---------------------------------------------------------


def _leer(nombre: str, *, parquet: bool = False) -> pd.DataFrame | None:
    """Lee un artefacto del pipeline, o devuelve None si no se ha generado."""
    ruta = PROCESADO / nombre
    if not ruta.exists():
        return None
    return pd.read_parquet(ruta) if parquet else pd.read_csv(ruta)


@st.cache_data(show_spinner=False)
def cargar_panel() -> pd.DataFrame | None:
    df = _leer("semanal.parquet", parquet=True)
    if df is not None:
        df["semana"] = pd.to_datetime(df["semana"])
    return df


@st.cache_data(show_spinner=False)
def cargar_pronosticos() -> pd.DataFrame | None:
    df = _leer("forecasts_log.csv")
    if df is None:
        return None
    for col in ("semana_origen", "semana_objetivo"):
        df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data(show_spinner=False)
def cargar_metricas_horizonte() -> pd.DataFrame | None:
    return _leer("metricas_por_horizonte.csv")


@st.cache_data(show_spinner=False)
def cargar_metricas_serie() -> pd.DataFrame | None:
    return _leer("metricas_por_serie.csv")


@st.cache_data(show_spinner=False)
def cargar_cobertura() -> pd.DataFrame | None:
    return _leer("cobertura_intervalos.csv")


@st.cache_data(show_spinner=False)
def cargar_degradacion() -> pd.DataFrame | None:
    return _leer("degradacion_por_regimen.csv")


@st.cache_data(show_spinner=False)
def cargar_semanas_choque() -> pd.DataFrame | None:
    df = _leer("semanas_choque.csv")
    if df is not None:
        df["semana"] = pd.to_datetime(df["semana"])
    return df


@st.cache_data(show_spinner=False)
def cargar_alertas() -> pd.DataFrame | None:
    df = _leer("alertas_log.csv")
    if df is None:
        return None
    for col in ("semana_origen", "semana_limite"):
        df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data(show_spinner=False)
def cargar_desempeno_vivo() -> pd.DataFrame | None:
    return _leer("desempeno_en_vivo.csv")


@st.cache_data(show_spinner=False)
def cargar_reporte_series() -> pd.DataFrame | None:
    return _leer("reporte_series.csv")


@st.cache_data(show_spinner=False)
def cargar_importancias() -> pd.DataFrame | None:
    return _leer("importancia_features.csv")


@st.cache_data(show_spinner=False)
def cargar_ablacion() -> pd.DataFrame | None:
    return _leer("ablacion.csv")


@st.cache_resource(show_spinner=False)
def etiquetas() -> tuple[dict[str, str], dict[str, str]]:
    """Mapa id -> etiqueta legible para productos y plazas.

    Sale del propio panel, así que si cambia `config/products.yaml` la app se
    actualiza sola sin duplicar el diccionario.
    """
    panel = cargar_panel()
    if panel is None:
        return {}, {}
    prod = panel.drop_duplicates("producto_id").set_index("producto_id")["producto"].to_dict()
    plaza = (
        panel.drop_duplicates("plaza_id")
        .assign(nombre=lambda d: d["ciudad"] + " · " + d["plaza"])
        .set_index("plaza_id")["nombre"]
        .to_dict()
    )
    return prod, plaza


# --- Componentes ------------------------------------------------------------


def encabezado(titulo: str, subtitulo: str = "") -> None:
    """Título de página más el aviso obligatorio y la fecha del último dato."""
    st.title(titulo)
    if subtitulo:
        st.caption(subtitulo)
    st.warning(AVISO, icon="⚠️")
    _pie_de_datos()


def _pie_de_datos() -> None:
    """Fecha del último dato disponible. Va en todas las páginas, siempre."""
    panel = cargar_panel()
    if panel is None:
        st.error(
            "No encuentro `data/processed/semanal.parquet`. "
            "Ejecuta `make pipeline` para generarlo.",
            icon="🚫",
        )
        st.stop()
    ultima = panel["semana"].max()
    st.caption(
        f"Último dato disponible: **semana del {ultima.date():%d/%m/%Y}** · "
        f"Fuente: servicio web SIPSA del DANE (CC BY-SA 4.0)"
    )


def falta_artefacto(nombre: str, comando: str) -> None:
    """Mensaje uniforme cuando un artefacto del pipeline no se ha generado."""
    st.info(
        f"Todavía no existe `{nombre}`. Genéralo con `{comando}`.",
        icon="ℹ️",
    )


def selector_serie(clave: str) -> tuple[str, str]:
    """Selector de producto y plaza compartido por las páginas que lo necesitan."""
    prod_map, plaza_map = etiquetas()
    panel = cargar_panel()
    col1, col2 = st.columns(2)
    with col1:
        producto = st.selectbox(
            "Producto",
            sorted(prod_map, key=lambda k: prod_map[k]),
            format_func=lambda k: prod_map[k],
            key=f"{clave}_producto",
        )
    disponibles = sorted(
        panel.loc[panel["producto_id"] == producto, "plaza_id"].unique(),
        key=lambda k: plaza_map.get(k, k),
    )
    with col2:
        plaza = st.selectbox(
            "Plaza",
            disponibles,
            format_func=lambda k: plaza_map.get(k, k),
            key=f"{clave}_plaza",
        )
    return producto, plaza
