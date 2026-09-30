"""Construcción de features para el modelo global.

Dos decisiones que condicionan todo lo demás:

1. **El objetivo es el log-cambio**, no el nivel: `log(y[t+h] / y[t])`. Un modelo
   global ve aguacate a 7.000 COP/kg y zanahoria a 1.000 en las mismas filas; si
   predijera niveles, aprendería sobre todo a distinguir productos. Prediciendo
   el cambio relativo, todas las series hablan el mismo idioma y "empatar con el
   naive" equivale exactamente a predecir cambio cero.

2. **Toda feature en la fila `t` usa únicamente información hasta `t`.** No hay
   interpolación previa: las semanas faltantes se quedan como NaN y LightGBM las
   maneja de forma nativa. Interpolar el panel completo antes de trocear habría
   rellenado huecos con datos posteriores al origen. Está testeado en
   `tests/test_features.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Rezagos (en semanas) usados para retornos y medias móviles.
REZAGOS_RETORNO: tuple[int, ...] = (1, 2, 4, 8, 13, 26, 52)
VENTANAS_MEDIA: tuple[int, ...] = (4, 8, 13, 26, 52)
VENTANAS_VOLATILIDAD: tuple[int, ...] = (4, 13)

#: Columnas categóricas que identifican la serie dentro del modelo global.
CATEGORICAS: tuple[str, ...] = ("producto_id", "plaza_id")

#: Claves que viajan con cada fila pero NO entran al modelo.
#: `producto_id` y `plaza_id` no están aquí a propósito: sí son features, como
#: categóricas, y son lo que permite al modelo global transferir entre series.
LLAVES: tuple[str, ...] = ("semana", "t", "y")


def construir_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de features sobre el panel semanal reindexado.

    Args:
        panel: salida de `clean`, con una fila por serie y semana calendario
            (las semanas faltantes presentes con `precio_kg` NaN).

    Returns:
        Un `DataFrame` con las llaves, el precio observado `y` y las features.
        Las filas conservan todas las semanas: filtrar es tarea de quien
        entrena, que sabe dónde está su origen.
    """
    # `t` se mide contra un origen de calendario común a todo el panel, no
    # contra la posición dentro de cada serie: el modelo global compara series
    # entre sí, y dos series que empiecen en semanas distintas tendrían índices
    # incompatibles. Hoy todas arrancan el mismo lunes, pero el alcance cambia.
    referencia = panel["semana"].min()
    partes = [
        _features_de_serie(g, referencia)
        for _, g in panel.groupby(list(CATEGORICAS), observed=True)
    ]
    out = pd.concat(partes, ignore_index=True)

    out["semana_anio"] = out["semana"].dt.isocalendar().week.astype(int)
    # Seno y coseno para que la semana 52 y la 1 queden contiguas.
    angulo = 2 * np.pi * out["semana_anio"] / 52.0
    out["semana_sin"] = np.sin(angulo)
    out["semana_cos"] = np.cos(angulo)
    out["mes"] = out["semana"].dt.month

    for col in CATEGORICAS:
        out[col] = out[col].astype("category")

    log.info(
        "Features: %d filas, %d columnas predictoras, %d series",
        len(out),
        len(columnas_features(out)),
        out.groupby(list(CATEGORICAS), observed=True).ngroups,
    )
    return out


def _features_de_serie(g: pd.DataFrame, referencia: pd.Timestamp) -> pd.DataFrame:
    """Features de una sola serie. Todo se calcula hacia atrás desde `t`."""
    g = g.sort_values("semana").reset_index(drop=True)
    y = g["precio_kg"].astype(float)
    log_y = np.log(y.where(y > 0))

    out = pd.DataFrame(
        {
            "producto_id": g["producto_id"],
            "plaza_id": g["plaza_id"],
            "semana": g["semana"],
            "t": ((g["semana"] - referencia).dt.days // 7).astype(int),
            "y": y,
            "log_precio": log_y,
        }
    )

    # Retornos: cuánto se movió el precio respecto a k semanas atrás.
    for k in REZAGOS_RETORNO:
        out[f"ret_{k}"] = log_y - log_y.shift(k)

    # Nivel relativo a su propia media móvil: mide qué tan caro está hoy
    # frente a su historia reciente. Es la señal de reversión a la media.
    for v in VENTANAS_MEDIA:
        media = log_y.rolling(v, min_periods=max(2, v // 2)).mean()
        out[f"rel_ma{v}"] = log_y - media

    # Volatilidad reciente de los retornos semanales.
    ret1 = log_y - log_y.shift(1)
    for v in VENTANAS_VOLATILIDAD:
        out[f"vol_{v}"] = ret1.rolling(v, min_periods=max(2, v // 2)).std()

    # Dispersión intrasemanal: el DANE reporta mínimo y máximo del kg.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["amplitud_intrasemanal"] = (g["precio_kg_max"] - g["precio_kg_min"]) / y

    out["n_dias"] = g["n_dias"].astype(float)
    # Un outlier reciente avisa de que la serie viene de un episodio raro.
    out["outlier_4s"] = (
        g["flag_outlier"].astype(float).rolling(4, min_periods=1).max().fillna(0.0)
    )
    return out


def columnas_features(df: pd.DataFrame) -> list[str]:
    """Columnas que entran al modelo, en orden estable."""
    excluir = set(LLAVES) | {"semana_anio"}
    return [c for c in df.columns if c not in excluir and not c.startswith("target_")]


def anadir_objetivos(df: pd.DataFrame, horizontes: Sequence[int]) -> pd.DataFrame:
    """Añade una columna `target_h` por horizonte.

    Se calculan **una sola vez sobre el panel completo**, lo que sí mira al
    futuro: `target_h` en la fila `t` es `log(y[t+h]/y[t])`. Eso no es leakage
    porque el filtrado posterior (`filas_entrenables`) solo conserva filas cuyo
    objetivo ya había ocurrido antes del origen. Separar el cálculo del filtrado
    evita perder la última fila entrenable de cada ventana.
    """
    df = df.copy()
    for h in horizontes:
        df[f"target_{h}"] = objetivo_log_cambio(df, h)
    return df


def filas_entrenables(df: pd.DataFrame, origen: int, h: int) -> pd.DataFrame:
    """Filas cuyo objetivo a horizonte `h` es conocido antes del origen.

    La fila `t` predice `t + h`, así que el objetivo es observable solo si
    `t + h <= origen - 1`.
    """
    return df[df["t"] <= origen - 1 - h]


def filas_a_predecir(df: pd.DataFrame, origen: int) -> pd.DataFrame:
    """Última semana conocida en el origen: una fila por serie."""
    return df[df["t"] == origen - 1]


def objetivo_log_cambio(df: pd.DataFrame, h: int) -> pd.Series:
    """Objetivo para el horizonte `h`: `log(y[t+h] / y[t])`.

    Se calcula dentro de cada serie, sobre la rejilla semanal completa, así que
    `h` siempre significa *h semanas calendario*, aunque haya huecos.
    """
    log_y = np.log(df["y"].where(df["y"] > 0))
    futuro = log_y.groupby(
        [df["producto_id"], df["plaza_id"]], observed=True
    ).shift(-h)
    return futuro - log_y
