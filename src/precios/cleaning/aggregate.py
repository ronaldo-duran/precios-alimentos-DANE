"""Agregación diaria -> semanal y marcado de calidad.

Decisiones documentadas:

* **Mediana, no promedio.** En el histórico hay errores de digitación de un solo
  día (p. ej. Tomate\\* en Popayán cotizado a 250 COP/kg frente a una mediana de
  ~2.400). La mediana los absorbe; el promedio los propaga a la semana entera.
* **La semana se etiqueta con su lunes.** Es la convención ISO y hace trivial
  razonar sobre horizontes: el origen del pronóstico siempre es un lunes.
* **Se marca, no se borra.** Los outliers quedan con un flag para poder
  estudiarlos (los choques reales son la parte interesante del problema).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from precios.config import ReglasLimpieza

log = logging.getLogger(__name__)

#: Factor que vuelve la MAD comparable con una desviación estándar gaussiana.
_ESCALA_MAD = 0.6745


def _lunes(fechas: pd.Series) -> pd.Series:
    """Devuelve el lunes de la semana de cada fecha."""
    fechas = pd.to_datetime(fechas)
    return (fechas - pd.to_timedelta(fechas.dt.weekday, unit="D")).dt.normalize()


def descartar_semana_parcial(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina la última semana si aún está en curso.

    SIPSA cotiza ~6 días por semana. Si la corrida ocurre a mitad de semana, esa
    semana solo tiene 1-2 días y su mediana queda sesgada. Se compara el número
    de días distintos de la última semana contra la mediana de las 8 anteriores.
    """
    if df.empty:
        return df
    dias_por_semana = df.groupby("semana")["fecha"].nunique().sort_index()
    if len(dias_por_semana) < 2:
        return df

    ultima = dias_por_semana.index[-1]
    referencia = dias_por_semana.iloc[-9:-1].median()
    if dias_por_semana.iloc[-1] < referencia:
        log.warning(
            "Descartada la semana en curso %s: %d días frente a %.0f habituales",
            ultima.date(),
            dias_por_semana.iloc[-1],
            referencia,
        )
        return df[df["semana"] != ultima].copy()
    return df


def agregar_semanal(df: pd.DataFrame, reglas: ReglasLimpieza) -> pd.DataFrame:
    """Colapsa las cotizaciones diarias a un panel semanal por producto y plaza.

    Args:
        df: datos diarios ya normalizados y recortados al alcance.
        reglas: umbrales de `config/cleaning.yaml`.

    Returns:
        Panel semanal con `precio_kg`, dispersión intrasemanal y flags de calidad.
    """
    df = df.copy()
    df["semana"] = _lunes(df["fecha"])

    if reglas.agregacion.get("descartar_semana_parcial", True):
        df = descartar_semana_parcial(df)

    estadistico = reglas.agregacion.get("estadistico", "mediana")
    if estadistico not in {"mediana", "promedio"}:
        raise ValueError(f"estadistico no soportado: {estadistico!r}")
    func = "median" if estadistico == "mediana" else "mean"

    llaves = ["producto_id", "plaza_id", "semana"]
    agg = (
        df.groupby(llaves, observed=True)
        .agg(
            precio_kg=("precio_kg", func),
            precio_kg_min=("precio_kg_min", "min"),
            precio_kg_max=("precio_kg_max", "max"),
            n_dias=("fecha", "nunique"),
            producto=("producto", "first"),
            plaza=("plaza", "first"),
            ciudad=("ciudad", "first"),
            grupo=("grupo", "first"),
        )
        .reset_index()
        .sort_values(llaves)
        .reset_index(drop=True)
    )

    min_dias = int(reglas.agregacion.get("min_dias_por_semana", 1))
    antes = len(agg)
    agg = agg[agg["n_dias"] >= min_dias].reset_index(drop=True)
    if len(agg) < antes:
        log.info("Descartadas %d semanas con menos de %d día(s)", antes - len(agg), min_dias)

    agg["flag_pocos_dias"] = agg["n_dias"] <= 1
    agg = marcar_outliers(agg, reglas)

    log.info(
        "Panel semanal: %d filas, %d series, %s -> %s",
        len(agg),
        agg.groupby(["producto_id", "plaza_id"], observed=True).ngroups,
        agg["semana"].min().date(),
        agg["semana"].max().date(),
    )
    return agg


def marcar_outliers(df: pd.DataFrame, reglas: ReglasLimpieza) -> pd.DataFrame:
    """Marca saltos semanales anómalos con un z-score robusto sobre log-retornos.

    Se usa MAD y no desviación estándar porque la desviación estándar la inflan
    los propios outliers que se quieren detectar. Solo marca: nunca borra ni
    corrige, porque un salto grande puede ser un error de digitación **o** un
    choque real (un paro, una helada), y distinguirlos es parte del análisis.

    El z-score por sí solo no basta: en una serie casi plana la MAD tiende a
    cero y cualquier variación trivial se vuelve un z enorme. Por eso un
    movimiento debe superar además `salto_relativo_min` para marcarse.
    """
    df = df.copy()
    umbral_z = float(reglas.outliers.get("umbral_z", 5.0))
    salto_min = float(reglas.outliers.get("salto_relativo_min", 0.15))
    salto_max = float(reglas.outliers.get("salto_relativo_max", 1.0))

    df["log_retorno"] = np.nan
    df["z_robusto"] = np.nan

    for _, idx in df.groupby(["producto_id", "plaza_id"], observed=True).groups.items():
        g = df.loc[idx].sort_values("semana")
        precios = g["precio_kg"].to_numpy(dtype=float)
        if len(precios) < 3:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(precios), prepend=np.nan)
        centro = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - centro))
        z = np.full_like(r, np.nan) if mad == 0 else _ESCALA_MAD * (r - centro) / mad
        df.loc[g.index, "log_retorno"] = r
        df.loc[g.index, "z_robusto"] = z

    salto_relativo = np.expm1(np.abs(df["log_retorno"]))
    anomalo_estadisticamente = (df["z_robusto"].abs() > umbral_z) & (salto_relativo >= salto_min)
    salto_extremo = salto_relativo > salto_max
    df["flag_outlier"] = (anomalo_estadisticamente | salto_extremo).fillna(False)

    n = int(df["flag_outlier"].sum())
    log.info("Marcados %d outliers de %d filas (%.2f%%)", n, len(df), 100 * n / max(len(df), 1))
    return df


def reindexar_semanas(df: pd.DataFrame) -> pd.DataFrame:
    """Inserta filas vacías en las semanas faltantes de cada serie.

    No imputa: deja `precio_kg` como NaN y marca `flag_faltante`. La decisión de
    interpolar o no pertenece a la construcción de features, donde se sabe cuál
    es el origen del pronóstico y por tanto qué información es legítimo usar.
    """
    partes = []
    for (prod, plaza), g in df.groupby(["producto_id", "plaza_id"], observed=True):
        g = g.sort_values("semana")
        rango = pd.date_range(g["semana"].min(), g["semana"].max(), freq="W-MON")
        reind = g.set_index("semana").reindex(rango)
        reind.index.name = "semana"
        reind["producto_id"] = prod
        reind["plaza_id"] = plaza
        for col in ("producto", "plaza", "ciudad", "grupo"):
            if col in reind:
                reind[col] = g[col].iloc[0]
        reind["flag_faltante"] = reind["precio_kg"].isna()
        partes.append(reind.reset_index())

    out = pd.concat(partes, ignore_index=True).sort_values(
        ["producto_id", "plaza_id", "semana"]
    )
    for col in ("flag_outlier", "flag_pocos_dias", "flag_faltante"):
        if col in out:
            # Tras reindexar la columna es `object` con NaN; `boolean` (nullable)
            # permite rellenar sin el downcast silencioso que pandas deprecó.
            out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    n_falt = int(out["flag_faltante"].sum())
    log.info("Reindexado: %d filas, %d semanas faltantes marcadas", len(out), n_falt)
    return out.reset_index(drop=True)
