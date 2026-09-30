"""Análisis de fallos: dónde y cuánto se degrada el modelo en episodios de choque.

El promedio global de error esconde lo que de verdad importa. Un modelo puede
tener un MASE decente y aun así ser inútil justo cuando más se necesita: durante
un paro, una helada o un pico de precio. Este módulo separa las semanas
"normales" de las "de choque" y compara el error en cada régimen.

La clasificación es **data-driven**: una semana es de choque si el movimiento
mediano del panel entero supera un cuantil alto de su propia distribución
histórica. `config/episodios.yaml` solo le pone nombre a lo que el dato ya
señaló, para poder contarlo en la app.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from precios.config import CONFIG_DIR

log = logging.getLogger(__name__)


def cargar_episodios(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Lee `config/episodios.yaml`.

    Returns:
        (episodios con fechas ya convertidas, parámetros de detección).
    """
    ruta = path or CONFIG_DIR / "episodios.yaml"
    with ruta.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    eps = pd.DataFrame(raw.get("episodios", []))
    if not eps.empty:
        eps["desde"] = pd.to_datetime(eps["desde"])
        eps["hasta"] = pd.to_datetime(eps["hasta"])
    return eps, raw.get("deteccion", {})


def marcar_semanas_de_choque(
    panel: pd.DataFrame, *, cuantil: float = 0.90, min_series: int = 5
) -> pd.DataFrame:
    """Clasifica cada semana como de choque o normal.

    El indicador es la **mediana** del valor absoluto del cambio porcentual
    semanal entre todas las series. Se usa la mediana y no el promedio para que
    una sola serie enloquecida no marque toda la semana: un choque de verdad
    mueve el panel entero.

    Returns:
        Un `DataFrame` por semana con el indicador y la marca `es_choque`.
    """
    p = panel.dropna(subset=["precio_kg"]).sort_values(["producto_id", "plaza_id", "semana"])
    log_precio = np.log(p["precio_kg"].where(p["precio_kg"] > 0))
    p = p.assign(
        cambio_abs=log_precio.groupby(
            [p["producto_id"], p["plaza_id"]], observed=True
        ).diff().abs()
    )

    por_semana = (
        p.groupby("semana")
        .agg(mov_mediano=("cambio_abs", "median"), n_series=("cambio_abs", "count"))
        .reset_index()
    )
    evaluable = por_semana["n_series"] >= min_series
    umbral = float(por_semana.loc[evaluable, "mov_mediano"].quantile(cuantil))

    por_semana["es_choque"] = evaluable & (por_semana["mov_mediano"] > umbral)
    por_semana["umbral"] = umbral
    # El indicador está en log; se expresa también en % para poder leerlo.
    por_semana["mov_mediano_pct"] = 100 * np.expm1(por_semana["mov_mediano"])

    n = int(por_semana["es_choque"].sum())
    log.info(
        "Semanas de choque: %d de %d (umbral %.2f%% de movimiento mediano)",
        n,
        len(por_semana),
        100 * np.expm1(umbral),
    )
    return por_semana


def etiquetar_con_episodios(semanas: pd.DataFrame, episodios: pd.DataFrame) -> pd.DataFrame:
    """Añade el nombre del episodio conocido al que pertenece cada semana."""
    semanas = semanas.copy()
    semanas["episodio"] = pd.NA
    for _, ep in episodios.iterrows():
        dentro = semanas["semana"].between(ep["desde"], ep["hasta"])
        semanas.loc[dentro, "episodio"] = ep["nombre"]
    return semanas


def degradacion_por_regimen(
    backtest: pd.DataFrame, semanas: pd.DataFrame
) -> pd.DataFrame:
    """Compara el error de cada modelo en semanas de choque frente a normales.

    La semana que se mira es la **objetivo** (la que se pronostica), no la del
    origen: lo que interesa es si el modelo falla cuando ocurre el choque.
    """
    marcas = semanas.set_index("semana")["es_choque"]
    bt = backtest.copy()
    bt["es_choque"] = bt["semana_objetivo"].map(marcas).fillna(False)
    bt["ae"] = (bt["y_true"] - bt["y_pred"]).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        bt["mase_punto"] = np.where(
            bt["escala_mase"] > 0, bt["ae"] / bt["escala_mase"], np.nan
        )

    filas = []
    for (modelo, h, choque), g in bt.groupby(["modelo", "h", "es_choque"], observed=True):
        fila = {
            "modelo": modelo,
            "h": h,
            "regimen": "choque" if choque else "normal",
            "n": len(g),
            "mase": float(np.nanmean(g["mase_punto"])),
            "mae": float(g["ae"].mean()),
        }
        if {"lo", "hi"}.issubset(g.columns) and g["lo"].notna().any():
            con_int = g.dropna(subset=["lo", "hi"])
            dentro = (con_int["y_true"] >= con_int["lo"]) & (con_int["y_true"] <= con_int["hi"])
            fila["cobertura_pct"] = float(100 * dentro.mean())
        filas.append(fila)

    out = pd.DataFrame(filas)
    # Se añade el factor de degradación, que es la lectura útil.
    pivote = out.pivot_table(index=["modelo", "h"], columns="regimen", values="mase")
    if {"choque", "normal"}.issubset(pivote.columns):
        pivote["factor_degradacion"] = pivote["choque"] / pivote["normal"]
        out = out.merge(
            pivote["factor_degradacion"].reset_index(), on=["modelo", "h"], how="left"
        )
    return out.sort_values(["modelo", "h", "regimen"]).reset_index(drop=True)
