"""Métricas de pronóstico.

La métrica principal es **MASE**, porque está normalizada contra el naive: un
MASE < 1 significa que el modelo le gana al naive, y > 1 que no. MAE y RMSE van
en COP/kg y no se pueden comparar entre productos (un aguacate y una papa viven
en escalas distintas). sMAPE se reporta con cautela.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def escala_mase(y_train: np.ndarray, periodo: int = 1) -> float:
    """Escala del MASE: MAE del naive **dentro del periodo de entrenamiento**.

    Se calcula sobre el train y no sobre el test, que es lo que hace comparable
    el MASE entre series y evita que el denominador dependa de lo que se predice.

    Args:
        y_train: historia disponible en el origen del pronóstico.
        periodo: 1 para el naive simple, 52 para el naive estacional semanal.

    Returns:
        La escala, o NaN si no hay suficientes datos o la serie es constante.
    """
    y = np.asarray(y_train, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) <= periodo:
        return float("nan")
    diferencias = np.abs(y[periodo:] - y[:-periodo])
    escala = float(np.mean(diferencias))
    # Una serie constante da escala 0 y volvería infinito cualquier MASE.
    return escala if escala > 0 else float("nan")


def mase(y_true: np.ndarray, y_pred: np.ndarray, escala: float) -> float:
    """Error absoluto medio escalado. < 1 = mejor que el naive."""
    if not np.isfinite(escala) or escala <= 0:
        return float("nan")
    return mae(y_true, y_pred) / escala


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE simétrico en porcentaje.

    Indefinido cuando `|y_true| + |y_pred|` es cero; esos puntos se omiten en
    vez de contarse como error 0, que sería engañoso. Con precios mayoristas
    (siempre > 0) el caso no debería darse, pero la guarda es barata.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominador = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valido = denominador > 0
    if not valido.any():
        return float("nan")
    return float(
        100.0 * np.mean(np.abs(y_true[valido] - y_pred[valido]) / denominador[valido])
    )


def cobertura(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Proporción de valores reales dentro del intervalo, en porcentaje.

    Es la métrica honesta de la incertidumbre: un intervalo del 80% que solo
    cubre el 55% de las veces está mintiendo sobre lo que sabe.
    """
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    valido = ~(np.isnan(y_true) | np.isnan(lo) | np.isnan(hi))
    if not valido.any():
        return float("nan")
    dentro = (y_true[valido] >= lo[valido]) & (y_true[valido] <= hi[valido])
    return float(100.0 * np.mean(dentro))


def amplitud_relativa(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Ancho medio del intervalo como % del valor real.

    Acompaña a la cobertura: un intervalo puede cubrir el 100% simplemente
    siendo absurdamente ancho, y eso no es informativo.
    """
    y_true = np.asarray(y_true, dtype=float)
    ancho = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    valido = ~(np.isnan(y_true) | np.isnan(ancho)) & (y_true > 0)
    if not valido.any():
        return float("nan")
    return float(100.0 * np.mean(ancho[valido] / y_true[valido]))


#: Columnas que debe tener el DataFrame largo de resultados.
COLUMNAS_RESULTADOS: tuple[str, ...] = (
    "producto_id",
    "plaza_id",
    "modelo",
    "origen",
    "semana_origen",
    "h",
    "semana_objetivo",
    "y_true",
    "y_pred",
    "escala_mase",
)


def resumir(
    resultados: pd.DataFrame,
    por: list[str] | None = None,
) -> pd.DataFrame:
    """Agrega el DataFrame largo de pronósticos a una tabla de métricas.

    Args:
        resultados: una fila por (serie, modelo, origen, horizonte).
        por: columnas de agrupación. Por defecto `["modelo", "h"]`.

    Returns:
        Tabla con n, MAE, RMSE, MASE, sMAPE y, si hay intervalos, cobertura.
    """
    por = por or ["modelo", "h"]
    faltantes = [c for c in ("y_true", "y_pred") if c not in resultados.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en resultados: {faltantes}")

    filas = []
    for llaves, g in resultados.groupby(por, observed=True, dropna=False):
        g = g.dropna(subset=["y_true", "y_pred"])
        if g.empty:
            continue
        llaves = llaves if isinstance(llaves, tuple) else (llaves,)
        fila = dict(zip(por, llaves, strict=True))
        fila["n"] = len(g)
        fila["mae"] = mae(g["y_true"], g["y_pred"])
        fila["rmse"] = rmse(g["y_true"], g["y_pred"])
        fila["smape"] = smape(g["y_true"], g["y_pred"])

        # MASE se promedia por punto usando la escala de su propio origen: así
        # una serie con más orígenes no distorsiona la escala de las demás.
        if "escala_mase" in g:
            escalas = g["escala_mase"].to_numpy(dtype=float)
            errores = np.abs(g["y_true"].to_numpy(float) - g["y_pred"].to_numpy(float))
            with np.errstate(divide="ignore", invalid="ignore"):
                por_punto = np.where(escalas > 0, errores / escalas, np.nan)
            fila["mase"] = float(np.nanmean(por_punto)) if np.isfinite(por_punto).any() else np.nan

        if {"lo", "hi"}.issubset(g.columns):
            fila["cobertura_pct"] = cobertura(g["y_true"], g["lo"], g["hi"])
            fila["amplitud_pct"] = amplitud_relativa(g["y_true"], g["lo"], g["hi"])
        filas.append(fila)

    return pd.DataFrame(filas).sort_values(por).reset_index(drop=True)
