"""Etapa `ablacion`: mide cuánto aporta cada bloque de variables exógenas.

Entrada: `data/processed/semanal.parquet`
Salida:  `data/processed/ablacion.csv` y `ablacion_por_producto.csv`

El método es el único que permite una respuesta honesta: se corre **el mismo
walk-forward, sobre los mismos orígenes, con los mismos hiperparámetros y la
misma semilla**, cambiando únicamente el bloque de features. Cualquier
diferencia que aparezca es atribuible al bloque.

Se reportan dos cifras por variante: sobre todos los orígenes y sobre la
**mitad reservada**, la que no se usó para elegir hiperparámetros. Si las dos
no coinciden en el signo, la mejora no es de fiar.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from precios.config import PROCESSED_DIR, load_evaluacion, load_lgbm_params
from precios.evaluation.backtest import backtest_global
from precios.evaluation.splits import rolling_origins
from precios.features.build import anadir_objetivos, columnas_features, construir_features
from precios.logging_setup import setup_logging
from precios.models.busqueda import partir_origenes
from precios.models.lgbm import LGBMGlobal
from precios.pipeline.evaluate import cargar_panel_apto, fijar_semillas

log = logging.getLogger(__name__)

ABLACION_PATH = PROCESSED_DIR / "ablacion.csv"
ABLACION_PRODUCTO_PATH = PROCESSED_DIR / "ablacion_por_producto.csv"

#: Variantes a comparar. La primera es la referencia.
VARIANTES: dict[str, tuple[str, ...]] = {
    "base": (),
    "calendario": ("calendario",),
    "enso": ("enso",),
    "calendario+enso": ("calendario", "enso"),
}


def ablacion() -> pd.DataFrame:
    """Corre todas las variantes y devuelve la tabla comparativa."""
    cfg = load_evaluacion()
    params = load_lgbm_params()
    panel, _ = cargar_panel_apto(solo_aptas=cfg.solo_series_aptas)

    n_semanas = int(panel.groupby(["producto_id", "plaza_id"], observed=True).size().max())
    folds = rolling_origins(
        n_semanas,
        min_train=cfg.min_train,
        horizonte_max=cfg.horizonte_max,
        paso=cfg.paso,
        max_origenes=cfg.max_origenes,
    )
    _, reservados = partir_origenes(folds)
    corte_reservado = reservados[0].origen
    log.info(
        "Ablación: %d variantes x %d orígenes (reservados desde la obs %d)",
        len(VARIANTES),
        len(folds),
        corte_reservado,
    )

    resultados = []
    for nombre, bloques in VARIANTES.items():
        fijar_semillas(cfg.semilla)
        log.info("--- variante %r (exógenas: %s) ---", nombre, bloques or "ninguna")
        features = anadir_objetivos(
            construir_features(panel, exogenas=bloques, horizontes=cfg.horizontes),
            cfg.horizontes,
        )
        bt = backtest_global(
            features,
            {"lgbm_cuantil": lambda: LGBMGlobal(
                horizontes=cfg.horizontes, params=params, semilla=cfg.semilla
            )},
            folds,
        )
        bt["variante"] = nombre
        bt["n_features"] = len(columnas_features(features))
        resultados.append(bt)

    crudo = pd.concat(resultados, ignore_index=True)
    tabla = _resumir(crudo, corte_reservado)
    por_producto = _por_producto(crudo, corte_reservado)

    tabla.to_csv(ABLACION_PATH, index=False, encoding="utf-8")
    por_producto.to_csv(ABLACION_PRODUCTO_PATH, index=False, encoding="utf-8")
    _reportar(tabla)
    return tabla


def _mase(g: pd.DataFrame) -> float:
    err = (g["y_true"] - g["y_pred"]).abs().to_numpy(dtype=float)
    escala = g["escala_mase"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        punto = np.where(escala > 0, err / escala, np.nan)
    return float(np.nanmean(punto))


def _resumir(crudo: pd.DataFrame, corte: int) -> pd.DataFrame:
    """MASE por variante y horizonte, en todos los orígenes y en los reservados."""
    filas = []
    for (variante, h), g in crudo.groupby(["variante", "h"], observed=True):
        reservado = g[g["origen"] >= corte]
        filas.append(
            {
                "variante": variante,
                "h": int(h),
                "n_features": int(g["n_features"].iloc[0]),
                "n": len(g),
                "mase_todos": _mase(g),
                "mase_reservados": _mase(reservado) if not reservado.empty else np.nan,
            }
        )
    tabla = pd.DataFrame(filas)

    base = tabla[tabla["variante"] == "base"].set_index("h")
    for col in ("mase_todos", "mase_reservados"):
        referencia = base[col]
        tabla[f"delta_{col.split('_')[1]}_pct"] = 100 * (
            1 - tabla[col] / tabla["h"].map(referencia)
        )
    return tabla.sort_values(["variante", "h"]).reset_index(drop=True)


def _por_producto(crudo: pd.DataFrame, corte: int) -> pd.DataFrame:
    """Igual que `_resumir` pero por producto: el promedio esconde el reparto."""
    filas = []
    for (variante, prod, h), g in crudo.groupby(
        ["variante", "producto_id", "h"], observed=True
    ):
        filas.append(
            {
                "variante": variante,
                "producto_id": prod,
                "h": int(h),
                "n": len(g),
                "mase": _mase(g),
                "mase_reservados": _mase(g[g["origen"] >= corte]),
            }
        )
    tabla = pd.DataFrame(filas)
    base = tabla[tabla["variante"] == "base"].set_index(["producto_id", "h"])["mase"]
    tabla["delta_pct"] = tabla.apply(
        lambda r: 100 * (1 - r["mase"] / base.loc[(r["producto_id"], r["h"])]), axis=1
    )
    return tabla.sort_values(["variante", "producto_id", "h"]).reset_index(drop=True)


def _reportar(tabla: pd.DataFrame) -> None:
    """Deja el veredicto en el log, sin adornos."""
    log.info("Ablación (delta %% de MASE frente a 'base'; positivo = mejora):")
    for variante, g in tabla.groupby("variante", observed=True):
        if variante == "base":
            continue
        todos = " ".join(f"h{int(r.h)}={r.delta_todos_pct:+.2f}%" for r in g.itertuples())
        res = " ".join(f"h{int(r.h)}={r.delta_reservados_pct:+.2f}%" for r in g.itertuples())
        log.info("  %-16s todos:      %s", variante, todos)
        log.info("  %-16s reservados: %s", "", res)

        coherente = (
            (g["delta_todos_pct"] > 0) == (g["delta_reservados_pct"] > 0)
        ).all()
        if not coherente:
            log.warning(
                "  %s: el signo cambia entre todos y reservados -> la mejora NO es de fiar",
                variante,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablación de variables exógenas")
    parser.parse_args()
    setup_logging()
    ablacion()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
