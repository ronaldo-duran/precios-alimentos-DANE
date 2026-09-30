"""Etapa `reconcile`: compara los pronósticos registrados con lo que ocurrió.

Entrada: `forecasts_log.csv`, `alertas_log.csv` y `data/processed/semanal.parquet`
Salida:  `reconciliacion.csv`, `desempeno_en_vivo.csv`, `alertas_reconciliadas.csv`
         y `desempeno_alertas.csv`

Los logs de entrada son **append-only y no se tocan aquí**. Las tablas de salida
son derivadas: se recalculan enteras en cada corrida, lo que las hace idempotentes
y hace imposible que un error de reconciliación corrompa el historial original.

Esta es la parte del proyecto que no se puede falsear. Un backtest se puede
repetir hasta que salga bien; un pronóstico escrito antes de que ocurriera la
semana, no.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from precios.config import PROCESSED_DIR
from precios.evaluation.backtest import preparar_train
from precios.evaluation.metrics import escala_mase
from precios.logging_setup import setup_logging
from precios.pipeline.evaluate import cargar_panel_apto
from precios.pipeline.forecast import ALERTAS_LOG, BACKFILL_LOG, FORECASTS_LOG

log = logging.getLogger(__name__)

RECONCILIACION_PATH = PROCESSED_DIR / "reconciliacion.csv"
DESEMPENO_PATH = PROCESSED_DIR / "desempeno_en_vivo.csv"
ALERTAS_RECON_PATH = PROCESSED_DIR / "alertas_reconciliadas.csv"
DESEMPENO_ALERTAS_PATH = PROCESSED_DIR / "desempeno_alertas.csv"


def reconcile() -> pd.DataFrame:
    """Cruza los logs con los precios reales y acumula el desempeño en vivo."""
    if not FORECASTS_LOG.exists():
        raise FileNotFoundError(
            f"No existe {FORECASTS_LOG.name}. Ejecuta `make forecast` primero."
        )
    panel, _ = cargar_panel_apto()
    log_pronosticos = _leer_logs()

    recon = _reconciliar_pronosticos(log_pronosticos, panel)
    recon.to_csv(RECONCILIACION_PATH, index=False, encoding="utf-8")

    desempeno = _resumir_desempeno(recon)
    desempeno.to_csv(DESEMPENO_PATH, index=False, encoding="utf-8")

    _reconciliar_alertas(panel)
    _reportar(recon, desempeno)
    return recon


def _leer_logs() -> pd.DataFrame:
    """Une el log en vivo con el sembrado, conservando la marca `procedencia`.

    Viven en archivos distintos por higiene de git, pero se reconcilian juntos;
    la separación de resultados la garantiza la columna, no el archivo.
    """
    partes = [_leer_log(FORECASTS_LOG)]
    if BACKFILL_LOG.exists():
        partes.append(_leer_log(BACKFILL_LOG))
    return pd.concat(partes, ignore_index=True)


def _leer_log(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    for col in ("semana_origen", "semana_objetivo", "semana_limite"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _reconciliar_pronosticos(log_pron: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Añade el precio real a cada pronóstico cuya semana objetivo ya se publicó."""
    reales = (
        panel.dropna(subset=["precio_kg"])
        .set_index(["producto_id", "plaza_id", "semana"])["precio_kg"]
    )
    escalas = _escalas_por_origen(panel, log_pron)

    recon = log_pron.copy()
    claves = list(
        zip(
            recon["producto_id"],
            recon["plaza_id"],
            recon["semana_objetivo"],
            strict=True,
        )
    )
    recon["y_true"] = [float(reales.get(k, np.nan)) for k in claves]
    recon["escala_mase"] = [
        escalas.get(clave, np.nan)
        for clave in zip(
            recon["producto_id"], recon["plaza_id"], recon["semana_origen"], strict=True
        )
    ]
    recon["estado"] = np.where(recon["y_true"].notna(), "resuelto", "pendiente")

    resuelto = recon["y_true"].notna()
    recon["error"] = np.where(resuelto, recon["y_true"] - recon["y_pred"], np.nan)
    recon["error_abs"] = np.abs(recon["error"])
    with np.errstate(divide="ignore", invalid="ignore"):
        recon["mase_punto"] = np.where(
            recon["escala_mase"] > 0, recon["error_abs"] / recon["escala_mase"], np.nan
        )
    tiene_intervalo = recon["lo"].notna() & recon["hi"].notna()
    recon["dentro_intervalo"] = np.where(
        resuelto & tiene_intervalo,
        (recon["y_true"] >= recon["lo"]) & (recon["y_true"] <= recon["hi"]),
        np.nan,
    )
    return recon


def _escalas_por_origen(
    panel: pd.DataFrame, log_pron: pd.DataFrame
) -> dict[tuple[str, str, pd.Timestamp], float]:
    """Escala del MASE por serie y semana de origen.

    Mismo criterio que el backtest (MAE del naive dentro de la historia previa
    al origen), para que el MASE en vivo sea comparable con el del backtest.
    """
    origenes = sorted(log_pron["semana_origen"].unique())
    salida: dict[tuple[str, str, pd.Timestamp], float] = {}
    for (prod, plaza), g in panel.groupby(["producto_id", "plaza_id"], observed=True):
        g = g.sort_values("semana")
        semanas = g["semana"].to_numpy()
        precios = g["precio_kg"].to_numpy(dtype=float)
        for origen in origenes:
            hasta = int(np.searchsorted(semanas, np.datetime64(origen), side="right"))
            if hasta < 3:
                continue
            salida[(prod, plaza, pd.Timestamp(origen))] = escala_mase(
                preparar_train(precios[:hasta]), periodo=1
            )
    return salida


def _resumir_desempeno(recon: pd.DataFrame) -> pd.DataFrame:
    """Error acumulado por procedencia, modelo y horizonte, con la cobertura real.

    `vivo` y `backfill` se reportan SIEMPRE por separado. Los dos son
    out-of-sample, pero solo los `vivo` se escribieron antes de que ocurriera
    la semana, y esa diferencia es justamente lo que da valor al registro.
    """
    resuelto = recon[recon["estado"] == "resuelto"]
    if resuelto.empty:
        log.warning(
            "Todavía no hay ningún pronóstico resuelto: el log es más nuevo que los datos"
        )
        return pd.DataFrame(
            columns=[
                "procedencia", "modelo", "h", "n", "mae", "mase", "cobertura_pct",
                "primera", "ultima",
            ]
        )

    filas = []
    for (proc, modelo, h), g in resuelto.groupby(
        ["procedencia", "modelo", "h"], observed=True
    ):
        fila = {
            "procedencia": proc,
            "modelo": modelo,
            "h": int(h),
            "n": len(g),
            "mae": float(g["error_abs"].mean()),
            "mase": float(np.nanmean(g["mase_punto"])),
            "sesgo": float(g["error"].mean()),
            "primera": str(g["semana_objetivo"].min().date()),
            "ultima": str(g["semana_objetivo"].max().date()),
        }
        con_int = g["dentro_intervalo"].dropna()
        fila["cobertura_pct"] = float(100 * con_int.mean()) if len(con_int) else np.nan
        fila["n_con_intervalo"] = int(len(con_int))
        filas.append(fila)
    return (
        pd.DataFrame(filas)
        .sort_values(["procedencia", "modelo", "h"])
        .reset_index(drop=True)
    )


def _reconciliar_alertas(panel: pd.DataFrame) -> None:
    """Comprueba si cada alerta se cumplió antes de su semana límite.

    Una alerta acierta si el precio superó `precio_umbral` en **alguna** semana
    entre el origen y la semana límite. Se evalúa la alerta completa, no solo
    su último punto: avisar de un alza que ocurre en la semana 1 y se revierte
    en la 2 sigue siendo un aviso correcto.
    """
    if not ALERTAS_LOG.exists():
        log.info("Sin alertas_log.csv: no hay alertas que reconciliar")
        return

    alertas = _leer_log(ALERTAS_LOG)
    precios = panel.dropna(subset=["precio_kg"])

    filas = []
    for _, a in alertas.iterrows():
        ventana = precios[
            (precios["producto_id"] == a["producto_id"])
            & (precios["plaza_id"] == a["plaza_id"])
            & (precios["semana"] > a["semana_origen"])
            & (precios["semana"] <= a["semana_limite"])
        ]
        n_esperadas = int(a["horizonte_semanas"])
        if len(ventana) < n_esperadas:
            estado, ocurrio, maximo = "pendiente", np.nan, np.nan
        else:
            maximo = float(ventana["precio_kg"].max())
            ocurrio = bool(maximo >= a["precio_umbral"])
            estado = "resuelto"
        filas.append(
            {
                **a.to_dict(),
                "estado": estado,
                "precio_max_observado": maximo,
                "ocurrio_alza": ocurrio,
            }
        )

    out = pd.DataFrame(filas)
    out.to_csv(ALERTAS_RECON_PATH, index=False, encoding="utf-8")

    resuelto = out[out["estado"] == "resuelto"]
    if resuelto.empty:
        log.info("Ninguna alerta resuelta todavía (necesitan %s semanas)", "2")
        pd.DataFrame(columns=["grupo", "n", "tasa_ocurrencia_pct"]).to_csv(
            DESEMPENO_ALERTAS_PATH, index=False, encoding="utf-8"
        )
        return

    resumen = []
    for activa, g in resuelto.groupby("alerta_activa"):
        resumen.append(
            {
                "grupo": "alerta activa" if activa else "sin alerta",
                "n": len(g),
                "tasa_ocurrencia_pct": round(float(100 * g["ocurrio_alza"].mean()), 1),
                "probabilidad_media_pct": round(float(100 * g["probabilidad"].mean()), 1),
            }
        )
    tabla = pd.DataFrame(resumen)
    tabla.to_csv(DESEMPENO_ALERTAS_PATH, index=False, encoding="utf-8")
    log.info("Desempeño de alertas:\n%s", tabla.to_string(index=False))


def _reportar(recon: pd.DataFrame, desempeno: pd.DataFrame) -> None:
    """Deja en el log el estado del registro en vivo."""
    n_total = len(recon)
    n_resuelto = int((recon["estado"] == "resuelto").sum())
    log.info(
        "Reconciliación: %d pronósticos en el log, %d resueltos, %d pendientes",
        n_total,
        n_resuelto,
        n_total - n_resuelto,
    )
    if desempeno.empty:
        return

    for proc in desempeno["procedencia"].unique():
        etiqueta = (
            "REAL (escrito antes de los hechos)"
            if proc == "vivo"
            else "backfill del walk-forward (valor descriptivo, no probatorio)"
        )
        log.info("Desempeño %s:", etiqueta)
        _reportar_grupo(desempeno[desempeno["procedencia"] == proc])


def _reportar_grupo(desempeno: pd.DataFrame) -> None:
    for _, f in desempeno.iterrows():
        extra = (
            f" | cobertura {f['cobertura_pct']:.1f}% (n={int(f['n_con_intervalo'])})"
            if np.isfinite(f.get("cobertura_pct", np.nan))
            else ""
        )
        log.info(
            "  %-16s h%d  n=%4d  MASE=%.3f  MAE=%.0f%s",
            f["modelo"],
            f["h"],
            f["n"],
            f["mase"],
            f["mae"],
            extra,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcilia el log de pronósticos en vivo")
    parser.parse_args()
    setup_logging()
    reconcile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
