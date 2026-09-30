"""Etapa `forecast`: pronostica las próximas 4 semanas y las registra en vivo.

Entrada: `models/latest.json` + `data/processed/semanal.parquet`
Salida:  `data/processed/forecasts_log.csv` y `alertas_log.csv` (APPEND-ONLY)

El registro en vivo es lo que separa un backtest de una evaluación honesta. Un
backtest siempre se puede reconstruir hasta que dé bien; un pronóstico escrito
**antes** de que ocurra la semana objetivo, no.

Reglas del log:

* Se escribe **antes** de que exista el dato real.
* **Nunca se reescribe una fila pasada.** Si se vuelve a correr con el mismo
  origen y la misma versión de modelo, no se añade nada (es idempotente); si
  cambia el modelo o avanza el origen, se añade una fila nueva y la anterior
  queda como está.
* Se guarda también el pronóstico del **naive** para el mismo punto, porque sin
  la referencia el error acumulado no dice nada.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from precios.config import PROCESSED_DIR, load_evaluacion
from precios.features.build import anadir_objetivos, construir_features, filas_a_predecir
from precios.logging_setup import setup_logging
from precios.models import registry
from precios.models.alertas import cargar_config, generar_alertas
from precios.pipeline.evaluate import cargar_panel_apto, fijar_semillas

log = logging.getLogger(__name__)

FORECASTS_LOG = PROCESSED_DIR / "forecasts_log.csv"
ALERTAS_LOG = PROCESSED_DIR / "alertas_log.csv"

#: El sembrado del backtest va a su propio archivo, no al log en vivo. Son
#: ~14.600 filas estáticas que se escriben una vez; mezclarlas con el log haría
#: que cada commit semanal de CI reescribiera un blob de 2 MB en git, cuando el
#: crecimiento real son 35 KB por semana.
BACKFILL_LOG = PROCESSED_DIR / "forecasts_backfill.csv"

#: Columnas del log de pronósticos, en orden fijo.
COLUMNAS_LOG: tuple[str, ...] = (
    "fecha_pronostico",
    "version_modelo",
    "modelo",
    "producto_id",
    "plaza_id",
    "semana_origen",
    "h",
    "semana_objetivo",
    "precio_origen",
    "y_pred",
    "lo",
    "hi",
    "nivel_intervalo",
    "procedencia",
)

#: Distingue un pronóstico escrito ANTES de que ocurriera la semana ("vivo")
#: de uno sembrado a posteriori desde el backtest ("backfill"). Son
#: out-of-sample igual, pero no tienen el mismo valor probatorio y NUNCA se
#: mezclan en las tablas de desempeño.
PROCEDENCIAS: tuple[str, str] = ("vivo", "backfill")

COLUMNAS_ALERTAS: tuple[str, ...] = (
    "fecha_pronostico",
    "version_modelo",
    "producto_id",
    "plaza_id",
    "semana_origen",
    "horizonte_semanas",
    "precio_actual",
    "umbral_alza",
    "precio_umbral",
    "probabilidad",
    "alerta_activa",
    "semana_limite",
)


def forecast(*, version: str | None = None) -> pd.DataFrame:
    """Genera los pronósticos vigentes y los añade al log si son nuevos."""
    cfg = load_evaluacion()
    fijar_semillas(cfg.semilla)

    modelo, metadata = registry.cargar(version)
    panel, _ = cargar_panel_apto(solo_aptas=cfg.solo_series_aptas)
    features = anadir_objetivos(construir_features(panel), cfg.horizontes)

    origen = int(features["t"].max()) + 1
    base = filas_a_predecir(features, origen)
    semana_origen = pd.Timestamp(base["semana"].iloc[0])
    log.info(
        "Pronosticando desde la semana %s con el modelo %s",
        semana_origen.date(),
        metadata.version,
    )

    pred = modelo.predict(base)
    pred = _anadir_contexto(pred, base, semana_origen, metadata)
    naive = _pronostico_naive(base, semana_origen, metadata, cfg.horizontes)
    nuevas = pd.concat([pred, naive], ignore_index=True)[list(COLUMNAS_LOG)]

    llaves = (
        "procedencia",
        "version_modelo",
        "modelo",
        "producto_id",
        "plaza_id",
        "h",
        "semana_objetivo",
    )
    escritas = _anexar(nuevas, FORECASTS_LOG, llaves)
    log.info("Pronósticos añadidos al log: %d (de %d candidatos)", escritas, len(nuevas))

    _registrar_alertas(pred, base, modelo, metadata, semana_origen)
    return nuevas


def _anadir_contexto(
    pred: pd.DataFrame, base: pd.DataFrame, semana_origen: pd.Timestamp, metadata
) -> pd.DataFrame:
    """Completa el pronóstico con las columnas del log."""
    precios = base.set_index(["producto_id", "plaza_id"])["y"]
    pred = pred.copy()
    pred["fecha_pronostico"] = dt.date.today().isoformat()
    pred["version_modelo"] = metadata.version
    pred["modelo"] = "lgbm_conformal"
    pred["semana_origen"] = semana_origen
    pred["semana_objetivo"] = semana_origen + pd.to_timedelta(7 * pred["h"], unit="D")
    pred["precio_origen"] = [
        float(precios.get(par, np.nan))
        for par in zip(pred["producto_id"], pred["plaza_id"], strict=True)
    ]
    pred["nivel_intervalo"] = metadata.nivel_intervalo
    pred["procedencia"] = "vivo"
    return pred


def _pronostico_naive(
    base: pd.DataFrame, semana_origen: pd.Timestamp, metadata, horizontes
) -> pd.DataFrame:
    """El naive para los mismos puntos: sin referencia no hay evaluación."""
    filas = []
    for _, fila in base.iterrows():
        for h in horizontes:
            filas.append(
                {
                    "fecha_pronostico": dt.date.today().isoformat(),
                    "version_modelo": metadata.version,
                    "modelo": "naive",
                    "producto_id": fila["producto_id"],
                    "plaza_id": fila["plaza_id"],
                    "semana_origen": semana_origen,
                    "h": h,
                    "semana_objetivo": semana_origen + pd.Timedelta(7 * h, unit="D"),
                    "precio_origen": float(fila["y"]),
                    "y_pred": float(fila["y"]),
                    "lo": np.nan,
                    "hi": np.nan,
                    "nivel_intervalo": np.nan,
                    "procedencia": "vivo",
                }
            )
    return pd.DataFrame(filas)


def _anexar(nuevas: pd.DataFrame, ruta: Path, llaves: tuple[str, ...]) -> int:
    """Añade solo las filas que no estén ya en el log. Nunca reescribe.

    Returns:
        Número de filas efectivamente añadidas.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevas = nuevas.copy()
    for col in ("semana_origen", "semana_objetivo", "semana_limite"):
        if col in nuevas.columns:
            nuevas[col] = pd.to_datetime(nuevas[col]).dt.date.astype(str)

    if ruta.exists():
        previo = pd.read_csv(ruta, dtype=str)
        faltan = [c for c in llaves if c not in previo.columns]
        if faltan:
            raise ValueError(
                f"{ruta.name} tiene un esquema antiguo: le faltan {faltan}. "
                "El log es append-only, así que no se migra en silencio: "
                "archiva el archivo a mano y vuelve a ejecutar `make forecast`."
            )
        existentes = set(map(tuple, previo[list(llaves)].astype(str).to_numpy()))
        mascara = [
            tuple(str(v) for v in fila) not in existentes
            for fila in nuevas[list(llaves)].to_numpy()
        ]
        nuevas = nuevas[mascara]
        if nuevas.empty:
            log.info("Nada nuevo que registrar en %s (idempotente)", ruta.name)
            return 0
        nuevas.to_csv(ruta, mode="a", header=False, index=False, encoding="utf-8")
    else:
        nuevas.to_csv(ruta, index=False, encoding="utf-8")
    return len(nuevas)


def _registrar_alertas(
    pred: pd.DataFrame, base: pd.DataFrame, modelo, metadata, semana_origen: pd.Timestamp
) -> None:
    """Calcula y registra las alertas de alza asociadas a este origen."""
    cfg = cargar_config()
    precios = base.set_index(["producto_id", "plaza_id"])["y"]
    con_base = pred.copy()
    con_base["y_base"] = [
        float(precios.get(par, np.nan))
        for par in zip(con_base["producto_id"], con_base["plaza_id"], strict=True)
    ]

    alertas = generar_alertas(con_base, modelo, cfg)
    if alertas.empty:
        return
    alertas["fecha_pronostico"] = dt.date.today().isoformat()
    alertas["version_modelo"] = metadata.version
    alertas["semana_origen"] = semana_origen
    alertas["semana_limite"] = semana_origen + pd.Timedelta(
        7 * int(cfg.get("horizonte_semanas", 2)), unit="D"
    )

    escritas = _anexar(
        alertas[list(COLUMNAS_ALERTAS)],
        ALERTAS_LOG,
        ("version_modelo", "producto_id", "plaza_id", "semana_origen"),
    )
    log.info("Alertas añadidas al log: %d", escritas)


def sembrar_desde_backtest(backtest_path: Path | None = None) -> int:
    """Siembra el log con los pronósticos del backtest, marcados como `backfill`.

    El log en vivo empieza vacío por definición: su primera fila no se puede
    resolver hasta que pase la semana. Para que la reconciliación y la app
    tengan algo que mostrar desde el día uno, se siembran los pronósticos del
    walk-forward, que son igual de out-of-sample (ningún modelo vio datos
    posteriores a su origen) pero **no se escribieron antes de los hechos**.

    Por eso van marcados `procedencia="backfill"` y las tablas de desempeño los
    reportan por separado: tienen valor descriptivo, no probatorio.

    Returns:
        Número de filas añadidas.
    """
    from precios.pipeline.evaluate import BACKTEST_PATH

    ruta = backtest_path or BACKTEST_PATH
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta.name}. Ejecuta `make evaluate` primero.")

    bt = pd.read_parquet(ruta)
    bt = bt[bt["modelo"].isin(["lgbm_conformal", "naive"])].copy()
    if bt.empty:
        log.warning("El backtest no contiene los modelos esperados; no se siembra nada")
        return 0

    panel, _ = cargar_panel_apto()
    precios = panel.dropna(subset=["precio_kg"]).set_index(
        ["producto_id", "plaza_id", "semana"]
    )["precio_kg"]
    bt["semana_origen"] = pd.to_datetime(bt["semana_origen"])
    bt["semana_objetivo"] = pd.to_datetime(bt["semana_objetivo"])
    bt["precio_origen"] = [
        float(precios.get(clave, np.nan))
        for clave in zip(
            bt["producto_id"], bt["plaza_id"], bt["semana_origen"], strict=True
        )
    ]
    bt["fecha_pronostico"] = bt["semana_origen"].dt.date.astype(str)
    bt["version_modelo"] = "backtest"
    bt["nivel_intervalo"] = np.where(bt["lo"].notna(), 0.8, np.nan) if "lo" in bt else np.nan
    bt["procedencia"] = "backfill"
    for col in ("lo", "hi"):
        if col not in bt.columns:
            bt[col] = np.nan

    llaves = (
        "procedencia",
        "version_modelo",
        "modelo",
        "producto_id",
        "plaza_id",
        "h",
        "semana_objetivo",
    )
    escritas = _anexar(bt[list(COLUMNAS_LOG)], BACKFILL_LOG, llaves)
    log.info("Sembradas %d filas de backfill desde el backtest", escritas)
    return escritas


def main() -> int:
    parser = argparse.ArgumentParser(description="Pronostica y registra en el log en vivo")
    parser.add_argument("--version", help="versión del modelo (por defecto, latest)")
    parser.add_argument(
        "--sembrar-backtest",
        action="store_true",
        help="añade los pronósticos del walk-forward marcados como backfill",
    )
    args = parser.parse_args()
    setup_logging()
    if args.sembrar_backtest:
        sembrar_desde_backtest()
    forecast(version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
