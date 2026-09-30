"""Alertas de alza de precio, con umbral por producto.

Un umbral plano no sirve. Con 10% fijo, la alerta se dispararía el 13,7% de las
semanas en yuca y el 36,6% en tomate: la misma etiqueta significando cosas
distintas. Los umbrales de `config/alertas.yaml` se derivan del percentil 75 de
las alzas históricas a 2 semanas de cada producto, lo que iguala la tasa base
en el rango 10-17% y hace las alertas comparables entre sí.

La probabilidad sale de los **residuos de calibración conformal con signo**, que
son la distribución predictiva empírica alrededor del pronóstico puntual, ya
validada fuera de muestra. No se asume normalidad.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from precios.config import CONFIG_DIR

log = logging.getLogger(__name__)

#: Puntos de la rejilla de cuantiles con la que se acoplan los horizontes.
_N_REJILLA = 2000


def cargar_config(path: Path | None = None) -> dict:
    """Lee `config/alertas.yaml`."""
    ruta = path or CONFIG_DIR / "alertas.yaml"
    with ruta.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def umbral_de(producto_id: str, cfg: dict) -> float:
    """Umbral de alza para un producto, con respaldo al valor por defecto."""
    return float(cfg.get("umbrales", {}).get(producto_id, cfg.get("umbral_por_defecto", 0.15)))


def probabilidad_alza(
    pred_log: dict[int, float],
    residuos: dict[int, np.ndarray],
    umbral: float,
    horizontes: Sequence[int],
) -> float:
    """Probabilidad de que el precio suba más de `umbral` en algún horizonte.

    Los horizontes se acoplan de forma **comonotónica**: se evalúan todos en el
    mismo nivel de cuantil. Suponer independencia daría probabilidades infladas
    (los errores de h=1 y h=2 están muy correlacionados: si el precio se
    dispara, se dispara para ambos), y el máximo por horizonte ignoraría que
    dos oportunidades son más que una. El acople comonotónico es el supuesto
    honesto entre esos dos extremos para series tan persistentes como estas.

    Args:
        pred_log: pronóstico de `log(y[t+h]/y[t])` por horizonte.
        residuos: residuos de calibración con signo por horizonte.
        umbral: alza relativa mínima (0,25 = 25%).
        horizontes: horizontes a considerar (p. ej. 1 y 2).

    Returns:
        Probabilidad entre 0 y 1, o NaN si falta calibración.
    """
    utiles = [h for h in horizontes if h in pred_log and residuos.get(h, np.array([])).size > 0]
    if not utiles:
        return float("nan")

    niveles = (np.arange(_N_REJILLA) + 0.5) / _N_REJILLA
    objetivo = np.log1p(umbral)

    # Para cada nivel de cuantil, el mayor cambio logrado entre los horizontes.
    mejor = np.full(_N_REJILLA, -np.inf)
    for h in utiles:
        r = np.quantile(residuos[h], niveles)
        mejor = np.maximum(mejor, pred_log[h] + r)
    return float(np.mean(mejor > objetivo))


def generar_alertas(
    predicciones: pd.DataFrame,
    modelo_conformal,
    cfg: dict,
) -> pd.DataFrame:
    """Calcula la alerta de cada serie a partir de los pronósticos vigentes.

    Args:
        predicciones: salida de `ConformalGlobal.predict`, con `y_pred` y el
            precio base por serie en la columna `y_base`.
        modelo_conformal: modelo ya ajustado, del que se toman los residuos.
        cfg: contenido de `config/alertas.yaml`.

    Returns:
        Una fila por serie con umbral, probabilidad y si la alerta está activa.
    """
    horizonte_max = int(cfg.get("horizonte_semanas", 2))
    horizontes = list(range(1, horizonte_max + 1))
    prob_min = float(cfg.get("probabilidad_minima", 0.30))
    residuos = {h: modelo_conformal.residuos_con_signo(h) for h in horizontes}

    filas = []
    for (prod, plaza), g in predicciones.groupby(["producto_id", "plaza_id"], observed=True):
        base = float(g["y_base"].iloc[0])
        pred_log = {}
        for h in horizontes:
            fila = g[g["h"] == h]
            if fila.empty or base <= 0:
                continue
            pred_log[h] = float(np.log(fila["y_pred"].iloc[0] / base))

        umbral = umbral_de(prod, cfg)
        p = probabilidad_alza(pred_log, residuos, umbral, horizontes)
        filas.append(
            {
                "producto_id": prod,
                "plaza_id": plaza,
                "precio_actual": base,
                "umbral_alza": umbral,
                "precio_umbral": base * (1 + umbral),
                "horizonte_semanas": horizonte_max,
                "probabilidad": p,
                "alerta_activa": bool(np.isfinite(p) and p >= prob_min),
            }
        )

    out = pd.DataFrame(filas)
    n = int(out["alerta_activa"].sum()) if not out.empty else 0
    log.info(
        "Alertas activas: %d de %d series (umbral de probabilidad %.0f%%)",
        n,
        len(out),
        100 * prob_min,
    )
    return out


def derivar_umbrales(panel: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Recalcula los umbrales por producto desde el histórico.

    Devuelve la tabla con el umbral crudo, el redondeado y la tasa base que
    resultaría, para poder revisar el efecto antes de escribirlo en el config.
    """
    cfg = cfg or cargar_config()
    d = cfg.get("derivacion", {})
    cuantil = float(d.get("cuantil", 0.75))
    redondeo = float(d.get("redondeo", 0.05))
    piso = float(d.get("piso", 0.10))
    h = int(cfg.get("horizonte_semanas", 2))

    p = panel.dropna(subset=["precio_kg"]).sort_values(["producto_id", "plaza_id", "semana"])
    log_precio = np.log(p["precio_kg"].where(p["precio_kg"] > 0))
    cambio = log_precio.groupby([p["producto_id"], p["plaza_id"]], observed=True).diff(h)
    p = p.assign(cambio=cambio).dropna(subset=["cambio"])

    filas = []
    for prod, g in p.groupby("producto_id", observed=True):
        alzas = g.loc[g["cambio"] > 0, "cambio"]
        if alzas.empty:
            continue
        crudo = float(np.expm1(alzas.quantile(cuantil)))
        umbral = max(piso, round(crudo / redondeo) * redondeo)
        tasa = float(100 * (g["cambio"] > np.log1p(umbral)).mean())
        filas.append(
            {
                "producto_id": prod,
                "umbral_crudo_pct": round(100 * crudo, 1),
                "umbral_pct": round(100 * umbral, 0),
                "umbral": round(umbral, 2),
                "tasa_base_pct": round(tasa, 1),
            }
        )
    return pd.DataFrame(filas).sort_values("umbral").reset_index(drop=True)


def main() -> int:  # pragma: no cover - utilidad de línea de comandos
    import argparse

    from precios.logging_setup import setup_logging
    from precios.pipeline.evaluate import cargar_panel_apto

    parser = argparse.ArgumentParser(description="Utilidades de alertas")
    parser.add_argument(
        "--derivar-umbrales",
        action="store_true",
        help="recalcula los umbrales por producto y los imprime",
    )
    args = parser.parse_args()
    setup_logging()

    if args.derivar_umbrales:
        panel, _ = cargar_panel_apto()
        tabla = derivar_umbrales(panel)
        log.info("Umbrales derivados:\n%s", tabla.to_string(index=False))
        log.info("Cópialos a config/alertas.yaml si te convencen.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
