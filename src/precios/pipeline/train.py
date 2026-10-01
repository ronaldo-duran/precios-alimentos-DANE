"""Etapa `train`: entrena el modelo de producción y lo registra.

Entrada: `data/processed/semanal.parquet`
Salida:  `models/<fecha>_<hash>/` con el modelo, su metadata y un puntero
         `models/latest.json`

A diferencia de `evaluate`, que reajusta el modelo en 61 orígenes para medirlo,
aquí se entrena **una sola vez con toda la historia disponible**, que es el
modelo que producirá los pronósticos reales.

Se registra el `ConformalGlobal`: su pronóstico puntual es algo peor que el de
la regresión cuantílica (MASE 1,227 frente a 1,154 en h=1), pero sus intervalos
son los únicos que cubren lo que prometen (79-81% frente a 65-74%). Para un
producto que publica bandas de incertidumbre, esa es la elección correcta.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from precios.config import PROCESSED_DIR, ahora, hoy, load_evaluacion, load_lgbm_params
from precios.features.build import anadir_objetivos, construir_features
from precios.logging_setup import setup_logging
from precios.models import registry
from precios.models.conformal import ConformalGlobal
from precios.pipeline.evaluate import NIVEL_INTERVALO, cargar_panel_apto, fijar_semillas

log = logging.getLogger(__name__)

METRICAS_PATH = PROCESSED_DIR / "metricas_por_horizonte.csv"


def train(*, forzar: bool = False) -> Path:
    """Entrena con toda la historia y registra el resultado.

    Args:
        forzar: reentrena aunque el hash indique que esa versión ya existe.

    Returns:
        Ruta del directorio de la versión registrada.
    """
    cfg = load_evaluacion()
    fijar_semillas(cfg.semilla)
    params = load_lgbm_params()

    panel, _ = cargar_panel_apto(solo_aptas=cfg.solo_series_aptas)
    commit = registry.commit_actual()
    version_hash = registry.calcular_hash(panel, params, cfg.semilla, commit)
    version = f"{hoy().isoformat()}_{version_hash}"

    destino = registry.MODELS_DIR / version
    if destino.exists() and not forzar:
        log.info(
            "La versión %s ya está registrada; se reutiliza (usa --forzar para rehacerla)",
            version,
        )
        registry.marcar_latest(version)
        return destino

    features = anadir_objetivos(construir_features(panel), cfg.horizontes)
    # El origen es el número total de semanas: el modelo ve toda la historia.
    origen = int(features["t"].max()) + 1
    log.info("Entrenando con %d semanas de historia (origen=%d)", origen, origen)

    modelo = ConformalGlobal(
        horizontes=cfg.horizontes,
        nivel=NIVEL_INTERVALO,
        params=params,
        semilla=cfg.semilla,
    ).fit(features, origen)

    metadata = registry.Metadata(
        version=version,
        entrenado_en=ahora().isoformat(timespec="seconds"),
        semana_inicio=str(panel["semana"].min().date()),
        semana_fin=str(panel["semana"].max().date()),
        n_series=int(panel.groupby(["producto_id", "plaza_id"], observed=True).ngroups),
        n_observaciones=int(panel["precio_kg"].notna().sum()),
        productos=sorted(panel["producto_id"].unique()),
        plazas=sorted(panel["plaza_id"].unique()),
        horizontes=list(cfg.horizontes),
        nivel_intervalo=NIVEL_INTERVALO,
        semilla=cfg.semilla,
        params_lgbm=params,
        commit=commit,
        radios_conformales={str(h): r for h, r in modelo._radio.items()},
        metricas=_metricas_registradas(),
    )

    extras = {"diagnostico_intervalos": modelo.diagnostico}
    ruta = registry.guardar(modelo, metadata, extras=extras)
    log.info(
        "Versión %s registrada | datos %s -> %s | %d series",
        version,
        metadata.semana_inicio,
        metadata.semana_fin,
        metadata.n_series,
    )
    return ruta


def _metricas_registradas() -> dict:
    """Adjunta las métricas del último `evaluate`, si existen.

    Son las que justifican por qué este modelo está en producción; guardarlas
    con el artefacto evita tener que cruzar archivos meses después.
    """
    if not METRICAS_PATH.exists():
        log.warning("Sin %s: el modelo se registra sin métricas", METRICAS_PATH.name)
        return {}
    tabla = pd.read_csv(METRICAS_PATH)
    columnas = [c for c in ("modelo", "h", "mase", "mae", "cobertura_pct") if c in tabla.columns]
    return json.loads(tabla[columnas].to_json(orient="records"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena y registra el modelo de producción")
    parser.add_argument("--forzar", action="store_true", help="reentrena aunque la versión exista")
    args = parser.parse_args()
    setup_logging()
    train(forzar=args.forzar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
