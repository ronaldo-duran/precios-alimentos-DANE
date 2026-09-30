"""Motor de backtesting walk-forward sobre el panel semanal.

Regla invariable: en cada fold el modelo recibe `y[:origen]` y nada más. Incluso
el relleno de semanas faltantes se hace **dentro** de esa ventana, porque
interpolar la serie completa de antemano metería información del futuro en un
hueco anterior al origen. Está testeado en `tests/test_leakage.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from precios.evaluation.metrics import escala_mase
from precios.evaluation.splits import Fold, describir_folds, rolling_origins
from precios.features.build import filas_a_predecir
from precios.models.base import Forecaster

log = logging.getLogger(__name__)


def preparar_train(y_train: np.ndarray) -> np.ndarray:
    """Rellena huecos internos de la ventana de entrenamiento.

    Interpola linealmente los NaN interiores y propaga hacia atrás los
    iniciales. Solo usa valores de la propia ventana, así que no filtra futuro.

    Returns:
        Un array sin NaN, o con NaN si la ventana entera está vacía.
    """
    s = pd.Series(np.asarray(y_train, dtype=float))
    if s.notna().sum() == 0:
        return s.to_numpy()
    # `limit_area="inside"` deja fuera los extremos; bfill cubre el arranque.
    s = s.interpolate(method="linear", limit_area="inside").bfill().ffill()
    return s.to_numpy()


def backtest_serie(
    y: np.ndarray,
    semanas: pd.Series | np.ndarray,
    modelos: Sequence[Forecaster],
    folds: Sequence[Fold],
    *,
    producto_id: str = "",
    plaza_id: str = "",
) -> pd.DataFrame:
    """Corre todos los modelos sobre todos los folds de una serie.

    Los puntos objetivo sin dato observado se descartan: no se puede evaluar
    contra un hueco.
    """
    y = np.asarray(y, dtype=float)
    semanas = pd.to_datetime(pd.Series(semanas)).to_numpy()
    filas: list[dict] = []

    for fold in folds:
        y_train_bruto = y[: fold.origen]
        y_train = preparar_train(y_train_bruto)
        if np.isnan(y_train).all():
            continue

        escala = escala_mase(y_train, periodo=1)
        idx_test = np.array(fold.indices_test)
        y_true = y[idx_test]

        for modelo in modelos:
            if len(y_train) < getattr(modelo, "min_obs", 1):
                continue
            try:
                y_pred = modelo.fit(y_train).predict(fold.horizontes)
            except ValueError as err:  # historia insuficiente para este modelo
                log.debug("%s omitido en origen %d: %s", modelo.name, fold.origen, err)
                continue

            for i, h in enumerate(fold.horizontes):
                if np.isnan(y_true[i]):
                    continue  # semana faltante: no hay contra qué comparar
                filas.append(
                    {
                        "producto_id": producto_id,
                        "plaza_id": plaza_id,
                        "modelo": modelo.name,
                        "origen": fold.origen,
                        "semana_origen": semanas[fold.origen - 1],
                        "h": h,
                        "semana_objetivo": semanas[idx_test[i]],
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                        "escala_mase": escala,
                    }
                )

    return pd.DataFrame(filas)


def escalas_por_origen(
    features: pd.DataFrame, origenes: Sequence[int]
) -> dict[tuple[str, str, int], float]:
    """Precalcula la escala del MASE de cada serie en cada origen.

    Se usa exactamente el mismo criterio que en `backtest_serie` (MAE del naive
    de un paso sobre la ventana de entrenamiento), de modo que el MASE del
    modelo global sea directamente comparable con el de los baselines.
    """
    escalas: dict[tuple[str, str, int], float] = {}
    for (prod, plaza), g in features.groupby(["producto_id", "plaza_id"], observed=True):
        y = g.sort_values("t")["y"].to_numpy(dtype=float)
        for o in origenes:
            escalas[(prod, plaza, o)] = escala_mase(preparar_train(y[:o]), periodo=1)
    return escalas


def backtest_global(
    features: pd.DataFrame,
    constructores: dict[str, Callable[[], object]],
    folds: Sequence[Fold],
) -> pd.DataFrame:
    """Backtest de modelos de panel (ven todas las series a la vez).

    A diferencia de `backtest_panel`, aquí el modelo se reajusta una vez por
    origen sobre el panel completo y devuelve los pronósticos de todas las
    series de golpe.

    Args:
        features: panel de features con las columnas `target_h`.
        constructores: nombre -> función sin argumentos que crea el modelo.
        folds: orígenes a evaluar, los mismos que usan los baselines.
    """
    origenes = [f.origen for f in folds]
    escalas = escalas_por_origen(features, origenes)
    reales = features.set_index(["producto_id", "plaza_id", "t"])["y"]

    filas = []
    for nombre, constructor in constructores.items():
        for i, fold in enumerate(folds, 1):
            modelo = constructor()
            try:
                modelo.fit(features, fold.origen)
                pred = modelo.predict(filas_a_predecir(features, fold.origen))
            except (ValueError, RuntimeError) as err:
                log.debug("%s omitido en origen %d: %s", nombre, fold.origen, err)
                continue

            pred = pred.copy()
            pred["modelo"] = nombre
            pred["origen"] = fold.origen
            pred["t_objetivo"] = fold.origen + pred["h"] - 1
            claves = list(
                zip(pred["producto_id"], pred["plaza_id"], pred["t_objetivo"], strict=True)
            )
            pred["y_true"] = [reales.get(k, np.nan) for k in claves]
            pred["escala_mase"] = [
                escalas.get((prod, plaza, fold.origen), np.nan)
                for prod, plaza in zip(pred["producto_id"], pred["plaza_id"], strict=True)
            ]
            filas.append(pred.dropna(subset=["y_true"]))

            if i % 10 == 0 or i == len(folds):
                log.info("%s: %d/%d orígenes", nombre, i, len(folds))

    if not filas:
        raise RuntimeError("Ningún modelo global produjo pronósticos")

    out = pd.concat(filas, ignore_index=True)
    # Se añaden las semanas para poder cruzar con el panel y los episodios.
    semanas = features.drop_duplicates("t").set_index("t")["semana"]
    out["semana_origen"] = out["origen"].map(lambda o: semanas.get(o - 1))
    out["semana_objetivo"] = out["t_objetivo"].map(semanas.get)
    log.info("Backtest global: %d pronósticos de %d modelo(s)", len(out), len(constructores))
    return out


def backtest_panel(
    panel: pd.DataFrame,
    modelos: Sequence[Forecaster],
    *,
    min_train: int = 104,
    horizonte_max: int = 4,
    paso: int = 4,
    max_origenes: int | None = None,
    solo_aptas: pd.DataFrame | None = None,
    folds: Sequence[Fold] | None = None,
) -> pd.DataFrame:
    """Aplica `backtest_serie` a cada serie del panel semanal.

    Args:
        panel: salida de la etapa `clean` (una fila por serie y semana).
        modelos: pronosticadores a comparar.
        solo_aptas: si se pasa el `reporte_series`, se evalúan únicamente las
            series marcadas como aptas.
        folds: orígenes fijos para todas las series. Se usa para que los
            modelos univariados y los globales se midan exactamente sobre los
            mismos puntos; sin él, cada serie derivaría sus propios orígenes a
            partir de su longitud y la comparación dejaría de ser limpia.
    """
    aptas: set[tuple[str, str]] | None = None
    if solo_aptas is not None:
        aptas = set(
            zip(
                solo_aptas.loc[solo_aptas["apta"], "producto_id"],
                solo_aptas.loc[solo_aptas["apta"], "plaza_id"],
                strict=True,
            )
        )

    partes = []
    omitidas = 0
    for (prod, plaza), g in panel.groupby(["producto_id", "plaza_id"], observed=True):
        if aptas is not None and (prod, plaza) not in aptas:
            omitidas += 1
            continue
        g = g.sort_values("semana")
        if folds is None:
            folds_serie = rolling_origins(
                len(g),
                min_train=min_train,
                horizonte_max=horizonte_max,
                paso=paso,
                max_origenes=max_origenes,
            )
        else:
            # Con folds fijos se descartan los que no caben en esta serie.
            folds_serie = [f for f in folds if max(f.indices_test) <= len(g) - 1]
        if not folds_serie:
            log.warning("%s @ %s: sin folds (%d semanas)", prod, plaza, len(g))
            omitidas += 1
            continue
        partes.append(
            backtest_serie(
                g["precio_kg"].to_numpy(dtype=float),
                g["semana"],
                modelos,
                folds_serie,
                producto_id=prod,
                plaza_id=plaza,
            )
        )

    if not partes:
        raise RuntimeError("Ninguna serie pudo evaluarse; revisa min_train y el panel.")

    resultados = pd.concat(partes, ignore_index=True)
    ejemplo = folds or rolling_origins(
        int(panel.groupby(["producto_id", "plaza_id"], observed=True).size().max()),
        min_train=min_train,
        horizonte_max=horizonte_max,
        paso=paso,
        max_origenes=max_origenes,
    )
    log.info(
        "Backtest: %d pronósticos, %d series evaluadas (%d omitidas), %s",
        len(resultados),
        resultados.groupby(["producto_id", "plaza_id"], observed=True).ngroups,
        omitidas,
        describir_folds(ejemplo),
    )
    return resultados
