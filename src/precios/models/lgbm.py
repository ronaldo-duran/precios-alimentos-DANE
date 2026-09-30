"""Modelo global de gradient boosting sobre todas las series a la vez.

Estrategia **directa**: un modelo independiente por horizonte, en vez de iterar
un modelo de un paso. Evita acumular error de forma opaca y permite que cada
horizonte aprenda su propia dinámica (a 1 semana manda la inercia; a 4, la
reversión a la media).

Los intervalos salen de **regresión cuantílica**: tres modelos por horizonte
(q10, q50, q90). El q50 es además el pronóstico puntual, lo que alinea el
entrenamiento con el MAE, que es la base del MASE.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from precios.features.build import CATEGORICAS, columnas_features, filas_entrenables

log = logging.getLogger(__name__)

#: Hilos por ajuste. `n_jobs=-1` es una trampa aquí: con ~9.000 filas y 23
#: features el coste de sincronizar 12 hilos domina sobre el trabajo útil y el
#: ajuste tarda 21 s en vez de 2 s (medido en esta máquina, 12 núcleos). Se
#: acota a 4 hilos, que es donde deja de compensar.
N_HILOS: int = min(4, os.cpu_count() or 1)

#: Hiperparámetros por defecto. Deliberadamente modestos: el objetivo es medir
#: si el enfoque global aporta algo, no exprimir décimas con una búsqueda larga.
PARAMS_BASE: dict = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": N_HILOS,
}

#: Cuantiles entrenados: el 10 y el 90 forman el intervalo del 80%.
CUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)


class LGBMGlobal:
    """Entrena un LightGBM por (horizonte, cuantil) sobre el panel completo.

    A diferencia de los baselines, este modelo no es univariado: ve todas las
    series juntas y usa `producto_id` y `plaza_id` como categóricas, de modo que
    puede transferir patrones entre plazas del mismo producto.
    """

    name = "lgbm_global"

    def __init__(
        self,
        horizontes: Sequence[int] = (1, 2, 3, 4),
        cuantiles: Sequence[float] = CUANTILES,
        params: dict | None = None,
        semilla: int = 42,
    ) -> None:
        self.horizontes = tuple(horizontes)
        self.cuantiles = tuple(cuantiles)
        self.params = {**PARAMS_BASE, **(params or {})}
        self.semilla = semilla
        self._modelos: dict[tuple[int, float], lgb.LGBMRegressor] = {}
        self._features: list[str] = []

    def fit(self, features: pd.DataFrame, origen: int) -> LGBMGlobal:
        """Entrena un modelo por (horizonte, cuantil) con datos previos al origen.

        Args:
            features: panel completo de features con las columnas `target_h`
                añadidas por `anadir_objetivos`.
            origen: número de semanas conocidas. Para cada horizonte solo se usan
                filas con `t <= origen - 1 - h`, es decir, aquellas cuyo objetivo
                ya se había observado.
        """
        self._features = columnas_features(features)
        self._modelos.clear()

        for h in self.horizontes:
            train = filas_entrenables(features, origen, h)
            objetivo = train[f"target_{h}"]
            valido = objetivo.notna() & train["y"].notna()
            X = train.loc[valido, self._features]
            y = objetivo[valido]
            if len(X) < 100:
                log.warning("h=%d: solo %d filas de entrenamiento, se omite", h, len(X))
                continue
            for q in self.cuantiles:
                modelo = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    random_state=self.semilla,
                    **self.params,
                )
                modelo.fit(X, y, categorical_feature=list(CATEGORICAS))
                self._modelos[(h, q)] = modelo
        return self

    def predict(self, predecir: pd.DataFrame) -> pd.DataFrame:
        """Pronostica todos los horizontes para las filas dadas.

        Devuelve niveles en COP/kg: el modelo predice `log(y[t+h]/y[t])` y aquí
        se reconstruye `y[t] * exp(pred)`.
        """
        if not self._modelos:
            raise RuntimeError("El modelo no está entrenado")

        X = predecir[self._features]
        base = predecir["y"].to_numpy(dtype=float)
        filas = []

        for h in self.horizontes:
            if (h, 0.5) not in self._modelos:
                continue
            pred = {q: self._modelos[(h, q)].predict(X) for q in self.cuantiles}
            q_bajo, q_alto = min(self.cuantiles), max(self.cuantiles)
            # El boosting cuantílico puede cruzar cuantiles; se reordena.
            lo_log = np.minimum(pred[q_bajo], pred[q_alto])
            hi_log = np.maximum(pred[q_bajo], pred[q_alto])
            medio_log = np.clip(pred[0.5], lo_log, hi_log)

            filas.append(
                pd.DataFrame(
                    {
                        "producto_id": predecir["producto_id"].to_numpy(),
                        "plaza_id": predecir["plaza_id"].to_numpy(),
                        "h": h,
                        "y_pred": base * np.exp(medio_log),
                        "lo": base * np.exp(lo_log),
                        "hi": base * np.exp(hi_log),
                    }
                )
            )
        return pd.concat(filas, ignore_index=True)

    def importancias(self) -> pd.DataFrame:
        """Importancia media por feature, agregada sobre horizontes (solo q50)."""
        filas = []
        for (h, q), modelo in self._modelos.items():
            if q != 0.5:
                continue
            filas.append(
                pd.DataFrame(
                    {"h": h, "feature": self._features, "ganancia": modelo.feature_importances_}
                )
            )
        if not filas:
            return pd.DataFrame(columns=["feature", "ganancia"])
        todo = pd.concat(filas, ignore_index=True)
        return (
            todo.groupby("feature", as_index=False)["ganancia"]
            .mean()
            .sort_values("ganancia", ascending=False)
            .reset_index(drop=True)
        )
