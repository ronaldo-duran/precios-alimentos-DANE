"""Baselines obligatorios.

Son la vara de medir: un modelo sofisticado que no le gana al naive no sirve de
nada, y decirlo es parte del objetivo del proyecto. El MASE se define justamente
contra el naive, así que `NaiveUltimo` no es solo un competidor, es la escala.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from precios.models.base import Forecaster

#: Periodo estacional de una serie semanal con ciclo anual.
PERIODO_ANUAL = 52


class NaiveUltimo(Forecaster):
    """El pronóstico es el último valor observado, para todos los horizontes.

    Para un precio que se parece a un paseo aleatorio, es sorprendentemente
    difícil de batir.
    """

    name = "naive"
    min_obs = 1

    def fit(self, y: np.ndarray) -> NaiveUltimo:
        y = np.asarray(y, dtype=float)
        if len(y) < self.min_obs:
            raise ValueError(f"{self.name} necesita al menos {self.min_obs} observación")
        self._ultimo = float(y[-1])
        return self

    def predict(self, horizontes: Sequence[int]) -> np.ndarray:
        return np.full(len(horizontes), self._ultimo, dtype=float)


class NaiveEstacional(Forecaster):
    """El pronóstico es el valor de la misma semana del año anterior.

    Con `periodo=52`, el horizonte h toma `y[-52 + h - 1]`. Necesita al menos un
    año completo de historia; con menos, `fit` falla en vez de inventar.
    """

    name = "naive_estacional"

    def __init__(self, periodo: int = PERIODO_ANUAL) -> None:
        self.periodo = periodo
        self.min_obs = periodo

    def fit(self, y: np.ndarray) -> NaiveEstacional:
        y = np.asarray(y, dtype=float)
        if len(y) < self.min_obs:
            raise ValueError(
                f"{self.name} necesita {self.min_obs} observaciones y recibió {len(y)}"
            )
        self._y = y
        return self

    def predict(self, horizontes: Sequence[int]) -> np.ndarray:
        salida = np.empty(len(horizontes), dtype=float)
        for i, h in enumerate(horizontes):
            # Retrocede un periodo completo y avanza h pasos dentro de él.
            idx = len(self._y) - self.periodo + (h - 1)
            # Si h supera el periodo, se recicla el ciclo anterior disponible.
            while idx >= len(self._y):
                idx -= self.periodo
            salida[i] = self._y[idx]
        return salida


class PromedioMovil(Forecaster):
    """Promedio de las últimas `ventana` semanas, plano en el horizonte.

    Suaviza el ruido semanal a costa de reaccionar tarde a los cambios de nivel:
    exactamente el compromiso que interesa medir contra el naive.
    """

    name = "promedio_movil"

    def __init__(self, ventana: int = 4) -> None:
        if ventana < 1:
            raise ValueError("ventana debe ser >= 1")
        self.ventana = ventana
        self.min_obs = ventana
        self.name = f"promedio_movil_{ventana}"

    def fit(self, y: np.ndarray) -> PromedioMovil:
        y = np.asarray(y, dtype=float)
        if len(y) < self.min_obs:
            raise ValueError(
                f"{self.name} necesita {self.min_obs} observaciones y recibió {len(y)}"
            )
        self._media = float(np.mean(y[-self.ventana :]))
        return self

    def predict(self, horizontes: Sequence[int]) -> np.ndarray:
        return np.full(len(horizontes), self._media, dtype=float)


class DerivaNaive(Forecaster):
    """Naive con deriva: extiende la tendencia media de toda la historia.

    No lo pide el enunciado, pero es el baseline que suele delatar si un modelo
    "gana" solo por capturar una tendencia lineal trivial.
    """

    name = "naive_deriva"
    min_obs = 2

    def fit(self, y: np.ndarray) -> DerivaNaive:
        y = np.asarray(y, dtype=float)
        if len(y) < self.min_obs:
            raise ValueError(f"{self.name} necesita al menos {self.min_obs} observaciones")
        self._ultimo = float(y[-1])
        self._deriva = float((y[-1] - y[0]) / (len(y) - 1))
        return self

    def predict(self, horizontes: Sequence[int]) -> np.ndarray:
        return np.array([self._ultimo + self._deriva * h for h in horizontes], dtype=float)


def baselines_por_defecto() -> list[Forecaster]:
    """Conjunto de baselines que se evalúa siempre."""
    return [
        NaiveUltimo(),
        NaiveEstacional(PERIODO_ANUAL),
        PromedioMovil(4),
        PromedioMovil(8),
        DerivaNaive(),
    ]
