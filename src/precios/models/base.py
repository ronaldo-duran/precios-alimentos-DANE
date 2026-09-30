"""Contrato común de los modelos de pronóstico.

Un `Forecaster` ve **solo** la historia hasta el origen del pronóstico. Esa es
la regla que evita el data leakage, y está testeada en `tests/test_leakage.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Forecaster(ABC):
    """Pronosticador univariado sobre una serie semanal."""

    #: Identificador corto que aparece en las tablas de resultados.
    name: str

    #: Mínimo de observaciones que necesita para producir un pronóstico.
    min_obs: int = 1

    @abstractmethod
    def fit(self, y: np.ndarray) -> Forecaster:
        """Ajusta el modelo con la historia disponible hasta el origen.

        Args:
            y: valores de la serie ordenados en el tiempo, sin NaN al final.
        """

    @abstractmethod
    def predict(self, horizontes: Sequence[int]) -> np.ndarray:
        """Pronostica los horizontes pedidos (1 = la semana siguiente al origen)."""

    def fit_predict(self, y: np.ndarray, horizontes: Sequence[int]) -> np.ndarray:
        return self.fit(y).predict(horizontes)

    def __repr__(self) -> str:  # pragma: no cover - conveniencia
        return f"{type(self).__name__}(name={self.name!r})"
