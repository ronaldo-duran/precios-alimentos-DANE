"""Validación rolling-origin (walk-forward) con ventana expansiva.

Un split aleatorio sobre series de tiempo está **prohibido**: entrenaría con
semanas posteriores a las que pronostica y daría métricas fantasía. Aquí cada
fold define un origen y el modelo solo ve `y[:origen]`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """Un origen de pronóstico y las posiciones que debe predecir.

    Attributes:
        origen: número de observaciones de entrenamiento. `y[:origen]` es lo
            único que el modelo puede ver; `y[origen]` es el horizonte 1.
        horizontes: horizontes evaluados, en semanas (1 = la semana siguiente).
    """

    origen: int
    horizontes: tuple[int, ...]

    @property
    def indices_test(self) -> tuple[int, ...]:
        """Posiciones del array original que se pronostican."""
        return tuple(self.origen + h - 1 for h in self.horizontes)


def rolling_origins(
    n_obs: int,
    *,
    min_train: int = 104,
    horizonte_max: int = 4,
    paso: int = 4,
    max_origenes: int | None = None,
) -> list[Fold]:
    """Genera folds de origen móvil con ventana expansiva.

    Args:
        n_obs: longitud de la serie.
        min_train: observaciones mínimas antes del primer origen.
        horizonte_max: horizonte máximo a evaluar (1..horizonte_max).
        paso: separación en semanas entre orígenes consecutivos.
        max_origenes: si se indica, conserva solo los N orígenes más recientes.

    Returns:
        Lista de folds ordenada del origen más antiguo al más reciente. Vacía si
        la serie no alcanza para un solo origen completo.
    """
    if min_train < 1:
        raise ValueError("min_train debe ser >= 1")
    if horizonte_max < 1:
        raise ValueError("horizonte_max debe ser >= 1")
    if paso < 1:
        raise ValueError("paso debe ser >= 1")

    horizontes = tuple(range(1, horizonte_max + 1))
    # El último origen válido deja sitio para el horizonte completo.
    ultimo_origen = n_obs - horizonte_max
    if ultimo_origen < min_train:
        log.debug(
            "Serie de %d obs insuficiente para min_train=%d y horizonte=%d",
            n_obs,
            min_train,
            horizonte_max,
        )
        return []

    origenes = list(range(min_train, ultimo_origen + 1, paso))
    if max_origenes is not None and len(origenes) > max_origenes:
        # Se conservan los más recientes: son los que reflejan el régimen actual.
        origenes = origenes[-max_origenes:]
    return [Fold(origen=o, horizontes=horizontes) for o in origenes]


def describir_folds(folds: list[Fold]) -> str:
    """Resumen legible para el log."""
    if not folds:
        return "sin folds"
    return (
        f"{len(folds)} orígenes, de la obs {folds[0].origen} a la {folds[-1].origen}, "
        f"horizontes {folds[0].horizontes}"
    )
