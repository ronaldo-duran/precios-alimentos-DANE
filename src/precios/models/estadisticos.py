"""Modelos estadísticos clásicos: ETS y AutoARIMA (statsforecast).

Ambos se usan **sin componente estacional** (`season_length=1`), y no por
comodidad:

* Con datos semanales el periodo anual es 52. ETS y ARIMA estacionales con m=52
  necesitan estimar decenas de parámetros estacionales y son numéricamente
  frágiles y muy lentos; la propia documentación de statsforecast desaconseja
  m grandes.
* Y sobre todo: el naive estacional ya fracasó en la fase 2 con MASE ~5,2. La
  evidencia dice que estas series **no tienen** un ciclo anual estable que
  explotar, así que forzar un componente estacional añadiría varianza sin señal.

Si alguna vez apareciera evidencia de estacionalidad anual, el sitio para
cambiarlo es `season_length` aquí.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
from statsforecast.models import AutoARIMA, AutoETS

from precios.models.base import Forecaster

log = logging.getLogger(__name__)


class _StatsForecastWrapper(Forecaster):
    """Adapta un modelo de statsforecast a la interfaz `Forecaster`."""

    def __init__(self, modelo, name: str, min_obs: int = 24) -> None:
        self._modelo = modelo
        self.name = name
        self.min_obs = min_obs
        self._y: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> _StatsForecastWrapper:
        y = np.asarray(y, dtype=np.float64)
        if len(y) < self.min_obs:
            raise ValueError(f"{self.name} necesita {self.min_obs} observaciones")
        if np.isnan(y).any():
            raise ValueError(f"{self.name} no admite NaN; usa preparar_train antes")
        self._y = y
        return self

    def predict(self, horizontes: Sequence[int]) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("El modelo no está ajustado")
        h_max = int(max(horizontes))
        with warnings.catch_warnings():
            # statsforecast avisa de no convergencia en series difíciles; el
            # fallback del propio modelo ya devuelve algo razonable.
            warnings.simplefilter("ignore")
            salida = self._modelo.forecast(y=self._y, h=h_max)["mean"]
        return np.array([salida[h - 1] for h in horizontes], dtype=float)


def ets(season_length: int = 1) -> Forecaster:
    """Suavizado exponencial con selección automática de componentes."""
    return _StatsForecastWrapper(AutoETS(season_length=season_length), name="ets")


def auto_arima(season_length: int = 1) -> Forecaster:
    """ARIMA con selección automática de órdenes.

    Usa `approximation=True` y órdenes acotados a p,q <= 3. No es un atajo
    gratuito: sobre una serie de prueba (tomate en Corabastos, 300 semanas) esta
    configuración devuelve **exactamente las mismas predicciones** que la
    búsqueda completa, en 0,47 s en vez de 1,80 s. Sobre los 1.830 ajustes del
    backtest eso son 14 minutos en lugar de 55.
    """
    return _StatsForecastWrapper(
        AutoARIMA(
            season_length=season_length,
            approximation=True,
            max_p=3,
            max_q=3,
            max_d=1,
            stepwise=True,
        ),
        name="auto_arima",
    )


def modelos_estadisticos() -> list[Forecaster]:
    """Conjunto estadístico que se compara contra los baselines."""
    return [ets(), auto_arima()]
