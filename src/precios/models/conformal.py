"""Intervalos por predicción conformal dividida (split conformal).

La regresión cuantílica de LightGBM aprende los cuantiles del objetivo, pero
nada garantiza que su cobertura empírica sea la nominal: un q10/q90 puede cubrir
el 60% o el 95% de las veces. La predicción conformal ataca justo eso: calibra
el ancho del intervalo con **residuos fuera de muestra**, y bajo intercambiabilidad
garantiza la cobertura marginal pedida.

Con series de tiempo la intercambiabilidad no se cumple del todo (los residuos
están correlacionados y el régimen cambia), así que la garantía es aproximada.
Por eso el proyecto **mide** la cobertura empírica en vez de darla por buena.

La calibración se hace en el espacio del log-cambio, de modo que el intervalo
resulta multiplicativo sobre el precio: apropiado para una variable positiva y
de varianza proporcional al nivel.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from precios.models.lgbm import LGBMGlobal

log = logging.getLogger(__name__)


class ConformalGlobal:
    """LightGBM global con intervalos calibrados por split conformal.

    El entrenamiento se parte en dos: un tramo *propio* con el que se ajusta el
    modelo y un tramo de *calibración*, posterior, que el modelo nunca ve y del
    que salen los residuos que fijan el ancho del intervalo.
    """

    name = "lgbm_conformal"

    def __init__(
        self,
        horizontes: Sequence[int] = (1, 2, 3, 4),
        *,
        nivel: float = 0.8,
        semanas_calibracion: int = 26,
        params: dict | None = None,
        semilla: int = 42,
    ) -> None:
        if not 0 < nivel < 1:
            raise ValueError("nivel debe estar entre 0 y 1")
        self.horizontes = tuple(horizontes)
        self.nivel = nivel
        self.semanas_calibracion = semanas_calibracion
        self._interno = LGBMGlobal(
            horizontes=horizontes, cuantiles=(0.5,), params=params, semilla=semilla
        )
        self._radio: dict[int, float] = {}
        self._n_calibracion: dict[int, int] = {}
        self._residuos_con_signo: dict[int, np.ndarray] = {}

    def fit(self, features: pd.DataFrame, origen: int) -> ConformalGlobal:
        """Ajusta el modelo en el tramo propio y calibra en el posterior.

        El corte se define sobre el origen, no sobre el máximo de la tabla: el
        modelo interno se entrena como si el origen fuera `origen - n_calib`, y
        las semanas intermedias quedan libres para calibrar sin haber sido vistas.
        """
        origen_propio = origen - self.semanas_calibracion
        if origen_propio <= max(self.horizontes) + 1:
            raise ValueError(
                f"origen={origen} demasiado temprano para calibrar con "
                f"{self.semanas_calibracion} semanas"
            )

        self._interno.fit(features, origen_propio)
        self._radio.clear()
        self._n_calibracion.clear()
        self._residuos_con_signo.clear()

        for h in self.horizontes:
            residuos = self._residuos_calibracion(features, origen, origen_propio, h)
            if residuos.size == 0:
                continue
            self._residuos_con_signo[h] = residuos
            self._radio[h] = _cuantil_conformal(np.abs(residuos), self.nivel)
            self._n_calibracion[h] = int(residuos.size)
        return self

    def residuos_con_signo(self, h: int) -> np.ndarray:
        """Residuos de calibración con signo para el horizonte `h`.

        Los intervalos solo necesitan su valor absoluto, pero una pregunta
        direccional —¿cuál es la probabilidad de que el precio *suba* más de
        un umbral?— necesita el signo. Es la distribución predictiva empírica
        alrededor del pronóstico puntual, y viene ya calibrada fuera de muestra.
        """
        return self._residuos_con_signo.get(h, np.array([]))

    def _residuos_calibracion(
        self, features: pd.DataFrame, origen: int, origen_propio: int, h: int
    ) -> np.ndarray:
        """Residuos en log-cambio (con signo) del tramo que el modelo no vio.

        Son las filas posteriores al entrenamiento propio cuyo objetivo aún
        ocurrió antes del origen real.
        """
        mascara = (features["t"] > origen_propio - 1 - h) & (features["t"] <= origen - 1 - h)
        calibracion = features[mascara]
        objetivo = calibracion[f"target_{h}"]
        valido = objetivo.notna() & calibracion["y"].notna()
        filas = calibracion[valido]
        if filas.empty:
            return np.array([])

        modelo = self._interno._modelos.get((h, 0.5))
        if modelo is None:
            return np.array([])
        pred_log = modelo.predict(filas[self._interno._features])
        # Se devuelven CON signo; quien necesite el valor absoluto lo toma.
        return objetivo[valido].to_numpy(dtype=float) - pred_log

    def predict(self, predecir: pd.DataFrame) -> pd.DataFrame:
        """Pronóstico puntual con intervalo conformal simétrico en log."""
        # El modelo interno solo entrena el cuantil 0,5, así que su `lo`/`hi`
        # coinciden con el punto y no aportan nada: se descartan y el intervalo
        # se reconstruye con el radio conformal.
        base = self._interno.predict(predecir).drop(columns=["lo", "hi"], errors="ignore")
        radios = base["h"].map(self._radio)
        factor = np.exp(radios.astype(float))
        base["lo"] = base["y_pred"] / factor
        base["hi"] = base["y_pred"] * factor
        return base

    @property
    def diagnostico(self) -> pd.DataFrame:
        """Radio del intervalo y tamaño de calibración por horizonte."""
        return pd.DataFrame(
            {
                "h": list(self._radio),
                "radio_log": [self._radio[h] for h in self._radio],
                "ancho_pct": [100 * (np.exp(self._radio[h]) - 1) for h in self._radio],
                "n_calibracion": [self._n_calibracion.get(h, 0) for h in self._radio],
            }
        )


def _cuantil_conformal(residuos: np.ndarray, nivel: float) -> float:
    """Cuantil conformal con la corrección de muestra finita.

    Se usa el índice `ceil((n + 1) * nivel) / n` en vez del cuantil empírico
    directo: es lo que da la garantía de cobertura >= nivel, y con n pequeño la
    diferencia importa.
    """
    n = residuos.size
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * nivel))
    if k > n:
        # Con muy pocos puntos no se puede garantizar el nivel; se usa el máximo.
        return float(np.max(residuos))
    return float(np.sort(residuos)[k - 1])
