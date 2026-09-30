"""Detección de cambios de unidad de medida.

Es el fallo de datos más peligroso de todos, porque **no se parece a un error**.
Si el DANE pasara de cotizar en COP/kg a COP/500 g, todos los precios se
partirían a la mitad de un día para otro y el pipeline lo leería como una caída
generalizada del 50%: entrenaría con ella, la pronosticaría y la publicaría.

El detector de outliers no lo cubre. Ese busca **picos** —un salto seguido de
una vuelta a la normalidad— y un cambio de unidad es un **escalón**: salta una
vez y se queda. Después del escalón los retornos vuelven a ser normales, así
que el z-score robusto no ve nada.

## Cómo se llegó a este criterio

Las dos primeras versiones de este detector fallaron de formas opuestas, y
conviene dejarlo escrito porque explica el diseño final.

La primera miraba **serie por serie**: buscaba un salto por un factor redondo
que persistiera, y lo reportaba si coincidían dos series. Sobre los datos
reales disparó **más de cien falsos positivos**: con productos de CV 40-57%,
que el tomate duplique su precio en ocho semanas es un martes cualquiera.

La segunda apretó la tolerancia al 3% por serie. Dejó de dar falsos positivos
y también **dejó de detectar un cambio de unidad inyectado a propósito**: cada
serie tiene su propia deriva, así que ninguna aterriza exactamente en ×0,5.

El error de fondo era mirar las series de una en una. Un cambio de unidad no
mueve series: mueve **el panel entero por el mismo factor**. Así que lo que hay
que medir es el **ratio transversal**: la mediana, entre todas las series, del
cambio de nivel de esa semana.

Medido sobre los datos reales (333 semanas, 30 series), ese ratio transversal
se mueve entre **0,707 y 1,427**, con mediana 1,014. Nunca se acerca a 0,5 ni a
2,0. Con un cambio de unidad inyectado (todo el panel a la mitad), cae a
**0,47**. La separación es limpia y por eso el criterio funciona.

El factor que se reporta es **orientativo, no un diagnóstico**: con la
tolerancia holgada que necesita el método, una conversión de kg a libras
(÷2,2046) puede quedar atribuida a ×0,5. La señal útil es "el panel entero
cambió de escala esta semana, ve a mirar la fuente", no el número exacto.

Se marca y se avisa, nunca se corrige: convertir un precio por un factor
adivinado sería mucho peor que reportarlo.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Factores que aparecen al cambiar una unidad de peso o de moneda. Se
#: comprueban también sus inversos.
FACTORES_SOSPECHOSOS: tuple[float, ...] = (2.0, 2.2046, 4.0, 10.0, 100.0, 1000.0)

#: Tolerancia alrededor de cada factor. Holgada a propósito: el panel conserva
#: su deriva natural bajo el escalón. Con 0,15 la banda de ×0,5 llega a 0,575,
#: muy por debajo del 0,707 mínimo observado en datos reales.
TOLERANCIA = 0.15

#: Series mínimas que deben aportar ratio esa semana para que la mediana
#: transversal signifique algo.
MIN_SERIES = 5

COLUMNAS = ("semana", "factor", "ratio_transversal", "n_series", "pct_en_banda")


def _factor_sospechoso(ratio: float) -> float | None:
    """Devuelve el factor redondo al que se parece el ratio, o None."""
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    for f in FACTORES_SOSPECHOSOS:
        for candidato in (f, 1 / f):
            if abs(ratio / candidato - 1) <= TOLERANCIA:
                return candidato
    return None


def ratios_transversales(panel: pd.DataFrame, ventana: int = 8) -> dict:
    """Ratio de nivel de cada serie en cada semana candidata.

    Para cada serie y semana compara la mediana de las `ventana` semanas
    siguientes con la de las `ventana` anteriores. Se usa la mediana y no el
    último valor para que un pico aislado no cuente como escalón.

    Returns:
        `{semana: array de ratios, uno por serie}`.
    """
    datos = panel.dropna(subset=["precio_kg"])
    por_semana: dict = {}
    for _, g in datos.groupby(["producto_id", "plaza_id"], observed=True):
        g = g.sort_values("semana").reset_index(drop=True)
        if len(g) < 2 * ventana:
            continue
        precios = g["precio_kg"].to_numpy(dtype=float)
        for corte in range(ventana, len(g) - ventana + 1):
            antes = np.median(precios[corte - ventana : corte])
            if antes <= 0:
                continue
            ratio = np.median(precios[corte : corte + ventana]) / antes
            por_semana.setdefault(g.loc[corte, "semana"], []).append(ratio)
    return {k: np.asarray(v) for k, v in por_semana.items()}


def detectar_cambio_unidad(
    panel: pd.DataFrame,
    *,
    ventana: int = 8,
    min_series: int = MIN_SERIES,
) -> pd.DataFrame:
    """Busca semanas en las que el panel entero saltó por un factor redondo.

    Args:
        panel: panel semanal con `producto_id`, `plaza_id`, `semana`, `precio_kg`.
        ventana: semanas a cada lado del corte.
        min_series: series mínimas que deben aportar ratio esa semana.

    Returns:
        Un `DataFrame` con las sospechas, vacío si no hay ninguna.
    """
    por_semana = ratios_transversales(panel, ventana)
    if not por_semana:
        return pd.DataFrame(columns=list(COLUMNAS))

    filas = []
    for semana, ratios in sorted(por_semana.items()):
        if len(ratios) < min_series:
            continue
        transversal = float(np.median(ratios))
        factor = _factor_sospechoso(transversal)
        if factor is None:
            continue
        # Qué proporción de series acompaña al movimiento, no solo la mediana.
        en_banda = float(np.mean(np.abs(ratios / factor - 1) <= TOLERANCIA * 2))
        filas.append(
            {
                "semana": semana,
                "factor": factor,
                "ratio_transversal": round(transversal, 4),
                "n_series": int(len(ratios)),
                "pct_en_banda": round(100 * en_banda, 1),
            }
        )

    if not filas:
        log.info(
            "Sin indicios de cambio de unidad (%d semanas revisadas, ratio "
            "transversal siempre lejos de un factor redondo)",
            len(por_semana),
        )
        return pd.DataFrame(columns=list(COLUMNAS))

    sospechas = pd.DataFrame(filas).sort_values("semana").reset_index(drop=True)
    for _, s in sospechas.iterrows():
        log.error(
            "POSIBLE CAMBIO DE UNIDAD el %s: el panel entero saltó por x%.4g "
            "(ratio transversal %.3f sobre %d series; %.0f%% de ellas acompañan). "
            "Revisa la fuente antes de usar estos datos; NO se corrige solo.",
            s["semana"].date(),
            s["factor"],
            s["ratio_transversal"],
            s["n_series"],
            s["pct_en_banda"],
        )
    return sospechas


def marcar_cambio_unidad(panel: pd.DataFrame, sospechas: pd.DataFrame) -> pd.DataFrame:
    """Añade `flag_cambio_unidad` desde la semana señalada en adelante.

    Si la unidad cambió, todo lo posterior está en la escala nueva.
    """
    panel = panel.copy()
    panel["flag_cambio_unidad"] = False
    if sospechas.empty:
        return panel
    desde = sospechas["semana"].min()
    panel.loc[panel["semana"] >= desde, "flag_cambio_unidad"] = True
    log.error(
        "Marcadas %d filas desde %s con flag_cambio_unidad",
        int(panel["flag_cambio_unidad"].sum()),
        desde.date(),
    )
    return panel
