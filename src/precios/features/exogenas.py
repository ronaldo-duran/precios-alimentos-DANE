"""Variables exógenas: calendario de festivos y ENSO (índice ONI).

Las dos se activan desde `config/exogenas.yaml` y se miden con ablación: se
corre el mismo walk-forward con y sin cada bloque, y se reporta la diferencia
salga como salga.

**El calendario es un caso especial y conviene entender por qué.** Casi toda
feature de este proyecto solo puede mirar hacia atrás, pero los festivos de
2027 ya se conocen hoy: son deterministas. Por eso sí es legítimo darle al
modelo el calendario de la **semana objetivo**, no solo el de la semana del
origen. Ahí es justamente donde puede haber señal: una plaza cerrada el lunes
concentra la oferta en menos días.

**El ONI no tiene ese lujo.** La NOAA publica cada trimestre móvil con retraso,
así que se rezaga dos meses antes de unirlo a la semana. Sin ese rezago, el
modelo estaría viendo un valor que en esa fecha todavía no existía.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from precios.config import RAW_DIR

log = logging.getLogger(__name__)

URL_ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

#: Trimestres móviles de la NOAA, en orden. El mes central de 'DJF' es enero.
_TEMPORADAS = (
    "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
    "JJA", "JAS", "ASO", "SON", "OND", "NDJ",
)

#: Meses de rezago al unir el ONI a una semana. La NOAA publica un trimestre
#: móvil con ~1 mes de retraso; 2 meses da margen y evita mirar al futuro.
REZAGO_ONI_MESES = 2

#: Umbral estándar de la NOAA para declarar fase El Niño / La Niña.
UMBRAL_FASE = 0.5


# --- Calendario -------------------------------------------------------------


def _festivos(anios) -> set[dt.date]:
    """Festivos colombianos de los años pedidos.

    `holidays` implementa la Ley Emiliani, que traslada varios festivos al
    lunes siguiente: en 2026 Reyes pasa del 6 al 12 de enero y San José del 19
    al 23 de marzo. Hacerlo a mano sería una fuente de errores silenciosos.
    """
    import holidays

    return set(holidays.country_holidays("CO", years=list(anios)))


def features_calendario(semanas: pd.Series, horizontes=(1, 2, 3, 4)) -> pd.DataFrame:
    """Features de calendario por semana (lunes de cada semana ISO).

    Args:
        semanas: lunes de cada semana, únicos y ordenados.
        horizontes: para cada uno se añade el calendario de la semana objetivo.

    Returns:
        Una fila por semana, indexada por `semana`.
    """
    semanas = pd.to_datetime(pd.Series(sorted(set(pd.to_datetime(semanas))), name="semana"))
    anios = range(semanas.min().year - 1, semanas.max().year + 2)
    festivos = _festivos(anios)

    filas = []
    for lunes in semanas:
        base = {"semana": lunes}
        base.update(_resumen_semana(lunes, festivos, sufijo=""))
        for h in horizontes:
            objetivo = lunes + pd.Timedelta(7 * h, unit="D")
            base.update(_resumen_semana(objetivo, festivos, sufijo=f"_h{h}"))
        filas.append(base)

    out = pd.DataFrame(filas).set_index("semana")
    log.info("Calendario: %d semanas, %d columnas", len(out), out.shape[1])
    return out


def _resumen_semana(lunes: pd.Timestamp, festivos: set[dt.date], sufijo: str) -> dict:
    """Resume el calendario de la semana que empieza en `lunes`."""
    dias = [(lunes + pd.Timedelta(i, unit="D")).date() for i in range(7)]
    # SIPSA cotiza de lunes a sábado: el domingo no cuenta como día perdido.
    habiles = dias[:6]
    en_festivo = [d for d in habiles if d in festivos]

    return {
        f"festivos{sufijo}": len(en_festivo),
        f"festivo_lunes{sufijo}": int(dias[0] in festivos),
        f"dias_cotizacion{sufijo}": len(habiles) - len(en_festivo),
        f"semana_santa{sufijo}": int(_es_semana_santa(dias, festivos)),
        f"fin_de_anio{sufijo}": int(any(d.month == 12 and d.day >= 24 for d in dias)),
        f"quincena{sufijo}": int(any(d.day == 15 for d in dias)),
    }


def _es_semana_santa(dias, festivos: set[dt.date]) -> bool:
    """La semana contiene Jueves y Viernes Santo.

    Se detecta por la firma de dos festivos consecutivos en jueves y viernes,
    que en el calendario colombiano solo produce la Semana Santa.
    """
    jueves = [d for d in dias if d.weekday() == 3 and d in festivos]
    viernes = [d for d in dias if d.weekday() == 4 and d in festivos]
    return bool(jueves and viernes)


# --- ENSO / ONI -------------------------------------------------------------


def descargar_oni(*, raw_dir: Path = RAW_DIR, force: bool = False) -> pd.DataFrame:
    """Descarga el índice ONI de la NOAA, con caché diario.

    El archivo son ~23 KB de texto de ancho fijo con cabecera
    `SEAS YR TOTAL ANOM`; `ANOM` es el ONI.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    destino = raw_dir / f"oni_{dt.date.today().isoformat()}.csv"
    if destino.exists() and not force:
        log.info("Reutilizando ONI cacheado: %s", destino.name)
        return pd.read_csv(destino, parse_dates=["mes"])

    log.info("Descargando ONI de la NOAA")
    resp = requests.get(URL_ONI, timeout=120)
    resp.raise_for_status()
    df = _parsear_oni(resp.text)
    df.to_csv(destino, index=False)
    log.info("ONI: %d meses, %s -> %s", len(df), df["mes"].min().date(), df["mes"].max().date())
    return df


def _parsear_oni(texto: str) -> pd.DataFrame:
    """Convierte el texto de la NOAA en (mes, oni).

    A cada trimestre móvil se le asigna su **mes central**: 'DJF' de 1950 es
    enero de 1950, 'JFM' es febrero, y así.
    """
    orden = {s: i + 1 for i, s in enumerate(_TEMPORADAS)}
    filas = []
    for linea in texto.splitlines()[1:]:
        partes = linea.split()
        if len(partes) != 4 or partes[0] not in orden:
            continue
        filas.append((dt.date(int(partes[1]), orden[partes[0]], 1), float(partes[3])))
    if not filas:
        raise RuntimeError("El archivo del ONI no trajo ninguna fila legible")
    df = pd.DataFrame(filas, columns=["mes", "oni"])
    df["mes"] = pd.to_datetime(df["mes"])
    return df.sort_values("mes").reset_index(drop=True)


def features_enso(
    semanas: pd.Series, oni: pd.DataFrame | None = None, *, rezago_meses: int = REZAGO_ONI_MESES
) -> pd.DataFrame:
    """Features de ENSO por semana, con el rezago de publicación aplicado.

    Args:
        semanas: lunes de cada semana.
        oni: tabla (mes, oni); si se omite se descarga.
        rezago_meses: meses que se retrocede para no usar valores aún no
            publicados por la NOAA.

    Returns:
        Una fila por semana, indexada por `semana`.
    """
    oni = descargar_oni() if oni is None else oni
    serie = oni.set_index("mes")["oni"].sort_index()

    semanas = pd.to_datetime(pd.Series(sorted(set(pd.to_datetime(semanas))), name="semana"))
    filas = []
    for lunes in semanas:
        # Mes del que SÍ se disponía en esa fecha.
        mes = (lunes.to_period("M") - rezago_meses).to_timestamp()
        valor = _valor_en(serie, mes)
        hace_3 = _valor_en(serie, (lunes.to_period("M") - rezago_meses - 3).to_timestamp())
        hace_6 = _valor_en(serie, (lunes.to_period("M") - rezago_meses - 6).to_timestamp())
        filas.append(
            {
                "semana": lunes,
                "oni": valor,
                "oni_tendencia_3m": valor - hace_3 if _ambos(valor, hace_3) else np.nan,
                "oni_tendencia_6m": valor - hace_6 if _ambos(valor, hace_6) else np.nan,
                "enso_nino": int(valor >= UMBRAL_FASE) if pd.notna(valor) else np.nan,
                "enso_nina": int(valor <= -UMBRAL_FASE) if pd.notna(valor) else np.nan,
            }
        )
    out = pd.DataFrame(filas).set_index("semana")
    log.info(
        "ENSO: %d semanas, ONI de %.2f a %.2f",
        len(out),
        out["oni"].min(),
        out["oni"].max(),
    )
    return out


def _valor_en(serie: pd.Series, mes: pd.Timestamp) -> float:
    return float(serie.get(mes, np.nan))


def _ambos(a: float, b: float) -> bool:
    return bool(pd.notna(a) and pd.notna(b))
