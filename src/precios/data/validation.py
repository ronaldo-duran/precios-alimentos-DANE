"""Validación de datos: detiene el pipeline ante datos que no son de fiar.

Se usan checks propios en vez de pandera a propósito: las reglas son pocas y
muy específicas del dominio, y cada fallo debe explicar *qué* filas lo causaron
para poder auditarlo. Un `DataFrame` de ejemplos vale más que un booleano.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from precios.config import ReglasLimpieza
from precios.data.sources import COLUMNAS_CANONICAS

log = logging.getLogger(__name__)

#: Columnas sin las cuales una fila no significa nada.
CLAVES_DIARIAS: tuple[str, ...] = ("fecha", "producto_sipsa", "plaza_sipsa", "precio_kg")
CLAVES_SEMANALES: tuple[str, ...] = ("semana", "producto_id", "plaza_id", "precio_kg")


class ValidationError(AssertionError):
    """Una o más reglas de validación fallaron; el pipeline no debe continuar."""

    def __init__(self, fallos: list[str]) -> None:
        self.fallos = fallos
        cuerpo = "\n".join(f"  - {f}" for f in fallos)
        super().__init__(f"Validación fallida ({len(fallos)} regla(s)):\n{cuerpo}")


def _muestra(df: pd.DataFrame, n: int = 3) -> str:
    """Representación corta de filas infractoras, para el mensaje de error."""
    cols = [c for c in CLAVES_DIARIAS if c in df.columns] or list(df.columns[:4])
    return df[cols].head(n).to_dict("records").__repr__()


def validar_diario(df: pd.DataFrame, reglas: ReglasLimpieza) -> None:
    """Valida el esquema diario canónico recién ingerido.

    Raises:
        ValidationError: si alguna regla falla.
    """
    fallos: list[str] = []

    faltantes = [c for c in COLUMNAS_CANONICAS if c not in df.columns]
    if faltantes:
        # Sin columnas no tiene sentido seguir evaluando el resto.
        raise ValidationError([f"Esquema inesperado: faltan columnas {faltantes}"])

    if df.empty:
        raise ValidationError(["El conjunto diario llegó vacío (0 filas)"])

    for col in CLAVES_DIARIAS:
        n_nulos = int(df[col].isna().sum())
        if n_nulos:
            fallos.append(f"{n_nulos} nulos en columna clave {col!r}")

    if not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        fallos.append(f"'fecha' debería ser datetime, es {df['fecha'].dtype}")

    if not pd.api.types.is_numeric_dtype(df["precio_kg"]):
        fallos.append(f"'precio_kg' debería ser numérica, es {df['precio_kg'].dtype}")
    else:
        fuera = df[(df["precio_kg"] < reglas.precio_min) | (df["precio_kg"] > reglas.precio_max)]
        if not fuera.empty:
            fallos.append(
                f"{len(fuera)} precios fuera del rango "
                f"[{reglas.precio_min:g}, {reglas.precio_max:g}] COP/kg: {_muestra(fuera)}"
            )
        no_positivos = df[df["precio_kg"] <= 0]
        if not no_positivos.empty:
            fallos.append(f"{len(no_positivos)} precios <= 0: {_muestra(no_positivos)}")

    dups = df[df.duplicated(subset=["fecha", "producto_sipsa", "plaza_sipsa"], keep=False)]
    if not dups.empty:
        fallos.append(
            f"{len(dups)} filas duplicadas por (fecha, producto, plaza): {_muestra(dups)}"
        )

    if pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        manana = pd.Timestamp(dt.date.today()) + pd.Timedelta(1, unit="D")
        futuras = df[df["fecha"] >= manana]
        if not futuras.empty:
            fallos.append(f"{len(futuras)} filas con fecha futura: {_muestra(futuras)}")

    if fallos:
        raise ValidationError(fallos)
    log.info(
        "Validación diaria OK: %d filas, %s -> %s",
        len(df),
        df["fecha"].min().date(),
        df["fecha"].max().date(),
    )


def validar_semanal(df: pd.DataFrame, reglas: ReglasLimpieza) -> None:
    """Valida el panel semanal consolidado, listo para `data/processed/`."""
    fallos: list[str] = []

    faltantes = [c for c in CLAVES_SEMANALES if c not in df.columns]
    if faltantes:
        raise ValidationError([f"Esquema semanal inesperado: faltan columnas {faltantes}"])

    if df.empty:
        raise ValidationError(["El panel semanal quedó vacío"])

    dups = df[df.duplicated(subset=["semana", "producto_id", "plaza_id"], keep=False)]
    if not dups.empty:
        fallos.append(f"{len(dups)} duplicados por (semana, producto, plaza)")

    # La etiqueta de cada semana debe ser su lunes.
    no_lunes = df[pd.to_datetime(df["semana"]).dt.weekday != 0]
    if not no_lunes.empty:
        fallos.append(f"{len(no_lunes)} filas cuya 'semana' no cae en lunes")

    # Orden temporal dentro de cada serie.
    desordenadas = (
        df.groupby(["producto_id", "plaza_id"], observed=True)["semana"]
        .apply(lambda s: not s.is_monotonic_increasing)
        .pipe(lambda s: s[s].index.tolist())
    )
    if desordenadas:
        fallos.append(f"{len(desordenadas)} series con fechas fuera de orden: {desordenadas[:3]}")

    if fallos:
        raise ValidationError(fallos)
    log.info(
        "Validación semanal OK: %d filas, %d series, %s -> %s",
        len(df),
        df.groupby(["producto_id", "plaza_id"], observed=True).ngroups,
        df["semana"].min().date(),
        df["semana"].max().date(),
    )


def reporte_series(df: pd.DataFrame, reglas: ReglasLimpieza) -> pd.DataFrame:
    """Resume cada serie y marca cuáles son aptas para modelar.

    No lanza excepción: la decisión de excluir una serie es una regla de negocio
    explícita (`config/cleaning.yaml`), no un error de datos.

    Returns:
        Un `DataFrame` por serie con cobertura, faltantes y el motivo de exclusión.
    """
    filas = []
    for (prod, plaza), g in df.groupby(["producto_id", "plaza_id"], observed=True):
        semanas = pd.to_datetime(g["semana"]).sort_values()
        n_obs = len(semanas)
        span = int((semanas.max() - semanas.min()).days // 7) + 1
        pct_falt = 100.0 * (1 - n_obs / span) if span else 100.0

        motivos = []
        if pct_falt > reglas.max_pct_faltantes:
            motivos.append(f"faltantes {pct_falt:.1f}% > {reglas.max_pct_faltantes:g}%")
        if n_obs < reglas.min_semanas:
            motivos.append(f"historia {n_obs} sem < {reglas.min_semanas}")

        filas.append(
            {
                "producto_id": prod,
                "plaza_id": plaza,
                "n_semanas": n_obs,
                "span_semanas": span,
                "pct_faltantes": round(pct_falt, 2),
                "inicio": semanas.min().date(),
                "fin": semanas.max().date(),
                "precio_kg_mediano": round(float(g["precio_kg"].median()), 1),
                "cv_pct": round(100 * float(g["precio_kg"].std() / g["precio_kg"].mean()), 1),
                "n_outliers": int(g["flag_outlier"].sum()) if "flag_outlier" in g else 0,
                "apta": not motivos,
                "motivo_exclusion": "; ".join(motivos),
            }
        )
    rep = pd.DataFrame(filas).sort_values(["producto_id", "plaza_id"]).reset_index(drop=True)
    n_no_aptas = int((~rep["apta"]).sum())
    if n_no_aptas:
        log.warning("%d de %d series NO son aptas para modelar", n_no_aptas, len(rep))
    return rep
