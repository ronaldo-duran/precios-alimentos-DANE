"""Normalización de nombres y recorte al alcance configurado.

Los nombres de plazas y productos en SIPSA cambian con los años (renombres,
variantes de espaciado). El diccionario vive en `config/normalization.yaml`,
versionado, no en el código.
"""

from __future__ import annotations

import logging

import pandas as pd

from precios.cleaning.nombres import clave
from precios.config import Normalizacion, Scope

log = logging.getLogger(__name__)


def normalizar_nombres(df: pd.DataFrame, norm: Normalizacion) -> pd.DataFrame:
    """Aplica el diccionario de alias y limpia espacios sobrantes.

    Se hace ANTES de filtrar por alcance: si no, una serie renombrada se
    perdería por no coincidir con el nombre canónico de `products.yaml`.
    """
    df = df.copy()
    for col, alias in (("plaza_sipsa", norm.plazas), ("producto_sipsa", norm.productos)):
        antes = df[col].astype("string").str.strip()
        df[col] = antes.replace(alias)
        n_cambios = int((antes != df[col]).sum())
        if n_cambios:
            log.info("Normalización de %s: %d filas reasignadas", col, n_cambios)
    return df


def canonizar_contra_alcance(df: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    """Reconcilia grafías distintas del mismo nombre contra el alcance.

    Los nombres de `config/products.yaml` son la forma canónica. Cualquier
    variante que solo difiera en mayúsculas, tildes, puntuación o espaciado
    tiene la misma clave y se reasigna a la canónica.

    No es hipotético: IDEAM cambió `ANTIOQUIA` por `Antioquia` de un día para
    otro, y SIPSA escribe `Piña *` y `Piña*` indistintamente. Sin esto, un
    cambio de grafía en el origen hace desaparecer la serie en silencio, que es
    la peor forma de fallar.
    """
    df = df.copy()
    for col, nombres in (
        ("producto_sipsa", scope.nombres_sipsa_producto),
        ("plaza_sipsa", scope.nombres_sipsa_plaza),
    ):
        indice = {clave(n): n for n in nombres}
        presentes = set(df[col].dropna().unique())
        reasignar = {
            v: indice[clave(v)]
            for v in presentes
            if v not in nombres and clave(v) in indice
        }
        if reasignar:
            n_filas = int(df[col].isin(reasignar).sum())
            log.warning(
                "%s: %d filas reasignadas por grafía (%s)",
                col,
                n_filas,
                {k: v for k, v in list(reasignar.items())[:3]},
            )
            df[col] = df[col].replace(reasignar)
    return df


def colapsar_duplicados_diarios(df: pd.DataFrame) -> pd.DataFrame:
    """Resuelve los duplicados que puede crear un renombre de plaza.

    Los datos crudos no traen duplicados por (fecha, producto, plaza), pero un
    renombre sí puede crearlos: `Cali, Santa Helena` y `Cali, Santa Elena`
    coexisten entre 2022-01-04 y 2022-02-08 durante la transición. Al unificar
    los nombres, esas fechas quedan con dos cotizaciones del mismo mercado.
    Se colapsan con la mediana y se deja constancia en el log.
    """
    llaves = ["fecha", "producto_sipsa", "plaza_sipsa"]
    n_dup = int(df.duplicated(subset=llaves, keep=False).sum())
    if not n_dup:
        return df

    log.warning(
        "%d filas duplicadas tras normalizar nombres (solapamiento de renombre); "
        "se colapsan con la mediana",
        n_dup,
    )
    numericas = ["precio_kg", "precio_kg_min", "precio_kg_max"]
    otras = [c for c in df.columns if c not in llaves + numericas]
    agg = {c: "median" for c in numericas if c in df.columns}
    agg.update({c: "first" for c in otras})
    return df.groupby(llaves, as_index=False, observed=True).agg(agg)[list(df.columns)]


def filtrar_alcance(df: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    """Recorta a los productos y plazas del alcance y añade los IDs estables.

    Descarta también los pares (producto, plaza) excluidos explícitamente en
    `config/products.yaml`, antes de cualquier regla de calidad: una serie
    cerrada no es un hueco que haya que reportar.
    """
    mapa_prod = {p.sipsa: p for p in scope.productos}
    mapa_plaza = {p.sipsa: p for p in scope.plazas}

    faltan_prod = scope.nombres_sipsa_producto - set(df["producto_sipsa"].unique())
    if faltan_prod:
        raise ValueError(
            f"Productos de products.yaml ausentes en los datos: {sorted(faltan_prod)}. "
            "¿Cambió el nombre en SIPSA? Revisa config/normalization.yaml."
        )
    faltan_plaza = scope.nombres_sipsa_plaza - set(df["plaza_sipsa"].unique())
    if faltan_plaza:
        raise ValueError(
            f"Plazas de products.yaml ausentes en los datos: {sorted(faltan_plaza)}. "
            "¿Cambió el nombre en SIPSA? Revisa config/normalization.yaml."
        )

    n_inicial = len(df)
    df = df[
        df["producto_sipsa"].isin(mapa_prod) & df["plaza_sipsa"].isin(mapa_plaza)
    ].copy()

    df["producto_id"] = df["producto_sipsa"].map(lambda s: mapa_prod[s].id)
    df["plaza_id"] = df["plaza_sipsa"].map(lambda s: mapa_plaza[s].id)
    df["producto"] = df["producto_sipsa"].map(lambda s: mapa_prod[s].etiqueta)
    df["plaza"] = df["plaza_sipsa"].map(lambda s: mapa_plaza[s].etiqueta)
    df["ciudad"] = df["plaza_sipsa"].map(lambda s: mapa_plaza[s].ciudad)

    excluidos = scope.pares_excluidos
    if excluidos:
        par = list(zip(df["producto_id"], df["plaza_id"], strict=True))
        mascara = pd.Series([p in excluidos for p in par], index=df.index)
        n_exc = int(mascara.sum())
        df = df[~mascara]
        log.info(
            "Excluidas %d filas de %d serie(s) marcadas en products.yaml", n_exc, len(excluidos)
        )

    log.info(
        "Alcance aplicado: %d -> %d filas, %d series",
        n_inicial,
        len(df),
        df.groupby(["producto_id", "plaza_id"], observed=True).ngroups,
    )
    return df.reset_index(drop=True)
