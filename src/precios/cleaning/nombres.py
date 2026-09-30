"""Canonización de nombres propios colombianos.

El problema, con números reales de este proyecto: IDEAM publica **79 grafías
distintas de departamento** para 33 departamentos, y cambió de `ANTIOQUIA` a
`Antioquia` el 12 de agosto de 2026 sin avisar. SIPSA renombró
`Cali, Santa Helena` a `Cali, Santa Elena`. Y una sola ciudad puede aparecer
como Bogotá, bogota, BOGOTÁ, Bogotá D.C., Bogota DC, Bogotá, D. C.,
Distrito Capital de Bogotá... y así docenas de formas.

Enumerar esas docenas en un diccionario es insostenible. La estrategia aquí es
otra: **reducir cualquier grafía a una clave** quitando tildes, mayúsculas,
puntuación y espacios. Las 36 formas de escribir Bogotá colapsan en apenas
cinco claves, y el diccionario solo tiene que cubrir esas cinco.

    Bogotá, D.C.  ->  bogotadc
    Bogota D. C.  ->  bogotadc
    BOGOTÁ DC     ->  bogotadc

Lo que la clave **no** resuelve son los nombres de verdad distintos —
`San José de Cúcuta` frente a `Cúcuta`, `Santa Helena` frente a `Santa Elena` —
y esos sí van en `config/nombres.yaml`, que es corto y auditable.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from precios.config import CONFIG_DIR

log = logging.getLogger(__name__)


def clave(texto: str) -> str:
    """Reduce un nombre a su clave canónica.

    Quita tildes y diacríticos, pasa a minúsculas y elimina todo lo que no sea
    alfanumérico —incluidos los espacios—. Eso hace equivalentes las
    variaciones de mayúsculas, acentuación, puntuación y separación, que es de
    donde viene la inmensa mayoría del ruido.

    Se eliminan los espacios a propósito: `Bogotá D. C.` y `Bogota DC` tienen
    distinto espaciado y deben dar la misma clave.

        >>> clave("Bogotá, D.C.")
        'bogotadc'
        >>> clave("BOGOTA D. C.")
        'bogotadc'
        >>> clave("San José de Cúcuta")
        'sanjosedecucuta'
    """
    if texto is None:
        return ""
    # NFKD separa la letra de su diacrítico para poder descartar el diacrítico.
    descompuesto = unicodedata.normalize("NFKD", str(texto))
    sin_tildes = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return "".join(c for c in sin_tildes.lower() if c.isalnum())


@dataclass
class Canonizador:
    """Resuelve cualquier grafía de un nombre a su forma canónica.

    Attributes:
        canonicos: forma canónica -> lista de variantes estructuralmente
            distintas. Las variantes de mayúsculas o tildes NO hacen falta
            aquí: las absorbe `clave`.
        etiqueta: qué se está canonizando, solo para los mensajes de error.
    """

    canonicos: dict[str, list[str]]
    etiqueta: str = "nombre"
    _indice: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._indice = {}
        for canonico, variantes in self.canonicos.items():
            for forma in [canonico, *variantes]:
                k = clave(forma)
                if not k:
                    continue
                previo = self._indice.get(k)
                if previo is not None and previo != canonico:
                    raise ValueError(
                        f"La grafía {forma!r} apunta a dos formas canónicas "
                        f"distintas: {previo!r} y {canonico!r}"
                    )
                self._indice[k] = canonico

    def resolver(self, texto: str) -> str | None:
        """Devuelve la forma canónica, o None si el nombre es desconocido."""
        return self._indice.get(clave(texto))

    def resolver_estricto(self, texto: str) -> str:
        """Como `resolver`, pero falla con un mensaje accionable.

        Un nombre desconocido casi siempre significa que la fuente renombró
        algo. Fallar aquí es preferible a perder la serie en silencio.
        """
        canonico = self.resolver(texto)
        if canonico is None:
            raise ValueError(
                f"{self.etiqueta} desconocido: {texto!r} (clave {clave(texto)!r}). "
                f"Si la fuente lo renombró, añade la variante en config/nombres.yaml."
            )
        return canonico

    def conocidos(self) -> set[str]:
        """Formas canónicas disponibles."""
        return set(self.canonicos)

    def desconocidos(self, textos) -> dict[str, str]:
        """Nombres que no se saben resolver, con su clave. Para diagnosticar."""
        return {t: clave(t) for t in dict.fromkeys(textos) if self.resolver(t) is None}


def cargar_canonizadores(path: Path | None = None) -> dict[str, Canonizador]:
    """Lee `config/nombres.yaml` y construye un canonizador por categoría."""
    ruta = path or CONFIG_DIR / "nombres.yaml"
    if not ruta.exists():
        log.warning("Sin %s: no hay canonización de nombres configurada", ruta.name)
        return {}
    with ruta.open(encoding="utf-8") as fh:
        crudo = yaml.safe_load(fh) or {}

    salida = {}
    for categoria, entradas in crudo.items():
        if categoria.startswith("_"):
            continue
        canonicos = {
            canonico: list(variantes or []) for canonico, variantes in (entradas or {}).items()
        }
        salida[categoria] = Canonizador(canonicos=canonicos, etiqueta=categoria[:-1] or categoria)
    log.info(
        "Canonizadores cargados: %s",
        {c: len(v.canonicos) for c, v in salida.items()},
    )
    return salida


def normalizar_serie(valores, canonizador: Canonizador, *, estricto: bool = False):
    """Aplica un canonizador a un iterable, conservando los desconocidos.

    Args:
        valores: nombres crudos.
        canonizador: el que resuelve.
        estricto: si es True, falla ante el primer nombre desconocido.

    Returns:
        Lista con la forma canónica donde se pudo resolver y el valor original
        donde no. Dejar pasar el original —en vez de None— evita convertir un
        nombre nuevo en un hueco silencioso.
    """
    salida = []
    desconocidos: set[str] = set()
    for v in valores:
        canonico = canonizador.resolver(v)
        if canonico is None:
            if estricto:
                canonizador.resolver_estricto(v)
            desconocidos.add(str(v))
            salida.append(v)
        else:
            salida.append(canonico)
    if desconocidos:
        log.warning(
            "%d nombre(s) sin canonizar en %s: %s",
            len(desconocidos),
            canonizador.etiqueta,
            sorted(desconocidos)[:5],
        )
    return salida
