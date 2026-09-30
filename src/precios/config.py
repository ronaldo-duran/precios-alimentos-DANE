"""Carga de la configuración versionada en `config/`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INCOMING_DIR = DATA_DIR / "incoming"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class Producto:
    id: str
    sipsa: str
    etiqueta: str


@dataclass(frozen=True)
class Plaza:
    id: str
    sipsa: str
    ciudad: str
    etiqueta: str


@dataclass(frozen=True)
class Exclusion:
    producto: str
    plaza: str
    motivo: str


@dataclass(frozen=True)
class Scope:
    """Productos, plazas y exclusiones que definen el alcance del pipeline."""

    productos: tuple[Producto, ...]
    plazas: tuple[Plaza, ...]
    exclusiones: tuple[Exclusion, ...] = field(default=())

    @property
    def nombres_sipsa_producto(self) -> set[str]:
        return {p.sipsa for p in self.productos}

    @property
    def nombres_sipsa_plaza(self) -> set[str]:
        return {p.sipsa for p in self.plazas}

    def producto_por_sipsa(self, nombre: str) -> Producto | None:
        return next((p for p in self.productos if p.sipsa == nombre), None)

    def plaza_por_sipsa(self, nombre: str) -> Plaza | None:
        return next((p for p in self.plazas if p.sipsa == nombre), None)

    @property
    def pares_excluidos(self) -> set[tuple[str, str]]:
        """Pares (producto_id, plaza_id) excluidos explícitamente."""
        return {(e.producto, e.plaza) for e in self.exclusiones}

    @property
    def n_series_esperadas(self) -> int:
        return len(self.productos) * len(self.plazas) - len(self.exclusiones)


@dataclass(frozen=True)
class Normalizacion:
    """Diccionario de alias historico -> nombre canónico."""

    plazas: dict[str, str]
    productos: dict[str, str]


@dataclass(frozen=True)
class ReglasLimpieza:
    """Umbrales explícitos de agregación, límites, outliers y series."""

    agregacion: dict[str, Any]
    limites: dict[str, Any]
    outliers: dict[str, Any]
    series: dict[str, Any]

    @property
    def precio_min(self) -> float:
        return float(self.limites["precio_kg_min"])

    @property
    def precio_max(self) -> float:
        return float(self.limites["precio_kg_max"])

    @property
    def max_pct_faltantes(self) -> float:
        return float(self.series["max_pct_faltantes"])

    @property
    def min_semanas(self) -> int:
        return int(self.series["min_semanas"])


@dataclass(frozen=True)
class ConfigEvaluacion:
    """Parámetros de la validación walk-forward."""

    min_train: int
    horizonte_max: int
    paso: int
    max_origenes: int | None
    semilla: int
    solo_series_aptas: bool

    @property
    def horizontes(self) -> tuple[int, ...]:
        return tuple(range(1, self.horizonte_max + 1))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_scope(path: Path | None = None) -> Scope:
    """Lee `config/products.yaml`."""
    raw = _read_yaml(path or CONFIG_DIR / "products.yaml")
    scope = Scope(
        productos=tuple(Producto(**p) for p in raw.get("productos", [])),
        plazas=tuple(Plaza(**p) for p in raw.get("plazas", [])),
        exclusiones=tuple(Exclusion(**e) for e in raw.get("exclusiones", [])),
    )
    _validar_scope(scope)
    log.info(
        "Alcance: %d productos x %d plazas - %d exclusiones = %d series",
        len(scope.productos),
        len(scope.plazas),
        len(scope.exclusiones),
        scope.n_series_esperadas,
    )
    return scope


def _validar_scope(scope: Scope) -> None:
    """Falla temprano ante un `products.yaml` incoherente."""
    ids_prod = [p.id for p in scope.productos]
    ids_plaza = [p.id for p in scope.plazas]
    for nombre, ids in (("producto", ids_prod), ("plaza", ids_plaza)):
        dups = {i for i in ids if ids.count(i) > 1}
        if dups:
            raise ValueError(f"IDs de {nombre} duplicados en products.yaml: {sorted(dups)}")
    for exc in scope.exclusiones:
        if exc.producto not in ids_prod:
            raise ValueError(f"Exclusión con producto desconocido: {exc.producto!r}")
        if exc.plaza not in ids_plaza:
            raise ValueError(f"Exclusión con plaza desconocida: {exc.plaza!r}")


@lru_cache(maxsize=1)
def load_normalizacion(path: Path | None = None) -> Normalizacion:
    """Lee `config/normalization.yaml`."""
    raw = _read_yaml(path or CONFIG_DIR / "normalization.yaml")
    return Normalizacion(
        plazas=dict(raw.get("plazas") or {}),
        productos=dict(raw.get("productos") or {}),
    )


def load_lgbm_params(path: Path | None = None) -> dict[str, Any]:
    """Lee la configuración de LightGBM elegida por `make tune`, si existe.

    Devuelve `{}` cuando no se ha corrido la búsqueda, y entonces LightGBM usa
    sus parámetros por defecto. El pipeline nunca falla por esto.
    """
    ruta = path or CONFIG_DIR / "lgbm_params.yaml"
    if not ruta.exists():
        log.info("Sin %s: LightGBM usará sus parámetros por defecto", ruta.name)
        return {}
    params = dict(_read_yaml(ruta).get("params", {}))
    log.info("Hiperparámetros de la búsqueda: %s", params)
    return params


@lru_cache(maxsize=1)
def load_evaluacion(path: Path | None = None) -> ConfigEvaluacion:
    """Lee `config/evaluation.yaml`."""
    raw = _read_yaml(path or CONFIG_DIR / "evaluation.yaml")
    wf = raw.get("walk_forward", {})
    cfg = ConfigEvaluacion(
        min_train=int(wf.get("min_train", 104)),
        horizonte_max=int(wf.get("horizonte_max", 4)),
        paso=int(wf.get("paso", 4)),
        max_origenes=wf.get("max_origenes"),
        semilla=int(raw.get("semilla", 42)),
        solo_series_aptas=bool(raw.get("solo_series_aptas", True)),
    )
    log.info(
        "Walk-forward: min_train=%d, horizontes 1-%d, paso=%d, semilla=%d",
        cfg.min_train,
        cfg.horizonte_max,
        cfg.paso,
        cfg.semilla,
    )
    return cfg


@lru_cache(maxsize=1)
def load_reglas(path: Path | None = None) -> ReglasLimpieza:
    """Lee `config/cleaning.yaml`."""
    raw = _read_yaml(path or CONFIG_DIR / "cleaning.yaml")
    return ReglasLimpieza(
        agregacion=raw.get("agregacion", {}),
        limites=raw.get("limites", {}),
        outliers=raw.get("outliers", {}),
        series=raw.get("series", {}),
    )
