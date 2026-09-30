"""Configuración única de logging para scripts y notebooks."""

from __future__ import annotations

import logging
import sys

_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el root logger. Idempotente: no duplica handlers."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMATO, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
