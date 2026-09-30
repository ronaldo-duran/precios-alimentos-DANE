"""El log de pronósticos en vivo: append-only e idempotente.

La propiedad central del proyecto: **un pronóstico registrado nunca cambia**.
Si se pudiera reescribir, el error acumulado dejaría de significar nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.pipeline.forecast import COLUMNAS_LOG, _anexar

LLAVES = (
    "procedencia",
    "version_modelo",
    "modelo",
    "producto_id",
    "plaza_id",
    "h",
    "semana_objetivo",
)


def _fila(**kwargs) -> dict:
    base = {
        "fecha_pronostico": "2026-09-29",
        "version_modelo": "v1",
        "modelo": "lgbm_conformal",
        "producto_id": "papa",
        "plaza_id": "bogota",
        "semana_origen": pd.Timestamp("2026-09-21"),
        "h": 1,
        "semana_objetivo": pd.Timestamp("2026-09-28"),
        "precio_origen": 2000.0,
        "y_pred": 2050.0,
        "lo": 1800.0,
        "hi": 2300.0,
        "nivel_intervalo": 0.8,
        "procedencia": "vivo",
    }
    base.update(kwargs)
    return base


def _tabla(*filas: dict) -> pd.DataFrame:
    return pd.DataFrame(list(filas))[list(COLUMNAS_LOG)]


def test_escribe_el_log_la_primera_vez(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    n = _anexar(_tabla(_fila()), ruta, LLAVES)
    assert n == 1
    assert ruta.exists()
    assert len(pd.read_csv(ruta)) == 1


def test_reescribir_lo_mismo_no_anade_nada(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    _anexar(_tabla(_fila()), ruta, LLAVES)
    n = _anexar(_tabla(_fila()), ruta, LLAVES)
    assert n == 0
    assert len(pd.read_csv(ruta)) == 1


def test_un_horizonte_distinto_si_se_anade(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    _anexar(_tabla(_fila(h=1)), ruta, LLAVES)
    _anexar(
        _tabla(_fila(h=2, semana_objetivo=pd.Timestamp("2026-10-05"))), ruta, LLAVES
    )
    assert len(pd.read_csv(ruta)) == 2


def test_una_version_de_modelo_nueva_se_anade_sin_borrar_la_anterior(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    _anexar(_tabla(_fila(version_modelo="v1", y_pred=2050.0)), ruta, LLAVES)
    _anexar(_tabla(_fila(version_modelo="v2", y_pred=9999.0)), ruta, LLAVES)

    log = pd.read_csv(ruta)
    assert len(log) == 2
    # El pronóstico viejo sigue intacto: nunca se reescribe el pasado.
    viejo = log[log["version_modelo"] == "v1"].iloc[0]
    assert viejo["y_pred"] == 2050.0


def test_el_backfill_y_el_vivo_conviven_sin_pisarse(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    _anexar(_tabla(_fila(procedencia="vivo", y_pred=2050.0)), ruta, LLAVES)
    _anexar(_tabla(_fila(procedencia="backfill", y_pred=1900.0)), ruta, LLAVES)

    log = pd.read_csv(ruta)
    assert len(log) == 2
    assert set(log["procedencia"]) == {"vivo", "backfill"}


def test_anadir_un_lote_mixto_solo_escribe_lo_nuevo(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    _anexar(_tabla(_fila(h=1)), ruta, LLAVES)

    lote = _tabla(
        _fila(h=1),  # ya existe
        _fila(h=2, semana_objetivo=pd.Timestamp("2026-10-05")),  # nueva
        _fila(h=3, semana_objetivo=pd.Timestamp("2026-10-12")),  # nueva
    )
    assert _anexar(lote, ruta, LLAVES) == 2
    assert len(pd.read_csv(ruta)) == 3


def test_las_fechas_se_guardan_como_dia_sin_hora(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    _anexar(_tabla(_fila()), ruta, LLAVES)
    log = pd.read_csv(ruta, dtype=str)
    assert log.loc[0, "semana_objetivo"] == "2026-09-28"
    assert log.loc[0, "semana_origen"] == "2026-09-21"


def test_un_esquema_antiguo_falla_con_instrucciones_en_vez_de_migrar(tmp_path):
    """Migrar en silencio un log append-only sería peor que fallar."""
    ruta = tmp_path / "forecasts_log.csv"
    antiguo = _tabla(_fila()).drop(columns=["procedencia"])
    antiguo.to_csv(ruta, index=False)

    with pytest.raises(ValueError, match="esquema antiguo"):
        _anexar(_tabla(_fila()), ruta, LLAVES)


def test_el_log_conserva_los_intervalos_y_el_nivel(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    _anexar(_tabla(_fila()), ruta, LLAVES)
    log = pd.read_csv(ruta)
    assert log.loc[0, "lo"] == 1800.0
    assert log.loc[0, "hi"] == 2300.0
    assert log.loc[0, "nivel_intervalo"] == 0.8


def test_el_naive_se_registra_sin_intervalo(tmp_path):
    ruta = tmp_path / "forecasts_log.csv"
    _anexar(
        _tabla(_fila(modelo="naive", lo=np.nan, hi=np.nan, nivel_intervalo=np.nan)),
        ruta,
        LLAVES,
    )
    log = pd.read_csv(ruta)
    assert pd.isna(log.loc[0, "lo"])
    assert log.loc[0, "modelo"] == "naive"
