"""Reconciliación del log en vivo contra los precios que realmente ocurrieron."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precios.pipeline.reconcile import _reconciliar_pronosticos, _resumir_desempeno


def _panel() -> pd.DataFrame:
    semanas = pd.date_range("2026-01-05", periods=10, freq="W-MON")
    return pd.DataFrame(
        {
            "semana": semanas,
            "producto_id": "papa",
            "plaza_id": "bogota",
            "precio_kg": [
                1000.0, 1100.0, 1050.0, 1200.0, 1150.0,
                1300.0, 1250.0, 1400.0, 1350.0, 1500.0,
            ],
        }
    )


def _log(**kwargs) -> pd.DataFrame:
    base = {
        "fecha_pronostico": "2026-02-16",
        "version_modelo": "v1",
        "modelo": "lgbm_conformal",
        "producto_id": "papa",
        "plaza_id": "bogota",
        "semana_origen": pd.Timestamp("2026-02-16"),  # índice 6, precio 1250
        "h": 1,
        "semana_objetivo": pd.Timestamp("2026-02-23"),  # índice 7, precio 1400
        "precio_origen": 1250.0,
        "y_pred": 1300.0,
        "lo": 1200.0,
        "hi": 1450.0,
        "nivel_intervalo": 0.8,
        "procedencia": "vivo",
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def test_asigna_el_precio_real_a_un_pronostico_resuelto():
    recon = _reconciliar_pronosticos(_log(), _panel())
    fila = recon.iloc[0]
    assert fila["estado"] == "resuelto"
    assert fila["y_true"] == 1400.0
    assert fila["error"] == pytest.approx(100.0)  # 1400 - 1300
    assert fila["error_abs"] == pytest.approx(100.0)


def test_un_pronostico_al_futuro_queda_pendiente():
    futuro = _log(semana_objetivo=pd.Timestamp("2027-01-04"))
    recon = _reconciliar_pronosticos(futuro, _panel())
    assert recon.iloc[0]["estado"] == "pendiente"
    assert pd.isna(recon.iloc[0]["y_true"])
    assert pd.isna(recon.iloc[0]["error"])


def test_detecta_que_el_valor_real_cayo_dentro_del_intervalo():
    recon = _reconciliar_pronosticos(_log(lo=1200.0, hi=1450.0), _panel())
    assert bool(recon.iloc[0]["dentro_intervalo"])


def test_detecta_que_el_valor_real_quedo_fuera_del_intervalo():
    recon = _reconciliar_pronosticos(_log(lo=1000.0, hi=1100.0), _panel())
    assert not bool(recon.iloc[0]["dentro_intervalo"])


def test_el_intervalo_incluye_sus_bordes():
    recon = _reconciliar_pronosticos(_log(lo=1400.0, hi=1400.0), _panel())
    assert bool(recon.iloc[0]["dentro_intervalo"])


def test_un_pronostico_sin_intervalo_no_cuenta_para_la_cobertura():
    recon = _reconciliar_pronosticos(
        _log(modelo="naive", lo=np.nan, hi=np.nan), _panel()
    )
    assert pd.isna(recon.iloc[0]["dentro_intervalo"])


def test_la_escala_del_mase_se_calcula_con_datos_previos_al_origen():
    recon = _reconciliar_pronosticos(_log(), _panel())
    # Serie hasta el origen (índices 0..6): diferencias absolutas de 100, 50, 150,
    # 50, 150, 50 -> media 91,67.
    assert recon.iloc[0]["escala_mase"] == pytest.approx(91.667, abs=0.01)


def test_el_mase_por_punto_usa_esa_escala():
    recon = _reconciliar_pronosticos(_log(), _panel())
    fila = recon.iloc[0]
    assert fila["mase_punto"] == pytest.approx(fila["error_abs"] / fila["escala_mase"])


def test_el_resumen_separa_vivo_de_backfill():
    """Mezclarlos borraría la única diferencia que importa."""
    log = pd.concat(
        [_log(procedencia="vivo", y_pred=1300.0), _log(procedencia="backfill", y_pred=1000.0)],
        ignore_index=True,
    )
    recon = _reconciliar_pronosticos(log, _panel())
    resumen = _resumir_desempeno(recon)

    assert set(resumen["procedencia"]) == {"vivo", "backfill"}
    vivo = resumen[resumen["procedencia"] == "vivo"].iloc[0]
    backfill = resumen[resumen["procedencia"] == "backfill"].iloc[0]
    assert vivo["mae"] == pytest.approx(100.0)
    assert backfill["mae"] == pytest.approx(400.0)


def test_el_resumen_separa_modelos_y_horizontes():
    log = pd.concat(
        [_log(modelo="lgbm_conformal"), _log(modelo="naive", y_pred=1250.0)],
        ignore_index=True,
    )
    recon = _reconciliar_pronosticos(log, _panel())
    resumen = _resumir_desempeno(recon)
    assert set(resumen["modelo"]) == {"lgbm_conformal", "naive"}
    naive = resumen[resumen["modelo"] == "naive"].iloc[0]
    assert naive["mae"] == pytest.approx(150.0)  # |1400 - 1250|


def test_el_resumen_ignora_los_pendientes():
    log = pd.concat(
        [_log(), _log(h=4, semana_objetivo=pd.Timestamp("2027-01-04"))], ignore_index=True
    )
    recon = _reconciliar_pronosticos(log, _panel())
    resumen = _resumir_desempeno(recon)
    assert resumen["n"].sum() == 1


def test_sin_nada_resuelto_el_resumen_viene_vacio_pero_con_columnas():
    recon = _reconciliar_pronosticos(
        _log(semana_objetivo=pd.Timestamp("2027-01-04")), _panel()
    )
    resumen = _resumir_desempeno(recon)
    assert resumen.empty
    assert "procedencia" in resumen.columns


def test_el_resumen_reporta_el_sesgo_ademas_del_error_absoluto():
    """Un modelo que siempre se queda corto no es lo mismo que uno que oscila."""
    log = pd.concat(
        [_log(y_pred=1300.0), _log(h=2, semana_objetivo=pd.Timestamp("2026-03-02"), y_pred=1250.0)],
        ignore_index=True,
    )
    recon = _reconciliar_pronosticos(log, _panel())
    resumen = _resumir_desempeno(recon)
    # Ambos pronósticos se quedan por debajo del real: el sesgo debe ser positivo.
    assert (resumen["sesgo"] > 0).all()


def test_la_reconciliacion_es_idempotente():
    """Volver a reconciliar el mismo log da exactamente lo mismo."""
    log = _log()
    a = _reconciliar_pronosticos(log, _panel())
    b = _reconciliar_pronosticos(log, _panel())
    pd.testing.assert_frame_equal(a, b)
