"""Registro de modelos: reproducibilidad y trazabilidad."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from precios.models import registry


@pytest.fixture
def panel() -> pd.DataFrame:
    semanas = pd.date_range("2024-01-01", periods=20, freq="W-MON")
    filas = []
    for prod in ("papa", "tomate"):
        filas.append(
            pd.DataFrame(
                {
                    "semana": semanas,
                    "producto_id": prod,
                    "plaza_id": "bogota",
                    "precio_kg": range(1000, 1020),
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


def _metadata(version: str = "2026-01-01_abcd1234") -> registry.Metadata:
    return registry.Metadata(
        version=version,
        entrenado_en=dt.datetime(2026, 1, 1, 12, 0).isoformat(),
        semana_inicio="2024-01-01",
        semana_fin="2024-05-13",
        n_series=2,
        n_observaciones=40,
        productos=["papa", "tomate"],
        plazas=["bogota"],
        horizontes=[1, 2, 3, 4],
        nivel_intervalo=0.8,
        semilla=42,
        params_lgbm={"num_leaves": 15},
        commit="deadbeef",
    )


def test_el_hash_es_estable_ante_las_mismas_entradas(panel):
    a = registry.calcular_hash(panel, {"num_leaves": 15}, 42, "abc")
    b = registry.calcular_hash(panel, {"num_leaves": 15}, 42, "abc")
    assert a == b
    assert len(a) == 8


def test_el_hash_cambia_si_cambian_los_hiperparametros(panel):
    a = registry.calcular_hash(panel, {"num_leaves": 15}, 42, "abc")
    b = registry.calcular_hash(panel, {"num_leaves": 31}, 42, "abc")
    assert a != b


def test_el_hash_cambia_si_cambia_la_semilla(panel):
    a = registry.calcular_hash(panel, {}, 42, "abc")
    b = registry.calcular_hash(panel, {}, 7, "abc")
    assert a != b


def test_el_hash_cambia_si_cambia_el_codigo(panel):
    a = registry.calcular_hash(panel, {}, 42, "commit1")
    b = registry.calcular_hash(panel, {}, 42, "commit2")
    assert a != b


def test_el_hash_cambia_si_cambian_los_datos(panel):
    a = registry.calcular_hash(panel, {}, 42, "abc")
    b = registry.calcular_hash(panel.iloc[:-1], {}, 42, "abc")
    assert a != b


def test_el_hash_no_depende_del_orden_de_los_hiperparametros(panel):
    a = registry.calcular_hash(panel, {"a": 1, "b": 2}, 42, "abc")
    b = registry.calcular_hash(panel, {"b": 2, "a": 1}, 42, "abc")
    assert a == b


def test_guardar_y_cargar_devuelve_el_mismo_modelo(tmp_path):
    modelo = {"parametros": [1, 2, 3]}
    meta = _metadata()
    registry.guardar(modelo, meta, models_dir=tmp_path)

    cargado, meta_cargada = registry.cargar(models_dir=tmp_path)
    assert cargado == modelo
    assert meta_cargada.version == meta.version
    assert meta_cargada.semilla == 42
    assert meta_cargada.commit == "deadbeef"


def test_guardar_actualiza_el_puntero_latest(tmp_path):
    registry.guardar({"a": 1}, _metadata("2026-01-01_aaaaaaaa"), models_dir=tmp_path)
    assert registry.version_latest(models_dir=tmp_path) == "2026-01-01_aaaaaaaa"

    registry.guardar({"a": 2}, _metadata("2026-02-01_bbbbbbbb"), models_dir=tmp_path)
    assert registry.version_latest(models_dir=tmp_path) == "2026-02-01_bbbbbbbb"


def test_se_puede_cargar_una_version_antigua_explicitamente(tmp_path):
    registry.guardar({"a": 1}, _metadata("2026-01-01_aaaaaaaa"), models_dir=tmp_path)
    registry.guardar({"a": 2}, _metadata("2026-02-01_bbbbbbbb"), models_dir=tmp_path)

    viejo, _ = registry.cargar("2026-01-01_aaaaaaaa", models_dir=tmp_path)
    assert viejo == {"a": 1}


def test_la_metadata_se_guarda_como_json_legible(tmp_path):
    destino = registry.guardar({"a": 1}, _metadata(), models_dir=tmp_path)
    crudo = json.loads((destino / registry.NOMBRE_METADATA).read_text(encoding="utf-8"))
    assert crudo["n_series"] == 2
    assert crudo["productos"] == ["papa", "tomate"]
    assert crudo["horizontes"] == [1, 2, 3, 4]


def test_guardar_acepta_tablas_extra(tmp_path):
    extra = pd.DataFrame({"h": [1, 2], "radio_log": [0.1, 0.2]})
    destino = registry.guardar(
        {"a": 1}, _metadata(), models_dir=tmp_path, extras={"diagnostico": extra}
    )
    assert (destino / "diagnostico.csv").exists()


def test_cargar_sin_modelos_registrados_falla_con_instruccion(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        registry.cargar(models_dir=tmp_path)


def test_listar_ordena_de_mas_reciente_a_mas_antigua(tmp_path):
    registry.guardar({"a": 1}, _metadata("2026-01-01_aaaaaaaa"), models_dir=tmp_path)
    meta2 = _metadata("2026-02-01_bbbbbbbb")
    meta2.entrenado_en = dt.datetime(2026, 2, 1, 12, 0).isoformat()
    registry.guardar({"a": 2}, meta2, models_dir=tmp_path)

    tabla = registry.listar(models_dir=tmp_path)
    assert list(tabla["version"]) == ["2026-02-01_bbbbbbbb", "2026-01-01_aaaaaaaa"]
    assert tabla.loc[0, "es_latest"]
    assert not tabla.loc[1, "es_latest"]


def test_listar_sin_modelos_devuelve_tabla_vacia_con_columnas(tmp_path):
    tabla = registry.listar(models_dir=tmp_path)
    assert tabla.empty
    assert "version" in tabla.columns


def test_los_artefactos_se_fechan_en_hora_de_colombia():
    """El runner de CI corre en UTC: un pipeline lanzado un lunes por la tarde
    en Bogotá produciría artefactos fechados el martes."""
    import datetime as dt

    from precios.config import ZONA, ahora, hoy

    assert ZONA.utcoffset(None) == dt.timedelta(hours=-5)
    assert hoy() == dt.datetime.now(ZONA).date()
    assert ahora().tzinfo is not None
    assert ahora().microsecond == 0


def test_la_fecha_de_colombia_puede_diferir_de_utc():
    """Entre las 19:00 y la medianoche de Bogotá, UTC ya es el día siguiente."""
    import datetime as dt

    from precios.config import ZONA

    noche = dt.datetime(2026, 9, 30, 20, 0, tzinfo=ZONA)
    assert noche.date() == dt.date(2026, 9, 30)
    assert noche.astimezone(dt.UTC).date() == dt.date(2026, 10, 1)
