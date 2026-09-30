"""Construcción de los folds walk-forward.

Si los folds están mal, todas las métricas del proyecto mienten.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from precios.evaluation.splits import Fold, rolling_origins


def test_ventana_expansiva_crece_en_cada_fold():
    folds = rolling_origins(200, min_train=100, horizonte_max=4, paso=4)
    origenes = [f.origen for f in folds]
    assert origenes == sorted(origenes)
    assert all(b > a for a, b in pairwise(origenes))


def test_el_primer_origen_respeta_min_train():
    folds = rolling_origins(200, min_train=104, horizonte_max=4, paso=4)
    assert folds[0].origen == 104


def test_el_paso_separa_los_origenes():
    folds = rolling_origins(200, min_train=100, horizonte_max=4, paso=7)
    origenes = [f.origen for f in folds]
    assert all(b - a == 7 for a, b in pairwise(origenes))


def test_el_ultimo_fold_cabe_en_la_serie():
    n = 200
    folds = rolling_origins(n, min_train=100, horizonte_max=4, paso=4)
    assert max(max(f.indices_test) for f in folds) <= n - 1


def test_train_y_test_nunca_se_solapan():
    folds = rolling_origins(200, min_train=100, horizonte_max=4, paso=4)
    for f in folds:
        indices_train = set(range(f.origen))
        assert indices_train.isdisjoint(set(f.indices_test))


def test_los_indices_de_test_van_despues_del_origen():
    folds = rolling_origins(120, min_train=100, horizonte_max=4, paso=4)
    for f in folds:
        assert min(f.indices_test) == f.origen
        assert list(f.indices_test) == sorted(f.indices_test)


def test_horizonte_h_apunta_h_semanas_despues_del_origen():
    fold = Fold(origen=50, horizontes=(1, 2, 3, 4))
    assert fold.indices_test == (50, 51, 52, 53)


def test_serie_demasiado_corta_no_produce_folds():
    assert rolling_origins(50, min_train=104, horizonte_max=4, paso=4) == []


def test_serie_justo_en_el_limite_produce_un_fold():
    folds = rolling_origins(108, min_train=104, horizonte_max=4, paso=4)
    assert len(folds) == 1
    assert folds[0].origen == 104


def test_max_origenes_conserva_los_mas_recientes():
    todos = rolling_origins(300, min_train=100, horizonte_max=4, paso=4)
    limitados = rolling_origins(300, min_train=100, horizonte_max=4, paso=4, max_origenes=5)
    assert len(limitados) == 5
    assert [f.origen for f in limitados] == [f.origen for f in todos[-5:]]


@pytest.mark.parametrize(
    ("kwargs", "mensaje"),
    [
        ({"min_train": 0}, "min_train"),
        ({"horizonte_max": 0}, "horizonte_max"),
        ({"paso": 0}, "paso"),
    ],
)
def test_rechaza_parametros_invalidos(kwargs, mensaje):
    base = {"min_train": 10, "horizonte_max": 4, "paso": 4}
    base.update(kwargs)
    with pytest.raises(ValueError, match=mensaje):
        rolling_origins(200, **base)
