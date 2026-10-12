import importlib
import importlib.util

import numpy as np
import pytest


def runner():
    assert importlib.util.find_spec('scripts.tune_lstm_learning_rate'), 'tuning runner missing'
    return importlib.import_module('scripts.tune_lstm_learning_rate')


def candidates(values):
    return [{'learning_rate': rate, 'seed': seed,
             'scores': {'exact_block_mae': score}, 'selected_epoch': 1}
            for rate, scores in values.items() for seed, score in zip((24, 42, 2026), scores)]


def test_select_by_three_seed_mean_not_best_single_run():
    data = candidates({.0003: [1., 30., 30.], .001: [15., 15., 15.], .003: [20., 20., 20.]})
    assert runner().select_learning_rate(data)['selected_learning_rate'] == .001


def test_identical_scores_keep_original_rate():
    data = candidates({.0003: [10., 10., 10.], .001: [10., 10., 10.], .003: [10., 10., 10.]})
    assert runner().select_learning_rate(data)['selected_learning_rate'] == .001


def test_final_scores_cannot_choose_rate():
    data = candidates({.0003: [10., 10., 10.], .001: [12., 12., 12.], .003: [15., 15., 15.]})
    for row in data:
        row['evaluation_mae'] = 1. if row['learning_rate'] == .003 else 100.
    assert runner().select_learning_rate(data)['selected_learning_rate'] == .0003


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'nan', 'unexpected'])
def test_invalid_candidate_population_is_rejected(fault):
    data = candidates({.0003: [10., 10., 10.], .001: [12., 12., 12.], .003: [15., 15., 15.]})
    if fault == 'missing':
        data.pop()
    elif fault == 'duplicate':
        data[-1] = data[0].copy()
    elif fault == 'nan':
        data[-1]['scores']['exact_block_mae'] = np.nan
    else:
        data[-1]['learning_rate'] = .1
    with pytest.raises(ValueError):
        runner().select_learning_rate(data)


def test_search_scope_is_fixed():
    module = runner()
    assert module.LEARNING_RATES == (.0003, .001, .003)
    assert module.SEEDS == (24, 42, 2026)
    assert module.YEARS == (2024, 2025)
