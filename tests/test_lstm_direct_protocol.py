import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest


def runner():
    assert importlib.util.find_spec('scripts.benchmark_lstm_direct_crossyear'), 'runner missing'
    return importlib.import_module('scripts.benchmark_lstm_direct_crossyear')


def test_fixed_scope():
    module = runner()
    assert module.SEEDS == (24, 42, 2026)
    assert module.ARMS == ('gru_history', 'lstm_history', 'lstm_repeat')
    assert module.YEARS == (2024, 2025)


def test_first_best_epoch_wins_ties():
    assert runner().choose_epoch([
        {'epoch': 1, 'selection_block_mae': 12.},
        {'epoch': 2, 'selection_block_mae': 10.},
        {'epoch': 3, 'selection_block_mae': 10.},
    ]) == 2


@pytest.mark.parametrize('scores', [[], [np.nan], [np.inf]])
def test_reject_invalid_development_scores(scores):
    with pytest.raises(ValueError):
        runner().choose_epoch([{'epoch': i + 1, 'selection_block_mae': v}
                               for i, v in enumerate(scores)])


def test_normalization_does_not_use_evaluation_values():
    history = np.full((3, 24, 12), 2.)
    history[2] = 999.
    snapshot = np.array([[4., np.nan], [4., np.nan], [999., 99.]])
    h, x, scaler = runner().normalize_arrays(history, snapshot, np.array([True, True, False]))
    np.testing.assert_array_equal(scaler['history_mean'], np.full(12, 2.))
    np.testing.assert_array_equal(scaler['snapshot_mean'], [4., 0.])
    assert h.shape == (3, 24, 24) and x.shape == (3, 4)
    assert x[-1, 1] == 0.


def test_repeat_control_uses_only_last_frame():
    h = np.tile(np.arange(24.)[None, :, None], (2, 1, 12))
    normalized, _, scaler = runner().normalize_arrays(h, np.ones((2, 1)),
                                                     np.array([True, True]), repeat=True)
    np.testing.assert_array_equal(scaler['history_mean'], np.full(12, 23.))
    np.testing.assert_array_equal(normalized[:, :, :12], np.zeros((2, 24, 12)))


def test_temporal_split_purges_crossing_process():
    t = pd.to_datetime(['2023-01-01', '2023-12-01', '2024-02-01']).tz_localize('Asia/Shanghai')
    frame = pd.DataFrame({'timestamp': t, 'label_available_at': t + pd.Timedelta(hours=10),
                          'continuity_block': ['safe', 'crossing', 'crossing']})
    fitting, selection = runner().split_members(frame, 2025)
    np.testing.assert_array_equal(fitting, [True, False, False])
    np.testing.assert_array_equal(selection, [False, False, True])


def test_unmatured_training_label_is_rejected():
    frame = pd.DataFrame({
        'timestamp': [pd.Timestamp('2024-12-31', tz='Asia/Shanghai')],
        'label_available_at': [pd.Timestamp('2025-01-02', tz='Asia/Shanghai')],
        'continuity_block': ['x'],
    })
    with pytest.raises(ValueError):
        runner().split_members(frame, 2025)


def fake_runs(module, values):
    return {f'{arm}_seed{seed}': {'scores': {k: value for k in module.METRICS}}
            for arm in module.ARMS for seed, value in zip(module.SEEDS, values)}


def test_summary_reports_all_seeds_not_just_best():
    module = runner()
    summary = module.summarize(fake_runs(module, [10., 20., 30.]))
    assert summary['lstm_history']['exact_row_mae'] == {'mean': 20., 'min': 10., 'max': 30.}


def test_single_good_seed_does_not_pass_gate():
    module = runner()
    runs = fake_runs(module, [1., 21., 21.])
    native = {k: 20. for k in module.METRICS}
    assert not module.exact_gate(native, runs, module.summarize(runs))


def test_worse_long_duration_error_does_not_pass_gate():
    module = runner()
    runs = fake_runs(module, [10., 11., 12.])
    for seed in module.SEEDS:
        runs[f'lstm_history_seed{seed}']['scores']['tail24_block_mae'] = 25.
    native = {k: 20. for k in module.METRICS}
    assert not module.exact_gate(native, runs, module.summarize(runs))
