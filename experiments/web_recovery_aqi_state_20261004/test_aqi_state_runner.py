import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('state_test_main', Path(__file__).with_name('aqi_state_runner.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def grid(values):
    return pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=len(values), freq='h', tz='Asia/Shanghai'),
                         'aqi': pd.Series(values, dtype=object)})


def test_observed_runs_and_start_flag_inherit_without_unknown_bridging():
    frame = grid([120., 150., 100., 110., 140., np.nan, 166., 170., 50., 120.])
    actual = runner.observed_aqi_state(frame)
    np.testing.assert_array_equal(actual[runner.STATE_FIELDS[0]], [1., 2., 0., 1., 2., np.nan, 1., 2., 0., 1.])
    np.testing.assert_array_equal(actual[runner.STATE_FIELDS[1]], [0., 0., np.nan, 1., 1., np.nan, 0., 0., np.nan, 1.])
    assert list(actual) == ['timestamp', *runner.STATE_FIELDS]


@pytest.mark.parametrize('clean_hours', [1, 5, 6])
def test_every_actual_clean_hour_breaks_high_observations(clean_hours):
    actual = runner.observed_aqi_state(grid([120., 140., *([90.] * clean_hours), 120., 160.]))
    assert actual.iloc[-2][runner.STATE_FIELDS[0]] == 1
    assert actual.iloc[-1][runner.STATE_FIELDS[0]] == 2
    assert actual.iloc[-1][runner.STATE_FIELDS[1]] == 1


def test_particle_and_future_label_columns_have_no_effect():
    frame = grid([90., 166., 168., np.nan, 130.])
    expected = runner.observed_aqi_state(frame)
    frame['pm2_5_iaqi'] = np.nan
    frame['pm10_iaqi'] = np.nan
    frame['remaining_hours'] = [400., 300., 200., 100., 1.]
    frame['event_id'] = 999
    pd.testing.assert_frame_equal(runner.observed_aqi_state(frame), expected)


def test_prefixes_and_future_perturbations_are_causal():
    frame = grid([120., 120., 70., 130., 180., np.nan, 145., 140., 150.])
    expected = runner.observed_aqi_state(frame)
    for stop in range(1, len(frame) + 1):
        pd.testing.assert_frame_equal(runner.observed_aqi_state(frame.iloc[:stop]), expected.iloc[:stop])
        changed = frame.copy()
        changed.loc[stop:, 'aqi'] = 20.
        pd.testing.assert_frame_equal(runner.observed_aqi_state(changed).iloc[:stop], expected.iloc[:stop])


def test_none_and_nullable_missing_preserved_as_unknown():
    frame = grid([120., None, pd.NA, 150.])
    np.testing.assert_array_equal(runner.observed_aqi_state(frame)[runner.STATE_FIELDS[0]], [1., np.nan, np.nan, 1.])


@pytest.mark.parametrize('invalid', [True, False, np.bool_(True), 1+0j, np.complex128(2), np.inf, -np.inf, '120', object()])
def test_invalid_aqi_rejected_before_float_conversion(invalid):
    with pytest.raises(ValueError, match='AQI'):
        runner.observed_aqi_state(grid([120., invalid]))


@pytest.mark.parametrize('kind', ['empty', 'naive', 'missing', 'duplicate', 'unsorted', 'gap', 'fractional'])
def test_invalid_time_grid_rejected(kind):
    frame = grid([90., 120., 140.])
    if kind == 'empty':
        frame = frame.iloc[:0]
    elif kind == 'naive':
        frame['timestamp'] = frame.timestamp.dt.tz_localize(None)
    elif kind == 'missing':
        frame.loc[1, 'timestamp'] = pd.NaT
    elif kind == 'duplicate':
        frame.loc[1, 'timestamp'] = frame.timestamp.iloc[0]
    elif kind == 'unsorted':
        frame = frame.iloc[::-1]
    elif kind == 'gap':
        frame = frame.iloc[[0, 2]]
    else:
        frame['timestamp'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match='hour|timestamp|grid'):
        runner.observed_aqi_state(frame)


def test_100_is_clean_and_long_run_is_not_capped():
    actual = runner.observed_aqi_state(grid([100., *([100.1] * 101)]))
    assert actual.iloc[-1][runner.STATE_FIELDS[0]] == 101.
    assert actual.iloc[-1][runner.STATE_FIELDS[1]] == 1.


def test_attach_uses_full_grid_and_preserves_99_columns_and_same_mask():
    frame = grid([90., 110., 120., 130., np.nan, 150.])
    state = runner.observed_aqi_state(frame)
    names = [f'x{i}' for i in range(99)]
    base = pd.DataFrame(np.arange(198).reshape(2, 99).astype(float), columns=names)
    base['timestamp'] = frame.timestamp.iloc[[3, 5]].to_numpy()
    base.loc[0, 'x0'] = np.nan
    augmented, matrices, combined = runner.attach_state(base, names, state)
    assert combined == [*names, *runner.STATE_FIELDS]
    np.testing.assert_array_equal(matrices['state_values'][names], base[names])
    np.testing.assert_array_equal(matrices['state_control'][names], base[names])
    np.testing.assert_array_equal(matrices['state_control'].isna(), matrices['state_values'].isna())
    np.testing.assert_array_equal(augmented[runner.STATE_FIELDS[0]], [3., 1.])
    np.testing.assert_array_equal(augmented[runner.STATE_FIELDS[1]], [1., 0.])
    assert matrices['state_control'][runner.STATE_FIELDS].eq(0).all().all()


def test_nonapplicable_start_flag_and_missing_run_stay_nan_in_control():
    frame = grid([90., np.nan, 160.])
    names = [f'x{i}' for i in range(99)]
    base = pd.DataFrame(np.zeros((3, 99)), columns=names)
    base['timestamp'] = frame.timestamp
    _, matrices, _ = runner.attach_state(base, names, runner.observed_aqi_state(frame))
    np.testing.assert_array_equal(matrices['state_control'][runner.STATE_FIELDS[0]], [0., np.nan, 0.])
    np.testing.assert_array_equal(matrices['state_control'][runner.STATE_FIELDS[1]], [np.nan, np.nan, 0.])


def test_attach_rejects_absent_or_duplicate_origins_and_wrong_base_schema():
    frame = grid([90., 120.])
    names = [f'x{i}' for i in range(99)]
    base = pd.DataFrame(np.zeros((2, 99)), columns=names)
    base['timestamp'] = frame.timestamp
    state = runner.observed_aqi_state(frame)
    with pytest.raises(ValueError, match='99'):
        runner.attach_state(base, names[:-1], state)
    with pytest.raises(ValueError, match='origins'):
        runner.attach_state(base.assign(timestamp=base.timestamp.iloc[0]), names, state)
    base.loc[1, 'timestamp'] += pd.Timedelta(days=100)
    with pytest.raises(ValueError, match='origins'):
        runner.attach_state(base, names, state)


def gate_metrics():
    score = {'prediction_coverage': 1., 'exact_rows': 1, 'exact_blocks': 1, 'tail24_rows': 1,
             'tail24_blocks': 1, 'exact_row_mae': 7., 'exact_block_mae': 7., 'tail24_block_mae': 7.}
    arms = {arm: score.copy() for arm in runner.ARMS}
    arms['archive']['exact_row_mae'] = 10.
    paired = {f'{candidate}_vs_{baseline}': {'all_row_delta_upper': 0., 'all_block_delta_upper': 0.}
              for baseline, candidate in runner.PAIRS}
    return {'arms': arms, 'paired': paired}


def test_primary_and_equal_dimension_information_gates_are_both_required():
    metrics = gate_metrics()
    gate = runner.main_gates(metrics)
    assert gate['primary']['passed']
    assert not gate['state_information']['passed']
    metrics['arms']['state_values']['exact_row_mae'] = 6.9
    assert runner.main_gates(metrics)['passed']
    metrics['arms']['state_values']['tail24_block_mae'] = 7.1
    assert not runner.main_gates(metrics)['passed']


def test_primary_coverage_includes_inherited_weather_reference():
    metrics = gate_metrics()
    metrics['arms']['state_values']['exact_row_mae'] = 6.9
    metrics['arms']['weather_values']['prediction_coverage'] = .99
    assert not runner.main_gates(metrics)['primary']['checks']['coverage_100']


def synthetic_population():
    cutoff = pd.Timestamp('2024-02-01', tz='UTC')
    names = [f'x{i}' for i in range(101)]
    population = pd.DataFrame(np.ones((3, 101)), columns=names)
    population['timestamp'] = pd.date_range(cutoff-pd.Timedelta(hours=4), periods=3, freq='h')
    population['known_at'] = population.timestamp
    population['population_row'] = [0, 1, 2]
    population['continuity_block'] = 1
    population['legacy_remaining_hours'] = 68.
    population['lower_bound_hours'] = [1., 2., 3.]
    population['upper_bound_hours'] = [1., 4., np.inf]
    population['exact_hours'] = [1., np.nan, np.nan]
    population['label_available_at'] = [cutoff-pd.Timedelta(hours=2), cutoff-pd.Timedelta(hours=1), cutoff]
    population['informative'] = True
    population['reason'] = 'synthetic'
    return population, names, cutoff


def test_maturity_is_strict_and_matches_exact_parent_row_inventory():
    population, _, cutoff = synthetic_population()
    members = runner.mature_members(population, cutoff, 3, 'state_values', 'm', [0, 1])
    assert members.population_row.tolist() == [0, 1]
    assert members.upper_bound_hours.tolist() == [1., 4.]
    with pytest.raises(ValueError, match='members'):
        runner.mature_members(population, cutoff, 3, 'state_values', 'm', [0, 1, 2])
    population.loc[0, 'known_at'] = cutoff
    with pytest.raises(ValueError, match='known_at'):
        runner.mature_members(population, cutoff, 3, 'state_values', 'm', [0, 1])


def test_fit_bounds_fixed_parameters_parent_members_and_native_mock_replay(monkeypatch, tmp_path):
    population, names, cutoff = synthetic_population()
    calls = []

    class Booster:
        feature_names = names

        def __init__(self, **kwargs):
            pass

        def save_config(self):
            return '{}'

        def num_boosted_rounds(self):
            return 500

        def save_model(self, path):
            path.write_text('{}')

        def load_model(self, path):
            assert path.exists()

        def predict(self, matrix):
            return np.full(len(matrix), 90.)

    class Model:
        def __init__(self, **kwargs):
            assert kwargs == {'seed': 42, 'num_boost_round': 500, 'nthread': 1}
            self.booster = Booster()

        def fit_bounds(self, features, lower, upper):
            assert list(features) == names
            np.testing.assert_array_equal(lower, [1., 2.])
            np.testing.assert_array_equal(upper, [1., 4.])
            calls.append(len(features))

        def predict(self, frame):
            return self.booster.predict(frame)

    monkeypatch.setattr(runner.bridge, 'ObservedAFTSurvivalRegressor', Model)
    monkeypatch.setattr(runner.bridge.xgb, 'Booster', Booster)
    monkeypatch.setattr(runner.bridge.xgb, 'DMatrix', lambda frame, **kwargs: frame)
    (tmp_path / 'models').mkdir()
    prediction, members, record = runner.fit_block(population, names, population[names], 3, cutoff,
        'state_values', tmp_path, [0, 1])
    assert calls == [2]
    np.testing.assert_array_equal(prediction, [90., 90., 90.])
    assert members.population_row.tolist() == [0, 1]
    assert record['parent_member_model_id'] == 'weather_values_b3'
    assert record['num_boosted_rounds'] == 500
    with pytest.raises(ValueError, match='approved'):
        runner.fit_block(population, names, population[names], 3, cutoff, 'weather_values', tmp_path, [0, 1])


def test_scoring_preserves_uncapped_full_duration_and_interval_rows():
    population, _, _ = synthetic_population()
    population['lower_bound_hours'] = [100., 2., 200.]
    population['upper_bound_hours'] = [100., 4., 200.]
    population['exact_hours'] = [100., np.nan, 200.]
    predictions = {arm: np.array([0., 3., 0.]) for arm in runner.ARMS}
    rows = runner.row_scores(population, predictions)
    assert len(rows) == 3
    assert rows['state_values_exact_error'].iloc[0] == 100.
    assert rows['state_values_exact_error'].iloc[2] == 200.
    scores = runner.score_cohort(population, predictions)
    assert scores['arms']['state_values']['exact_row_mae'] == 150.
    assert scores['arms']['state_values']['subgroups']['finite_interval']['rows'] == 1


@pytest.mark.parametrize('mutator', ['under20', 'paired_row', 'paired_block', 'missing', 'empty_tail'])
def test_no_gate_can_hide_small_gain_uncertainty_missing_or_empty_tail(mutator):
    metrics = gate_metrics()
    metrics['arms']['state_values']['exact_row_mae'] = 6.9
    if mutator == 'under20':
        metrics['arms']['state_values']['exact_row_mae'] = 8.1
        metrics['arms']['state_control']['exact_row_mae'] = 9.
    elif mutator in ('paired_row', 'paired_block'):
        field = 'all_row_delta_upper' if mutator == 'paired_row' else 'all_block_delta_upper'
        metrics['paired']['state_values_vs_archive'][field] = .1
    elif mutator == 'missing':
        metrics['arms']['state_values']['prediction_coverage'] = .99
    else:
        metrics['arms']['state_values']['tail24_rows'] = 0
        metrics['arms']['state_values']['tail24_blocks'] = 0
        metrics['arms']['state_values']['tail24_block_mae'] = None
    assert not runner.main_gates(metrics)['passed']


def test_model_inventory_exact_64_unique():
    models = [{'model_id': f'{arm}_b{block}', 'arm': arm, 'continuity_block': block}
              for arm in runner.NEW_ARMS for block in range(32)]
    runner.validate_model_inventory(models, range(32))
    with pytest.raises(ValueError):
        runner.validate_model_inventory(models[:-1] + models[:1], range(32))


def test_parent_status_and_digest_fail_closed():
    with pytest.raises(ValueError):
        runner.validate_parent({'status': 'pending_independent_verification'}, {}, 'bad')


def test_counts_copy_instead_of_duplicate_kwargs(monkeypatch):
    labels = {'exact': 7494, 'interval': 1144, 'right_censored': 0}
    parent = {'counts': {'planned_fits': 64, 'new_training_population': 8638,
                         'new_training_population_label_counts': labels}}
    monkeypatch.setattr(runner.bridge, 'label_counts', lambda frame: labels)
    inputs = {'populations': {'state_control': pd.DataFrame({'x': np.arange(8638)})}}
    assert runner.cohort_counts(inputs, parent) == parent['counts']


def test_protected_exact_inventory_rejects_added_file_or_empty_manifest(tmp_path):
    folders = ('artifacts/cnemc_2025_deployment_candidate', 'artifacts/strict_history_backtest',
               'project-showcase', 'sites-haze/dist', 'releases/runtime_20261002')
    hashes = {}
    for name in folders:
        folder = tmp_path / name
        folder.mkdir(parents=True)
        path = folder / 'locked.txt'
        path.write_text('unchanged')
        hashes[str(path.resolve())] = runner.bridge.sha(path)
    runner.proxy.validate_protected_inventory(tmp_path, hashes)
    (tmp_path / folders[-1] / 'extra.txt').write_text('new')
    with pytest.raises(ValueError, match='protected'):
        runner.proxy.validate_protected_inventory(tmp_path, hashes)
    with pytest.raises(ValueError, match='protected'):
        runner.proxy.validate_protected_inventory(tmp_path, {})
