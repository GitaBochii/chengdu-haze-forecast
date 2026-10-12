import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('proxy_test_module', Path(__file__).with_name('proxy_runner.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    times = pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC')
    names = [f'x{i}' for i in range(87)]
    base = pd.DataFrame(np.ones((3, 87)), columns=names)
    base['timestamp'] = times
    base.loc[0, 'x0'] = np.nan
    snap = pd.DataFrame({'known_at': [times[1]], 'run_at': [times[0]],
                         'valid_at': [times[0] + pd.Timedelta(hours=24)],
                         **{name: [i + 1.] for i, name in enumerate(runner.WEATHER_FIELDS)}})
    return base, names, snap


def test_matching_schema_masks_and_zero_placeholder():
    base, names, snap = fixture()
    result, matrices, combined = runner.attach_weather(base, names, snap)
    assert len(combined) == 99
    assert combined[:87] == names
    assert result.known_at.iloc[0] is pd.NaT
    np.testing.assert_array_equal(matrices['metadata_control'].isna(), matrices['weather_values'].isna())
    np.testing.assert_array_equal(matrices['weather_values'][names], base[names])
    assert matrices['metadata_control'].loc[1:, runner.WEATHER_FIELDS].eq(0).all().all()
    assert matrices['weather_values'].loc[1:, runner.WEATHER_FIELDS].gt(0).all().all()
    assert matrices['weather_values'].loc[0, runner.WEATHER_FIELDS].isna().all()
    np.testing.assert_array_equal(matrices['metadata_control'][runner.AGE_FIELDS], matrices['weather_values'][runner.AGE_FIELDS])


def test_future_known_never_selected():
    base, names, snap = fixture()
    snap['known_at'] += pd.Timedelta(days=2)
    snap['run_at'] += pd.Timedelta(days=2)
    snap['valid_at'] += pd.Timedelta(days=2)
    result, matrices, _ = runner.attach_weather(base, names, snap)
    assert result.known_at.isna().all()
    assert matrices['weather_values'][runner.WEATHER_FIELDS].isna().all().all()


def test_schema_rejects_reordered_snapshot_fields():
    base, names, snap = fixture()
    snap = snap[list(reversed(snap.columns))]
    with pytest.raises(ValueError):
        runner.attach_weather(base, names, snap)


def test_non_f024_rejected():
    base, names, snap = fixture()
    snap['valid_at'] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match='f024'):
        runner.attach_weather(base, names, snap)


def test_weather_information_gate_requires_strict_gain():
    score = {'prediction_coverage': 1., 'exact_rows': 1, 'exact_blocks': 1,
             'tail24_rows': 1, 'tail24_blocks': 1, 'exact_row_mae': 7.,
             'exact_block_mae': 7., 'tail24_block_mae': 7.}
    arms = {arm: score.copy() for arm in runner.ARMS}
    arms['archive']['exact_row_mae'] = 10.
    metrics = {'arms': arms, 'paired': {f'{candidate}_vs_{baseline}':
        {'all_row_delta_upper': 0., 'all_block_delta_upper': 0.}
        for baseline, candidate in runner.PAIRS}}
    gate = runner.main_gates(metrics)
    assert gate['primary']['passed']
    assert not gate['weather_information']['passed']
    assert not gate['passed']
    arms['weather_values']['exact_row_mae'] = 6.9
    assert runner.main_gates(metrics)['passed']
    arms['weather_values']['tail24_block_mae'] = 7.1
    assert not runner.main_gates(metrics)['passed']


def test_wrong_parent_rejected():
    with pytest.raises(ValueError):
        runner.validate_parent({'status': 'pending_independent_verification'}, {}, 'bad')


def test_inventory_requires_exact_64_unique_models():
    records = [{'arm': arm, 'continuity_block': block, 'model_id': f'{arm}_b{block}'}
               for arm in runner.NEW_ARMS for block in range(32)]
    runner.validate_model_inventory(records, range(32))
    with pytest.raises(ValueError):
        runner.validate_model_inventory(records[:-1], range(32))
    with pytest.raises(ValueError):
        runner.validate_model_inventory(records[:-1] + records[:1], range(32))


def test_mature_members_keep_exact_parent_population(monkeypatch):
    cutoff = pd.Timestamp('2024-01-02', tz='UTC')
    population = pd.DataFrame({'population_row': [1, 2], 'timestamp': [cutoff-pd.Timedelta(hours=3)]*2,
        'continuity_block': [1, 1], 'legacy_remaining_hours': [2., 2.],
        'known_at': [cutoff-pd.Timedelta(hours=2), pd.NaT]})
    for name in runner.bridge.BOUND_COLUMNS:
        population[name] = 1
    monkeypatch.setattr(runner.bridge, 'mature_mask', lambda *args: np.array([True, True]))
    assert len(runner.mature_members(population, cutoff, 3, 'weather_values', 'm', [1, 2])) == 2
    with pytest.raises(ValueError):
        runner.mature_members(population, cutoff, 3, 'weather_values', 'm', [1])
    population.loc[0, 'known_at'] = cutoff
    with pytest.raises(ValueError):
        runner.mature_members(population, cutoff, 3, 'weather_values', 'm', [1, 2])


def test_fit_block_uses_fixed_parameters_bounds_and_replays(monkeypatch, tmp_path):
    base, names, snap = fixture()
    population, features, names = runner.attach_weather(base, names, snap)
    cutoff = pd.Timestamp('2024-02-01', tz='UTC')
    population['population_row'] = [0, 1, 2]
    population['continuity_block'] = 1
    population['legacy_remaining_hours'] = 3.
    population['lower_bound_hours'] = [1., 2., 3.]
    population['upper_bound_hours'] = [1., 4., np.inf]
    population['exact_hours'] = [1., np.nan, np.nan]
    population['label_available_at'] = population.timestamp
    for name in runner.bridge.BOUND_COLUMNS:
        if name not in population:
            population[name] = 'synthetic'
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
            return np.full(len(matrix), 2.)

    class Model:
        def __init__(self, **kwargs):
            assert kwargs == runner.bridge.MODEL_CONFIG
            self.booster = Booster()

        def fit_bounds(self, x, lower, upper):
            assert list(x) == names
            np.testing.assert_array_equal(lower, population.lower_bound_hours)
            np.testing.assert_array_equal(upper, population.upper_bound_hours)
            calls.append(len(x))

        def predict(self, x):
            return self.booster.predict(x)

    monkeypatch.setattr(runner.bridge, 'ObservedAFTSurvivalRegressor', Model)
    monkeypatch.setattr(runner.bridge, 'mature_mask', lambda *args: np.ones(3, dtype=bool))
    monkeypatch.setattr(runner.bridge, 'label_counts', lambda frame: {'exact': 1, 'interval': 1, 'right_censored': 1})
    monkeypatch.setattr(runner.bridge.xgb, 'Booster', Booster)
    monkeypatch.setattr(runner.bridge.xgb, 'DMatrix', lambda frame, **kwargs: frame)
    (tmp_path / 'models').mkdir()
    values, members, record = runner.fit_block(population, names, features['weather_values'],
        3, cutoff, 'weather_values', tmp_path, [0, 1, 2])
    assert calls == [3]
    assert len(members) == len(values) == 3
    assert record['parent_member_model_id'] == 'grid_all_legacy_origins_b3'
    assert record['num_boosted_rounds'] == 500


def test_protected_inventory_rejects_extra_file(tmp_path):
    folders = ('artifacts/cnemc_2025_deployment_candidate', 'artifacts/strict_history_backtest',
               'project-showcase', 'sites-haze/dist', 'releases/runtime_20261002')
    hashes = {}
    for name in folders:
        folder = tmp_path / name
        folder.mkdir(parents=True)
        path = folder / 'locked.txt'
        path.write_text('unchanged')
        hashes[str(path.resolve())] = runner.bridge.sha(path)
    runner.validate_protected_inventory(tmp_path, hashes)
    (tmp_path / folders[-1] / 'extra.txt').write_text('untracked')
    with pytest.raises(ValueError, match='protected'):
        runner.validate_protected_inventory(tmp_path, hashes)


def test_protected_inventory_rejects_empty_manifest(tmp_path):
    with pytest.raises(ValueError, match='protected'):
        runner.validate_protected_inventory(tmp_path, {})


def test_counts_accept_parent_existing_population_keys(monkeypatch):
    labels = {'exact': 2, 'interval': 1, 'right_censored': 0}
    parent = {'counts': {'planned_fits': 64, 'new_training_population': 3,
                         'new_training_population_label_counts': labels}}
    monkeypatch.setattr(runner.bridge, 'label_counts', lambda frame: labels)
    inputs = {'populations': {'metadata_control': pd.DataFrame({'x': [1, 2, 3]})}}
    assert runner.cohort_counts(inputs, parent) == parent['counts']


def test_snapshot_report_fixed_hash(monkeypatch):
    monkeypatch.setattr(runner.bridge, 'sha', lambda path: 'wrong')
    with pytest.raises(ValueError, match='Snapshot source report hash'):
        runner.validate_snapshot_report(Path('unused.json'))
