import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_exporter():
    path = Path(__file__).resolve().parents[1] / 'scripts/export_rebound_cases.py'
    spec = importlib.util.spec_from_file_location('export_rebound_cases', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_six_hour_recovery_requires_finite_contiguous_hours():
    mod = load_exporter()
    assert mod.analyze([99] * 6)['recovery_start'] == 1
    assert mod.analyze([99] * 5)['recovery_start'] is None
    assert mod.analyze([99, 99, np.nan, 99, 99, 99])['recovery_start'] is None
    assert mod.analyze([99] * 6, [1, 2, 3, 5, 6, 7])['recovery_start'] is None


def test_short_low_run_rebound_is_not_missing_or_late_recovery():
    mod = load_exporter()
    assert mod.analyze([110, 99, 98, 110])['short_rebounds'] == [{'start': 2, 'end': 3, 'rebound_hour': 4}]
    assert mod.analyze([99, 99, np.nan, 110])['short_rebounds'] == []
    assert mod.analyze([99] * 6 + [110])['short_rebounds'] == []
    assert mod.analyze([99, 99, 110], [1, 2, 4])['short_rebounds'] == []


def test_category_is_based_on_separate_prediction_and_actual():
    mod = load_exporter()
    assert mod.category([99, 110], [98, 120]) == 'rebound'
    assert mod.category([99] * 6, [99] * 6) == 'sustained_recovery'
    assert mod.category([99] * 6, [99, 110]) == 'missed_rebound'
    assert mod.category([110] * 6, [99] * 6) is None


def test_export_matches_independent_raw_rows_and_frozen_predictions():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / 'project-showcase/rebound-cases.json').read_text('utf-8'))
    exp = root / 'experiments/continuous_trajectory_20260929'
    curves = exp / 'full_curve_runs/20260930T033524196780Z'
    source = root / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
    raw = pd.read_parquet(source)
    stamps = pd.DatetimeIndex(raw.timestamp)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == payload['source_hashes']['city_hourly.parquet']
    assert payload['version'] == 1
    assert set(c['category'] for c in payload['cases']) == {'rebound', 'sustained_recovery', 'missed_rebound'}
    for case in payload['cases']:
        index = stamps.get_loc(pd.Timestamp(case['timestamp']))
        year = case['test_year']
        origins = pd.read_parquet(curves / f'{year}_origins.parquet')
        positions = np.flatnonzero((origins.source_index == index) & (origins.partition == 'test'))
        assert len(positions) == 1
        forecasts = np.load(curves / f'{year}_lag1_chemistry.npy')[positions[0]]
        assert case['train_end_year'] + 1 == case['dev_year'] == year - 1
        for name, offsets in [('history', range(-24, 0)), ('actual', range(1, 25))]:
            assert len(case[name]) == 24
            for record, offset in zip(case[name], offsets):
                assert pd.Timestamp(record['timestamp']) == stamps[index + offset]
                for feature in ('aqi', 'pm2_5', 'pm10'):
                    assert record[feature] == float(raw.iloc[index + offset][feature])
        for hour, record in enumerate(case['forecast'], 1):
            assert record['hour'] == hour
            assert pd.Timestamp(record['timestamp']) == stamps[index + hour]
            assert record['aqi'] == float(forecasts[hour - 1])
            assert record['buffered_aqi'] == record['aqi'] + case['buffer_aqi']
        actual = np.array([record['aqi'] for record in case['actual']])
        pred_recovers = any(np.all(forecasts[start:start + 6] <= 100) for start in range(19))
        actual_recovers = any(np.all(actual[start:start + 6] <= 100) for start in range(19))
        if case['category'] == 'rebound':
            assert not pred_recovers and not actual_recovers
        elif case['category'] == 'sustained_recovery':
            assert pred_recovers and actual_recovers
        else:
            assert pred_recovers and not actual_recovers
