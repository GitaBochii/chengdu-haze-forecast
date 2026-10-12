import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sequence_clock_adapter import sequence_feature_row

from haze_forecast.features import MEASUREMENT_COLUMNS, add_missingness_indicators
from haze_forecast.observed_hourly_features import build_observed_hourly_features


ROOT = Path(__file__).resolve().parents[2]
FEATURES = json.loads((ROOT / 'artifacts/cnemc_2025_deployment_candidate/model_metadata.json').read_text(encoding='utf8'))['feature_names']


def measurements(count=25):
    times = pd.date_range('2025-01-03', periods=count, freq='h', tz='Asia/Shanghai')
    frame = pd.DataFrame({'timestamp': times})
    for offset, name in enumerate(MEASUREMENT_COLUMNS):
        frame[name] = np.arange(count, dtype=float) + offset + 100
    return frame


def canonical_row(frame):
    full = build_observed_hourly_features(frame)
    pbl = [name[:-11] for name in FEATURES if name.endswith('_is_missing') and '_change_' not in name]
    full = add_missingness_indicators(full, pbl)
    full['boundary_layer_height_m_change_3_is_missing'] = (
        full.boundary_layer_height_m.isna() | full.boundary_layer_height_m_lag_3.isna()
    ).astype(float)
    return full[FEATURES].tail(1).reset_index(drop=True).astype(float)


def test_25_hours_match_training_and_use_true_lag24():
    frame = measurements()
    actual = sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)
    pd.testing.assert_frame_equal(actual, canonical_row(frame))
    assert actual.aqi_lag_24.iloc[0] == frame.aqi.iloc[0]


def test_24_consecutive_rows_are_not_a_full_24_hour_history():
    frame = measurements(24)
    with pytest.raises(ValueError, match='24-hour'):
        sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)


def test_gap_is_unknown_not_the_previous_24th_observed_row():
    frame = measurements(26).drop(index=1).reset_index(drop=True)
    actual = sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)
    assert np.isnan(actual.aqi_lag_24.iloc[0])
    assert actual.aqi_lag_3.iloc[0] == frame.aqi.iloc[-4]


def test_interior_gap_remains_missing_and_is_not_filled():
    frame = measurements().drop(index=21).reset_index(drop=True)
    actual = sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)
    assert np.isnan(actual.aqi_lag_3.iloc[0])
    expected = np.mean([122.0, 123.0])
    assert actual.aqi_roll_3.iloc[0] == expected


def test_future_input_rejected():
    frame = measurements(26)
    with pytest.raises(ValueError, match='future'):
        sequence_feature_row(frame, frame.timestamp.iloc[-2], FEATURES)


def test_timezone_normalized_before_cyclic_features():
    frame = measurements()
    local = sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)
    utc = frame.copy()
    utc['timestamp'] = utc.timestamp.dt.tz_convert('UTC')
    actual = sequence_feature_row(utc, utc.timestamp.iloc[-1], FEATURES)
    pd.testing.assert_frame_equal(actual, local)


@pytest.mark.parametrize('problem', ['duplicate', 'off_hour', 'naive', 'infinity'])
def test_invalid_measurement_contract_rejected(problem):
    frame = measurements()
    if problem == 'duplicate':
        frame = pd.concat([frame, frame.tail(1)], ignore_index=True)
    elif problem == 'off_hour':
        frame.loc[0, 'timestamp'] += pd.Timedelta(minutes=1)
    elif problem == 'naive':
        frame['timestamp'] = frame.timestamp.dt.tz_localize(None)
    else:
        frame.loc[0, 'aqi'] = np.inf
    with pytest.raises(ValueError):
        sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)


def test_non_measurement_columns_do_not_leak():
    frame = measurements()
    base = sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)
    frame['remaining_hours'] = 999
    frame['exact_hours'] = 888
    frame['particle_primary'] = False
    frame['aqi_lag_24'] = -1000
    actual = sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)
    pd.testing.assert_frame_equal(actual, base)


def test_optional_weather_missing_remains_nan():
    frame = measurements().drop(columns=['boundary_layer_height_m'])
    actual = sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES)
    assert np.isnan(actual.boundary_layer_height_m.iloc[0])
    assert actual.boundary_layer_height_m_is_missing.iloc[0] == 1


def test_origin_must_be_latest_observed_timestamp():
    frame = measurements()
    with pytest.raises(ValueError, match='origin'):
        sequence_feature_row(frame, frame.timestamp.iloc[-1] + pd.Timedelta(hours=1), FEATURES)


def test_feature_order_must_be_unique_and_available():
    frame = measurements()
    with pytest.raises(ValueError):
        sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES + [FEATURES[0]])
    with pytest.raises(ValueError):
        sequence_feature_row(frame, frame.timestamp.iloc[-1], FEATURES[:-1] + ['future_feature'])


def test_all_701_web_origins_match_training_clock_features():
    full = pd.read_parquet(ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet').set_index('timestamp')
    origins = pd.read_parquet(ROOT / 'artifacts/strict_history_backtest/predictions.parquet')
    origins = origins.loc[origins.timestamp.dt.year.isin([2024, 2025])]
    assert len(origins) == 701
    for timestamp in origins.timestamp:
        window = full.loc[timestamp - pd.Timedelta(hours=24):timestamp, list(MEASUREMENT_COLUMNS)].reset_index()
        actual = sequence_feature_row(window, timestamp, FEATURES).iloc[0].to_numpy(float)
        expected = full.loc[timestamp, FEATURES].to_numpy(float)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True,
                                   err_msg=str(timestamp))
