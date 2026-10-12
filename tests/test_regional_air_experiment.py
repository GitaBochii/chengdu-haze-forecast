import numpy as np
import pandas as pd

from haze_forecast.regional_air_experiment import (
    TrainingScaler,
    causal_fill,
    masked_mae,
    prepare_panel,
    recovery_metrics,
    sequence_origins,
    stable_recovery,
)


def test_training_scaler_does_not_fit_test_values():
    values = np.array([[[1.0, 4.0]], [[3.0, 4.0]], [[1000.0, np.nan]]])
    scaler = TrainingScaler.fit(values[:2])
    np.testing.assert_allclose(scaler.mean, [2, 4])
    np.testing.assert_allclose(scaler.scale, [1, 1])
    np.testing.assert_allclose(scaler.transform(values)[2, 0, 0], 998)


def test_causal_fill_never_uses_a_future_observation():
    values = np.array([[[np.nan]], [[2.0]], [[np.nan]], [[100.0]]])
    filled = causal_fill(values, np.array([5.0]), limit=1)
    np.testing.assert_allclose(filled[:, 0, 0], [5, 2, 2, 100])
    assert np.isnan(values[0, 0, 0])


def test_origins_purge_all_targets_at_the_training_boundary():
    times = pd.date_range('2023-12-30', periods=96, freq='h', tz='Asia/Shanghai')
    origins = sequence_origins(times, history=4, horizon=6,
                               start=times[0], end=pd.Timestamp('2024-01-01', tz=times.tz))
    assert len(origins)
    assert (times[origins + 6] < pd.Timestamp('2024-01-01', tz=times.tz)).all()
    assert origins.min() == 3


def test_origins_reject_any_internal_gap():
    times = pd.date_range('2023-01-01', periods=20, freq='h', tz='Asia/Shanghai').delete(10)
    origins = sequence_origins(times, history=4, horizon=6, start=times[0], end=times[-1])
    assert all(not (i - 3 < 10 <= i + 6) for i in origins)


def test_recovery_is_start_and_no_crossing_is_missing_not_horizon():
    assert stable_recovery(np.array([110] + [80] * 6)) == 2
    assert np.isnan(stable_recovery(np.full(72, 110.0)))
    assert np.isnan(stable_recovery(np.array([80, 80, np.nan, 80, 80, 80])))


def test_missing_targets_do_not_enter_mae():
    assert masked_mae(np.array([1, 999]), np.array([3, np.nan])) == 2


def test_panel_selects_nodes_from_training_only_and_preserves_targets():
    times = pd.date_range('2023-12-31 20:00', periods=8, freq='h', tz='Asia/Shanghai')
    rows = []
    for i, time in enumerate(times):
        for node in ('a', 'b'):
            value = float(i) if node == 'a' or i >= 4 else np.nan
            rows.append({'timestamp': time, 'node_id': node, 'longitude': 104., 'latitude': 30.,
                             'pm2_5': value, 'aqi': value, 'pm10': value, 'so2': value,
                             'no2': value, 'co': value, 'o3': value})
    frame = pd.DataFrame(rows)
    weather = pd.DataFrame({'timestamp': times, 'temperature_c': 20.,
                                'relative_humidity_pct': 50., 'wind_u_ms': 1., 'wind_v_ms': 0.})
    result = prepare_panel(frame, weather, pd.Timestamp('2024-01-01', tz=times.tz))
    assert result['nodes'] == ['a']
    assert result['inputs'].shape[1] == 1
    np.testing.assert_allclose(result['scaler'].mean[0], 1.5)
    np.testing.assert_allclose(result['targets'][-1, 0], [7., 7.])


def test_recovery_metrics_penalizes_missed_recovery_without_fake_full_duration():
    records = pd.DataFrame({'actual_hours': [10., 100.], 'predicted_hours': [np.nan, 1.],
                                'event_id': [1, 2]})
    metrics = recovery_metrics(records, horizon=72)
    assert metrics['restricted_time_mae'] == 62.5  # ((68-10)+(68-1))/2
    assert metrics['missed_recovery_rate'] == 1
    assert metrics['false_recovery_rate'] == 1
    assert metrics['conditional_recovery_mae'] is None
