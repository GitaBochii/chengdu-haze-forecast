import numpy as np
import pandas as pd

from haze_forecast.regional_air_experiment import (
    boundary_event_mask,
    prepare_panel,
    purge_event_windows,
)


def test_training_start_excludes_old_values_and_old_only_nodes():
    times = pd.date_range('2022-12-31 22:00', periods=6, freq='h', tz='Asia/Shanghai')
    rows = []
    for i, time in enumerate(times):
        for node in ('a', 'old'):
            value = 1000. if i < 2 else (2. if node == 'a' else np.nan)
            rows.append(dict(timestamp=time, node_id=node, longitude=104., latitude=30.,
                             **{c: value for c in ('pm2_5', 'pm10', 'aqi', 'so2', 'no2', 'co', 'o3_8h')}))
    weather = pd.DataFrame({'timestamp': times, 'temperature_c': 20., 'relative_humidity_pct': 50.,
                            'wind_u_ms': 1., 'wind_v_ms': 0.})
    result = prepare_panel(pd.DataFrame(rows), weather, times[-1] + pd.Timedelta(hours=1),
                           training_start=times[2], ozone_column='o3_8h')
    assert result['nodes'] == ['a']
    assert result['scaler'].mean[0] == 2.
    assert 'o3_8h' in result['feature_names']


def test_boundary_mask_includes_recovery_confirmation_and_ignores_zero_ids():
    times = pd.date_range('2023-12-31 18:00', periods=20, freq='h', tz='Asia/Shanghai')
    labels = pd.DataFrame({'event_id': [0, 1, 1, 1] + [0] * 16}, index=times)
    # Event recovery starts at 22:00; confirmation continues across midnight.
    mask = boundary_event_mask(labels, [times[6]])
    assert np.flatnonzero(mask).tolist() == list(range(1, 10))
    assert not boundary_event_mask(labels, [times[12]]).any()
    selected = purge_event_windows(np.array([10, 11, 12, 13]), mask, history=3, horizon=2)
    assert selected.tolist() == [12, 13]


def test_event_starting_at_boundary_is_not_a_crossing_event():
    times = pd.date_range('2024-01-01', periods=10, freq='h', tz='Asia/Shanghai')
    labels = pd.DataFrame({'event_id': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}, index=times)
    assert not boundary_event_mask(labels, [times[0]]).any()
