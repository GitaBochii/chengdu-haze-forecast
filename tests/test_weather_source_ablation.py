import numpy as np
import pandas as pd
import pytest

from haze_forecast.weather_source_ablation import feature_arms, uniform_weather_rows


def proxy():
    times = pd.date_range('2023-12-29', periods=96, freq='h', tz='Asia/Shanghai')
    return pd.DataFrame({'timestamp': times, **{
        name: np.arange(96, dtype=float)
        for name in ('temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms')}})


def test_weather_lags_use_full_hourly_grid_not_sparse_event_rows():
    weather = proxy()
    rows = pd.DataFrame({'timestamp': weather.timestamp.iloc[[30, 50]].to_list(),
                         'event_key': ['a', 'b'], 'remaining_hours': [4., 8.],
                         'aqi': [120., 160.], 'hour_sin': [0.3, 0.4]})
    got = uniform_weather_rows(rows, weather)
    pd.testing.assert_frame_equal(got[rows.columns], rows)
    assert got.temperature_c_lag_1.tolist() == [29., 49.]
    assert got.temperature_c_lag_24.tolist() == [6., 26.]
    assert got.temperature_c_roll_24.tolist() == [17.5, 37.5]
    assert got.temperature_c_change_3.tolist() == [3., 3.]
    assert np.allclose(got.wind_speed_ms, np.sqrt(2) * np.array([30., 50.]))
    weather.loc[weather.index > 50, 'temperature_c'] = 1e8
    pd.testing.assert_frame_equal(got, uniform_weather_rows(rows, weather))


@pytest.mark.parametrize('problem', ['duplicate', 'naive', 'gap', 'uncovered'])
def test_bad_weather_alignment_rejected(problem):
    weather = proxy()
    rows = weather.iloc[[30]][['timestamp']].copy()
    if problem == 'duplicate':
        weather = pd.concat([weather, weather.iloc[[0]]])
    elif problem == 'naive':
        weather['timestamp'] = weather.timestamp.dt.tz_localize(None)
    elif problem == 'gap':
        weather = weather.drop(index=29)
    else:
        rows['timestamp'] += pd.Timedelta(days=10)
    with pytest.raises(ValueError):
        uniform_weather_rows(rows, weather)


def test_feature_groups_do_not_leak_weather_or_pbl():
    import json
    from pathlib import Path

    metadata = json.loads((Path(__file__).resolve().parents[1] / 'artifacts' /
                           'cnemc_2025_deployment_candidate/model_metadata.json').read_text(encoding='utf-8'))
    groups = feature_arms(metadata['feature_names'])
    assert {k: len(v) for k, v in groups.items()} == {
        'baseline': 87, 'mixed_four': 74, 'uniform_four': 74, 'pollution_only': 49}
    assert groups['mixed_four'] == groups['uniform_four']
    assert all(not c.startswith('boundary_layer_height_m') for c in groups['mixed_four'])
    assert all(not c.startswith(('temperature_c', 'relative_humidity_pct', 'wind_'))
               for c in groups['pollution_only'])
