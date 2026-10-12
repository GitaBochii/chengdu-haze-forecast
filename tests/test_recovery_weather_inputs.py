import numpy as np
import pandas as pd
import pytest

from haze_forecast.recovery_weather_inputs import WeatherScaler, build_weather_inputs


def frame():
    return pd.DataFrame({
        'issue_utc': ['2022-01-02T03:00Z'], 'valid_utc': ['2022-01-02T06:00Z'],
        'run_utc': ['2022-01-01T18:00Z'], 'available_at_utc': ['2022-01-01T22:00Z'],
        'temperature_c': [10.], 'relative_humidity_pct': [50.],
        'wind_u_ms': [1.], 'wind_v_ms': [np.nan],
    })


def test_fixed_shape_and_missing_mask():
    values, mask = build_weather_inputs(frame(), ['2022-01-02T03:00Z', '2022-01-03T03:00Z'])
    assert values.shape == mask.shape == (2, 24, 4)
    assert mask.sum() == 3
    assert values[0, 0].tolist() == [10., 50., 1., 0.]
    assert not mask[1].any()
    assert np.isfinite(values).all()


def test_not_yet_available_values_are_masked():
    rows = frame().assign(available_at_utc='2022-01-02T04:00Z')
    values, mask = build_weather_inputs(rows, ['2022-01-02T03:00Z'])
    assert not mask.any() and not values.any()


@pytest.mark.parametrize('column,value', [
    ('valid_utc', '2022-01-02T07:00Z'),
    ('run_utc', '2022-01-02T06:00Z'),
    ('available_at_utc', '2022-01-01T15:00Z'),
    ('issue_utc', '2022-01-02 03:00'),
])
def test_reject_invalid_time_contract(column, value):
    with pytest.raises(ValueError):
        build_weather_inputs(frame().assign(**{column: value}), ['2022-01-02T03:00Z'])


def test_duplicate_keys_rejected():
    with pytest.raises(ValueError):
        build_weather_inputs(pd.concat([frame(), frame()]), ['2022-01-02T03:00Z'])


def test_scaler_uses_training_only_and_masks_after_transform():
    values = np.array([[[1., 3., 5., 7.]], [[3., 5., 7., 9.]], [[1000., 1000., 1000., 1000.]]])
    mask = np.ones_like(values, dtype=bool)
    scaler = WeatherScaler.fit(values, mask, np.array([True, True, False]))
    assert scaler.mean.tolist() == [2., 4., 6., 8.]
    assert scaler.scale.tolist() == [1., 1., 1., 1.]
    mask[2] = False
    scaled = scaler.transform(values, mask)
    assert not scaled[2].any()
    np.testing.assert_array_equal(scaled[0], [[-1.] * 4])


def test_scaler_rejects_absent_training_feature():
    with pytest.raises(ValueError):
        WeatherScaler.fit(np.zeros((1, 24, 4)), np.zeros((1, 24, 4), dtype=bool), np.array([True]))
