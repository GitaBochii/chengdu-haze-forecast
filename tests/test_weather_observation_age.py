import numpy as np
import pandas as pd
import pytest

from haze_forecast.weather_observation_age import weather_observation_age
from haze_forecast.weather_source_ablation import WEATHER_BASE


def fixture():
    return pd.DataFrame({'timestamp': pd.date_range('2022-01-01', periods=5, freq='h', tz='UTC'),
                         **{name: [np.nan, 5., np.nan, np.nan, 10.] for name in WEATHER_BASE}})


def test_gap_age_and_prefix_causality():
    frame = fixture()
    actual = weather_observation_age(frame)
    np.testing.assert_allclose(actual.temperature_c_observation_age_h, [np.nan, 0, 1, 2, 0], equal_nan=True)
    np.testing.assert_array_equal(actual.temperature_c_unobserved, [1, 0, 1, 1, 0])
    pd.testing.assert_frame_equal(actual.iloc[:4], weather_observation_age(frame.iloc[:4]))
    changed = frame.copy()
    changed.loc[4, list(WEATHER_BASE)] = np.nan
    pd.testing.assert_frame_equal(actual.iloc[:4], weather_observation_age(changed).iloc[:4])


def test_missing_hour_is_not_silently_compressed():
    with pytest.raises(ValueError):
        weather_observation_age(fixture().drop(index=2))
