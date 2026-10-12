import numpy as np
import pandas as pd

from scripts.benchmark_interval_weather_triplet import weather_arms


def test_preserves_nonweather_and_original_missing_pattern():
    times = pd.date_range('2021-01-01', periods=30, freq='h', tz='UTC')
    proxy = pd.DataFrame({'timestamp': times, 'temperature_c': 10.,
                          'relative_humidity_pct': 80., 'wind_u_ms': 1., 'wind_v_ms': 2.})
    original = pd.DataFrame({'timestamp': times[-2:], 'temperature_c': [np.nan, 3.],
                             'aqi': [120., 130.], 'lower_bound_hours': [10., 9.]})
    arms = weather_arms(original, proxy, ['temperature_c', 'aqi'])
    np.testing.assert_allclose(arms['uniform'].temperature_c, [10, 10])
    np.testing.assert_allclose(arms['masked'].temperature_c, [np.nan, 10], equal_nan=True)
    for arm in arms.values():
        pd.testing.assert_frame_equal(arm[['timestamp', 'aqi', 'lower_bound_hours']],
                                      original[['timestamp', 'aqi', 'lower_bound_hours']])
    assert pd.isna(original.temperature_c.iloc[0])
