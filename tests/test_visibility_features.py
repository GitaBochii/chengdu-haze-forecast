import numpy as np
import pandas as pd

from haze_forecast.visibility_features import visibility_features


def test_saturation_and_clock():
    ts = pd.date_range('2020-01-01', periods=40, freq='h', tz='Asia/Shanghai')
    raw = pd.DataFrame({'timestamp': ts, 'visibility_m': np.full(40, 9999.)})
    raw.loc[3, 'visibility_m'] = 1000.
    raw.loc[24, 'visibility_m'] = 3000.
    np.testing.assert_array_equal(visibility_features(raw, ts[[30]]), [[5000., 2000., 4000.]])
    raw.loc[raw.visibility_m == 9999, 'visibility_m'] = 9900.
    np.testing.assert_array_equal(visibility_features(raw, ts[[30]]), [[5000., 2000., 4000.]])
    raw.loc[27, 'visibility_m'] = np.nan
    assert visibility_features(raw, ts[[30]]).isna().all().all()
