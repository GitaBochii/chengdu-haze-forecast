import numpy as np
import pandas as pd
import pytest

from haze_forecast.altimeter_features import altimeter_features


def fixture():
    times = pd.date_range('2020-01-01', periods=60, freq='h', tz='Asia/Shanghai')
    return pd.DataFrame({'timestamp': times, 'altimeter_setting_hpa': 1000.+np.arange(60)})


def test_clock_order_and_future_invariance():
    d = fixture()
    times = d.timestamp.iloc[[40, 30]]
    x = altimeter_features(d, times)
    np.testing.assert_array_equal(x.to_numpy(), [[1037., 3., 24.], [1027., 3., 24.]])
    changed = d.copy()
    changed.loc[changed.timestamp > times.min(), 'altimeter_setting_hpa'] = 999.
    np.testing.assert_array_equal(altimeter_features(changed, times.iloc[[1]]), x.iloc[[1]])


def test_missing_is_not_row_shift_or_filled():
    d = fixture()
    x = altimeter_features(d.drop(index=27), d.timestamp.iloc[[30]])
    assert x.isna().all().all()
    x = altimeter_features(d.drop(index=24), d.timestamp.iloc[[30]])
    assert x.iloc[0, 0] == 1027 and pd.isna(x.iloc[0, 1]) and x.iloc[0, 2] == 24


def test_duplicate_source_rejected():
    d = fixture()
    with pytest.raises(ValueError):
        altimeter_features(pd.concat([d, d.iloc[[0]]]), d.timestamp.iloc[[30]])
