import numpy as np
import pandas as pd
import pytest

from haze_forecast.causal_prediction_median import causal_prediction_median


def times(n):
    return pd.date_range('2023-01-01', periods=n, freq='h', tz='Asia/Shanghai')


def test_clock_conversion_not_raw_prediction_median():
    np.testing.assert_array_equal(causal_prediction_median(times(3), [10, 9, 8]), [10, 9, 8])
    np.testing.assert_array_equal(causal_prediction_median(times(3), [10, 20, 30]), [10, 14.5, 19])


def test_six_hour_window_and_expired_prediction():
    np.testing.assert_array_equal(causal_prediction_median(times(3), [0, 0, 12]), [0, 0, 0])
    values = causal_prediction_median(times(7), [100, 20, 20, 20, 20, 20, 20])
    assert values[-1] == 17.5


def test_gap_resets_history():
    t = times(4).delete(2)
    np.testing.assert_array_equal(causal_prediction_median(t, [10, 20, 2]), [10, 14.5, 2])


def test_future_does_not_change_prefix():
    prefix = causal_prediction_median(times(4), [10, 30, 3, 4])
    full = causal_prediction_median(times(6), [10, 30, 3, 4, 500, 0])
    np.testing.assert_array_equal(prefix, full[:4])


@pytest.mark.parametrize('t,p', [
    (times(2)[::-1], [1, 2]),
    (times(2).take([0, 0]), [1, 2]),
    (times(2).tz_localize(None), [1, 2]),
    (times(2), [1]),
    (times(2), [1, np.nan]),
    (times(2), [1, np.inf]),
    (times(2), [1, -1]),
    (times(0), []),
])
def test_invalid_inputs_rejected(t, p):
    with pytest.raises(ValueError):
        causal_prediction_median(t, p)
