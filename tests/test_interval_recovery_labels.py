import itertools

import numpy as np
import pandas as pd

from haze_forecast.interval_recovery_labels import interval_recovery_bounds


def frame(values):
    return pd.DataFrame({'timestamp': pd.date_range('2020-01-01', periods=len(values), freq='h', tz='Asia/Shanghai'), 'aqi': values})


def test_missing_gap_preserves_later_confirmation():
    f = frame([150.] * 11 + [np.nan] * 2 + [50.] * 6)
    result = interval_recovery_bounds(f, f.timestamp.iloc[:1]).iloc[0]
    assert result.lower_bound_hours == 11 and result.upper_bound_hours == 13
    assert pd.isna(result.exact_hours)
    assert result.label_available_at == f.timestamp.iloc[-1]


def test_known_high_excludes_incomplete_windows():
    f = frame([150.] * 10)
    r = interval_recovery_bounds(f, f.timestamp.iloc[:1]).iloc[0]
    assert r.lower_bound_hours == 10 and np.isposinf(r.upper_bound_hours)


def test_gap_does_not_always_destroy_exactness():
    f = frame([150., np.nan, 150.] + [50.] * 6)
    r = interval_recovery_bounds(f, f.timestamp.iloc[:1]).iloc[0]
    assert r.exact_hours == 3


def test_no_future_information():
    f = frame([150.])
    r = interval_recovery_bounds(f, f.timestamp).iloc[0]
    assert r.lower_bound_hours == 1 and np.isinf(r.upper_bound_hours) and not r.informative


def test_exhaustive_missing_assignments():
    rng = np.random.default_rng(41)
    for _ in range(80):
        values = rng.choice([50., 150.], size=12)
        missing = rng.choice(np.arange(1, 12), size=3, replace=False)
        values[missing] = np.nan
        f = frame(values)
        r = interval_recovery_bounds(f, f.timestamp.iloc[:1], stable_hours=3).iloc[0]
        earliest, latest = [], []
        for bits in itertools.product([50., 150.], repeat=3):
            v = values.copy()
            v[missing] = bits
            for future_value, destination in ((50., earliest), (150., latest)):
                extended = np.r_[v, [future_value] * 3]
                valid = [i for i in range(1, 13) if np.all(extended[i:i+3] <= 100)]
                destination.append(min(valid) if valid else np.inf)
        assert r.lower_bound_hours == min(earliest)
        assert r.upper_bound_hours == max(latest)
