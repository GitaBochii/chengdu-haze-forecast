import pandas as pd

from scripts.build_causal_gefs_bias_correction import past_bias


def test_future_errors_excluded_and_exact_availability_allowed():
    times = pd.date_range('2023-01-01', periods=3, freq='h', tz='UTC')
    errors = pd.DataFrame({'available_at': times, 'error': [2., 4., 1000.]})
    assert past_bias(errors, times[1], minimum=2) == (3., 2)
    assert past_bias(errors, times[0], minimum=2) == (0., 1)


def test_missing_errors_not_counted():
    times = pd.date_range('2023-01-01', periods=2, freq='h', tz='UTC')
    errors = pd.DataFrame({'available_at': times, 'error': [float('nan'), 4.]})
    assert past_bias(errors, times[1], minimum=2) == (0., 1)
