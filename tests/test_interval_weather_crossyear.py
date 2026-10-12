"""Protect timestamp alignment independently of cohort row ordering."""

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_interval_weather_crossyear import aligned_predictions


def test_prediction_alignment_uses_timestamp_not_cohort_position():
    times = pd.Series(pd.date_range('2025-01-01', periods=3, freq='h', tz='Asia/Shanghai'))
    cohort = pd.DataFrame({'timestamp': times.iloc[[2, 0]].reset_index(drop=True)})
    np.testing.assert_array_equal(aligned_predictions(cohort, times, [10, 20, 30]), [30, 10])


def test_prediction_alignment_rejects_duplicate_or_uncovered_origins():
    times = pd.Series(pd.date_range('2025-01-01', periods=2, freq='h', tz='Asia/Shanghai'))
    with pytest.raises(ValueError, match='Duplicate'):
        aligned_predictions(pd.DataFrame({'timestamp': [times[0], times[0]]}), times, [1, 2])
    with pytest.raises(ValueError, match='Duplicate'):
        aligned_predictions(pd.DataFrame({'timestamp': times}), pd.Series([times[0], times[0]]), [1, 2])
    with pytest.raises(KeyError):
        aligned_predictions(pd.DataFrame({'timestamp': [times[1] + pd.Timedelta(hours=1)]}), times, [1, 2])
