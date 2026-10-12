import pandas as pd
import pytest

from scripts.collect_gefs_2020_support import plan_runs


def test_plan_is_timestamp_only_sorted_and_unique():
    times = pd.to_datetime(['2020-10-24 10:00', '2021-01-01 00:00',
                            '2020-10-24 09:00', '2020-03-01 00:00']).tz_localize('Asia/Shanghai')
    frame = pd.DataFrame({'timestamp': times, 'exact_hours': [1, 100, 20, 5]})
    assert plan_runs(frame) == [pd.Timestamp('2020-10-23 18:00', tz='UTC')]
    frame.exact_hours = 999
    assert plan_runs(frame) == [pd.Timestamp('2020-10-23 18:00', tz='UTC')]


def test_naive_time_rejected():
    with pytest.raises(ValueError):
        plan_runs(pd.DataFrame({'timestamp': pd.to_datetime(['2020-10-24'])}))


def test_duplicate_origin_rejected():
    time = pd.Timestamp('2020-10-24', tz='UTC')
    with pytest.raises(ValueError):
        plan_runs(pd.DataFrame({'timestamp': [time, time]}))
