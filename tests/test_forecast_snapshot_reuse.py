import numpy as np
import pandas as pd
import pytest

from haze_forecast.forecast_snapshot_reuse import reuse_forecast_snapshots


def ts(hour):
    return pd.Timestamp('2023-01-01', tz='UTC') + pd.Timedelta(hours=hour)


def test_no_future_publication_or_expired_forecast_and_preserve_order():
    snapshots = pd.DataFrame({'known_at': [ts(2), ts(5)], 'valid_at': [ts(4), ts(8)],
                              'run_at': [ts(0), ts(3)], 'wind': [10., 20.]})
    result = reuse_forecast_snapshots(snapshots, [ts(5), ts(1), ts(3), ts(4), ts(8)], ['wind'])
    np.testing.assert_allclose(result.wind, [20, np.nan, 10, np.nan, np.nan], equal_nan=True)
    assert result.loc[0, 'forecast_known_age_hours'] == 0
    assert result.loc[2, 'forecast_remaining_lead_hours'] == 1


def test_older_snapshot_can_outlive_newer_snapshot():
    snapshots = pd.DataFrame({'known_at': [ts(1), ts(2)], 'valid_at': [ts(10), ts(3)],
                              'run_at': [ts(0), ts(0)], 'wind': [10., 20.]})
    result = reuse_forecast_snapshots(snapshots, [ts(4)], ['wind'])
    assert result.wind.iloc[0] == 10


def test_future_rows_cannot_change_past_result():
    snapshots = pd.DataFrame({'known_at': [ts(1), ts(5)], 'valid_at': [ts(10), ts(20)],
                              'run_at': [ts(0), ts(4)], 'wind': [10., 20.]})
    pd.testing.assert_frame_equal(reuse_forecast_snapshots(snapshots, [ts(3)], ['wind']),
                                  reuse_forecast_snapshots(snapshots.iloc[:1], [ts(3)], ['wind']))


def test_reject_run_after_known_time():
    snapshots = pd.DataFrame({'known_at': [ts(1)], 'valid_at': [ts(10)],
                              'run_at': [ts(2)], 'wind': [10.]})
    with pytest.raises(ValueError):
        reuse_forecast_snapshots(snapshots, [ts(3)], ['wind'])
