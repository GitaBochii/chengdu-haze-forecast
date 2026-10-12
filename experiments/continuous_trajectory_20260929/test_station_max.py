import numpy as np
import pandas as pd
from station_max import station_grid, station_history


def test_past_station_values_only_and_missing_preserved():
    ts = pd.date_range('2020-01-01', periods=60, freq='h', tz='Asia/Shanghai')
    grid = np.arange(1620, dtype=float).reshape(60, 27)
    grid[20, 0] = np.nan
    h = station_history(grid, ts, np.array([30]), 2019)
    assert h.shape == (1, 24, 27)
    np.testing.assert_array_equal(h[0, -1], grid[29])
    assert np.isnan(h[0, 14, 0])
    changed = grid.copy()
    changed[30:] = 99999
    np.testing.assert_array_equal(h, station_history(changed, ts, np.array([30]), 2019))


def test_station_partition_mask_and_no_early_fill():
    ts = pd.date_range('2019-12-31 12:00', periods=40, freq='h', tz='Asia/Shanghai')
    h = station_history(np.ones((40, 27)), ts, np.array([13]), 2019)
    assert np.isnan(h[0, :-1]).all()
    np.testing.assert_array_equal(h[0, -1], np.ones(27))


def test_membership_uses_training_observations_not_future_stations():
    times = pd.date_range('2019-12-31 23:00', periods=2, freq='h', tz='Asia/Shanghai')
    rows = [dict(timestamp=times[0], station_id=f'S{i}', pm2_5=i, pm10=i + 10, aqi=i + 20)
            for i in range(9)]
    rows.append(dict(timestamp=times[1], station_id='FUTURE', pm2_5=5, pm10=10, aqi=15))
    ids, columns, grid = station_grid(pd.DataFrame(rows), times, 2019)
    assert ids == [f'S{i}' for i in range(9)]
    assert 'FUTURE' not in ids
    assert columns[:3] == [dict(station_id='S0', channel=c) for c in ('pm2_5', 'pm10', 'aqi')]
    np.testing.assert_array_equal(grid[0, :6], [0, 10, 20, 1, 11, 21])
    assert np.isnan(grid[1]).all()
