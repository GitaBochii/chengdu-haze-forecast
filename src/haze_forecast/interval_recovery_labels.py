"""Sharp recovery-time bounds without guessing missing AQI observations."""

from numbers import Integral

import numpy as np
import pandas as pd

from haze_forecast.observed_recovery_labels import _hourly_index


def interval_recovery_bounds(frame, origins, stable_hours=6):
    if isinstance(stable_hours, bool) or not isinstance(stable_hours, Integral) or stable_hours < 1:
        raise ValueError('positive integer stable_hours required')
    times = _hourly_index(frame['timestamp'], 'frame')
    origin_times = _hourly_index(origins, 'origins').tz_convert(times.tz)
    values = pd.to_numeric(frame['aqi'], errors='raise').to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError('AQI cannot be infinite')
    grid = pd.date_range(times.min(), times.max(), freq='h')
    positions = grid.get_indexer(origin_times)
    if (positions < 0).any():
        raise ValueError('origins outside hourly grid')
    observed = pd.Series(values, index=times).reindex(grid).to_numpy()
    # Future beyond the archive is unknown, not clean. Padding allows a lower
    # bound even when the earliest possible clean run would cross the data end.
    padded = np.r_[observed, np.full(stable_hours, np.nan)]
    bad = np.isfinite(padded) & (padded > 100)
    clean = np.isfinite(padded) & (padded <= 100)
    kernel = np.ones(stable_hours, dtype=int)
    possible = np.flatnonzero(np.convolve(bad.astype(int), kernel, mode='valid') == 0)
    confirmed = np.flatnonzero(np.convolve(clean.astype(int), kernel, mode='valid') == stable_hours)
    rows = []
    for origin, pos in zip(origin_times, positions, strict=True):
        lower_pos = int(possible[np.searchsorted(possible, pos + 1)])
        k = np.searchsorted(confirmed, pos + 1)
        upper_pos = int(confirmed[k]) if k < len(confirmed) else None
        lower = float(lower_pos - pos)
        upper = float(upper_pos - pos) if upper_pos is not None else np.inf
        assert lower <= upper
        exact = lower == upper
        rows.append({'timestamp': origin, 'lower_bound_hours': lower, 'upper_bound_hours': upper,
            'exact_hours': lower if exact else np.nan,
            'informative': bool(np.isfinite(upper) or lower > 1),
            'label_available_at': grid[upper_pos + stable_hours - 1] if upper_pos is not None else grid[-1],
            'reason': 'exact' if exact else ('interval' if upper_pos is not None else 'right_censored')})
    return pd.DataFrame(rows)
