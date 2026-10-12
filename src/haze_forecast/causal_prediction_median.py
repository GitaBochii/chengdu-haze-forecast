"""Six-hour causal endpoint aggregation, isolated from deployed predictors."""

import numpy as np
import pandas as pd


def causal_prediction_median(timestamps, predictions):
    t = pd.DatetimeIndex(timestamps)
    p = np.asarray(predictions, dtype=float)
    if (not len(t) or t.tz is None or t.hasnans or not t.is_unique
            or not t.is_monotonic_increasing or p.shape != (len(t),)
            or not np.isfinite(p).all() or (p < 0).any()):
        raise ValueError('Unique increasing zoned times and aligned finite nonnegative predictions required')
    result = np.empty(len(t))
    start = 0
    for i, now in enumerate(t):
        if i and now - t[i - 1] != pd.Timedelta(hours=1):
            start = i
        start = max(start, i - 5)
        # Compare recovery endpoints, not remaining times issued at different hours.
        hours = (now - t[start:i + 1]).total_seconds().to_numpy() / 3600
        result[i] = np.median(np.maximum(0., p[start:i + 1] - hours))
    return result
