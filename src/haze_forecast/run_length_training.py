"""Observed-prefix supervision and non-capped recovery medians."""

import numpy as np


def observed_transitions(future_aqi):
    values = np.asarray(future_aqi, dtype=float)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError('Require one-dimensional AQI, finite or missing')
    hours, states, targets = [], [], []
    state = 0
    for hour, value in enumerate(values, start=1):
        if np.isnan(value):
            break
        clean = int(value <= 100)
        hours.append(hour)
        states.append(state)
        targets.append(clean)
        state = state + 1 if clean else 0
        if state == 6:
            break
    return tuple(np.asarray(x, dtype=int) for x in (hours, states, targets))


def recovery_median(distribution):
    p = np.asarray(distribution, dtype=float)
    if (p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError('Require normalized nonnegative recovery probabilities and a tail')
    cdf = p[:, :-1].cumsum(axis=1)
    resolved = cdf[:, -1] >= .5
    return np.where(resolved, (cdf >= .5).argmax(axis=1) + 1., np.nan)
