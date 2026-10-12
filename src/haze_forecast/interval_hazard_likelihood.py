"""Stable integer-hour interval likelihood and EM sufficient statistics."""

import numpy as np
from scipy.special import logsumexp


def _hazards(values):
    hazard = np.asarray(values, dtype=float)
    if (hazard.ndim != 2 or 0 in hazard.shape or not np.isfinite(hazard).all()
            or (hazard <= 0).any() or (hazard >= 1).any()):
        raise ValueError('Expected nonempty rows by hours, with hazards strictly between zero and one')
    return hazard


def expectation(hazards, lower, upper):
    """Return log P(L<=T<=U), expected event counts, and at-risk counts.

    Hours start at one. An infinite U establishes survival through L-1 only;
    there is no imputed event after the observed censoring boundary.
    """
    hazard = _hazards(hazards)
    rows, horizon = hazard.shape
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.shape != (rows,) or hi.shape != (rows,):
        raise ValueError('Bounds must have one value per hazard row')
    finite = np.isfinite(hi)
    if (not np.isfinite(lo).all() or (lo < 1).any() or (lo != np.floor(lo)).any()
            or np.isnan(hi).any() or np.isneginf(hi).any() or (hi < lo).any()
            or (hi[finite] != np.floor(hi[finite])).any()
            or (hi[finite] > horizon).any() or (lo[~finite] > horizon + 1).any()):
        raise ValueError('Expected integer bounds covered by the hazard horizon')
    log_survival = np.column_stack([np.zeros(rows), np.cumsum(np.log1p(-hazard), axis=1)])
    log_mass = log_survival[:, :-1] + np.log(hazard)
    events = np.zeros_like(hazard)
    risk = np.zeros_like(hazard)
    log_probability = np.empty(rows)
    for row in range(rows):
        first = int(lo[row]) - 1
        if finite[row]:
            last = int(hi[row])
            normalizer = logsumexp(log_mass[row, first:last])
            log_probability[row] = normalizer
            events[row, first:last] = np.exp(log_mass[row, first:last] - normalizer)
            risk[row] = np.cumsum(events[row, ::-1])[::-1]
        else:
            log_probability[row] = log_survival[row, first]
            risk[row, :first] = 1.
    return log_probability, events, risk


def median_hours(hazards):
    """First integer hour whose CDF reaches one half; NaN if unresolved."""
    hazard = _hazards(hazards)
    crossed = np.cumsum(np.log1p(-hazard), axis=1) <= np.log(.5)
    result = np.argmax(crossed, axis=1).astype(float) + 1.
    result[~crossed.any(axis=1)] = np.nan
    return result
