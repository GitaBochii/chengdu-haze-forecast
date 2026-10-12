"""Conditional remaining-time lognormal primitives; no deployed model changes.

Age is elapsed time since an observed anchor, not a claimed physical onset.
Existing remaining-time regression is valid; this is an alternative family.
"""

import numpy as np
from scipy.special import log_ndtr, ndtri_exp

from haze_forecast.lognormal_aft_audit import _check_indexes, lognormal_interval_terms


def _age_inputs(margin, age, scale):
    _check_indexes([margin, age])
    mu, elapsed = np.asarray(margin, dtype=float), np.asarray(age, dtype=float)
    if mu.ndim != 1 or not len(mu) or elapsed.shape != mu.shape:
        raise ValueError('Require aligned nonempty one-dimensional margin and age')
    if not np.isfinite(mu).all() or not np.isfinite(elapsed).all() or (elapsed < 0).any():
        raise ValueError('Finite margins and nonnegative finite ages required')
    if np.asarray(scale).shape != () or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite scalar scale required')
    return mu, elapsed, float(scale)


def conditional_age_terms(lower, upper, margin, age, scale=1.2):
    """NLL/gradient for [age+lower,age+upper] conditional on survival past age."""
    _check_indexes([lower, upper, margin, age])
    mu, elapsed, scale = _age_inputs(margin, age, scale)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.shape != mu.shape or hi.shape != mu.shape:
        raise ValueError('Bounds must align with age and margin')
    if not np.isfinite(lo).all() or (lo <= 0).any() or np.isnan(hi).any() or (hi < lo).any():
        raise ValueError('Positive remaining bounds with upper >= lower required')
    shifted_lo, shifted_hi = lo + elapsed, hi + elapsed
    if (shifted_lo <= elapsed).any() or not np.isfinite(shifted_lo).all():
        raise FloatingPointError('Shifted lower bound lost precision or overflowed')
    if not np.array_equal(lo == hi, shifted_lo == shifted_hi):
        raise FloatingPointError('Shift changed interval semantics')
    nll, gradient = lognormal_interval_terms(shifted_lo, shifted_hi, mu, scale)
    positive = elapsed > 0
    if positive.any():
        survival_nll, survival_gradient = lognormal_interval_terms(
            elapsed[positive], np.full(positive.sum(), np.inf), mu[positive], scale)
        nll[positive] -= survival_nll
        gradient[positive] -= survival_gradient
    if not np.isfinite(nll).all() or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite conditional likelihood')
    return nll, gradient


def remaining_median(margin, age, scale=1.2):
    """Solve S(age+r)=S(age)/2 in log probability space, without a time cap."""
    mu, elapsed, scale = _age_inputs(margin, age, scale)
    positive = elapsed > 0
    result = np.empty(len(mu))
    with np.errstate(over='raise', invalid='raise'):
        result[~positive] = np.exp(mu[~positive])
        log_age = np.log(elapsed[positive])
        z = (log_age - mu[positive]) / scale
        log_tail_half = log_ndtr(-z) - np.log(2.)
        log_total = mu[positive] - scale * ndtri_exp(log_tail_half)
        result[positive] = elapsed[positive] * np.expm1(log_total - log_age)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError('Unrepresentable positive remaining median')
    return result
