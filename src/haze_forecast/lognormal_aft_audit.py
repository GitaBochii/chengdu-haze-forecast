"""Unclipped lognormal likelihood and linear-AFT first-order diagnostics.

This module does not call a model library or fit a candidate. Exact observations
use a density, interval observations use integrated probability, and right
censoring uses survival probability. An interval is never imputed as one time.
All derivatives are with respect to the log-time location (raw margin).
"""

import numpy as np
import pandas as pd
from scipy.special import erfcx, log_ndtr, logsumexp, roots_legendre

_LOG_SQRT_2PI = 0.5 * np.log(2. * np.pi)
_SQRT_2 = np.sqrt(2.)
_SQRT_2_OVER_PI = np.sqrt(2. / np.pi)
_NARROW_VARIATION = 1e-3
_QUAD_NODES, _QUAD_WEIGHTS = roots_legendre(8)
_QUAD_NODES = (_QUAD_NODES + 1.) / 2.
_QUAD_LOG_WEIGHTS = np.log(_QUAD_WEIGHTS / 2.)


def _scalar(value, name, strictly_positive):
    if np.asarray(value).shape != ():
        raise ValueError(f'{name} must be a finite scalar')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if not np.isfinite(result) or (result <= 0 if strictly_positive else result < 0):
        raise ValueError(f'Invalid {name}')
    return result


def _check_indexes(values, expected=None):
    for value in values:
        if isinstance(value, pd.Series):
            if expected is None:
                expected = value.index
            elif not value.index.equals(expected):
                raise ValueError('Series index must match other inputs in order')


def _inputs(lower, upper, margin, scale):
    _check_indexes([lower, upper, margin])
    lo, hi, raw = (np.asarray(value, dtype=float) for value in (lower, upper, margin))
    if lo.ndim != 1 or not len(lo) or hi.shape != lo.shape or raw.shape != lo.shape:
        raise ValueError('Require nonempty aligned one-dimensional bounds and margin')
    if not np.isfinite(lo).all() or (lo <= 0).any():
        raise ValueError('Lower bounds must be finite and strictly positive')
    if np.isnan(hi).any() or np.isneginf(hi).any() or (hi < lo).any():
        raise ValueError('Upper bounds must be >= lower and finite or positive infinity')
    if not np.isfinite(raw).all():
        raise ValueError('Margin must be finite')
    return lo, hi, raw, _scalar(scale, 'scale', strictly_positive=True)


def _finite_interval(log_lower, lo, hi, raw, scale):
    log_width = np.log(hi) - log_lower
    close = (hi - lo) <= lo
    log_width[close] = np.log1p((hi[close] - lo[close]) / lo[close])
    width = log_width / scale
    a = (log_lower - raw) / scale
    b = a + width
    if (width <= 0).any():
        raise FloatingPointError('A nonzero observation interval lost its numerical width')
    nll = np.empty(len(lo))
    gradient = np.empty(len(lo))
    narrow = width * np.maximum(1., np.maximum(abs(a), abs(b))) <= _NARROW_VARIATION

    # Integrate a smooth density ratio on [0, 1] only when CDF subtraction
    # would be ill-conditioned. This preserves the complete interval label.
    if narrow.any():
        start, span = a[narrow], width[narrow]
        offset = span[:, None] * _QUAD_NODES
        log_weights = (_QUAD_LOG_WEIGHTS - start[:, None] * offset - 0.5 * offset**2)
        log_integral = logsumexp(log_weights, axis=1)
        nll[narrow] = ((0.5 * start) * start + _LOG_SQRT_2PI - np.log(span) - log_integral)
        conditional_offset = np.sum(offset * np.exp(log_weights - log_integral[:, None]), axis=1)
        gradient[narrow] = -(start + conditional_offset) / scale

    tails = ~narrow & ((a >= 0) | (b <= 0))
    if tails.any():
        right = a[tails] >= 0
        start = np.where(right, a[tails], -b[tails])
        span = width[tails]
        scaled_sf = erfcx(start / _SQRT_2)
        next_scaled_sf = erfcx((start + span) / _SQRT_2)
        log_pdf_ratio = -span * (start + 0.5 * span)
        log_sf_ratio = log_pdf_ratio + np.log(next_scaled_sf) - np.log(scaled_sf)
        probability_fraction = -np.expm1(log_sf_ratio)
        nll[tails] = -log_ndtr(-start) - np.log(probability_fraction)
        # erfcx removes the common exp(-z**2/2) factor from the Mills ratio.
        magnitude = ((_SQRT_2_OVER_PI / scaled_sf) * (-np.expm1(log_pdf_ratio))
                     / probability_fraction / scale)
        gradient[tails] = np.where(right, -magnitude, magnitude)

    central = ~narrow & ~tails
    if central.any():
        left, right, span = a[central], b[central], width[central]
        log_cdf_high, log_cdf_low = log_ndtr(right), log_ndtr(left)
        log_probability = log_cdf_high + np.log(-np.expm1(log_cdf_low - log_cdf_high))
        nll[central] = -log_probability
        log_pdf_ratio = -span * (left + 0.5 * span)
        log_pdf_max = -0.5 * np.minimum(left**2, right**2) - _LOG_SQRT_2PI
        nonzero = log_pdf_ratio != 0
        central_gradient = np.zeros(len(left))
        log_numerator = (log_pdf_max[nonzero]
                         + np.log(-np.expm1(-abs(log_pdf_ratio[nonzero]))))
        central_gradient[nonzero] = (np.sign(log_pdf_ratio[nonzero])
                                    * np.exp(log_numerator - log_probability[nonzero]) / scale)
        gradient[central] = central_gradient
    return nll, gradient


def lognormal_interval_terms(lower, upper, margin, scale=1.2):
    """Return arrays (unclipped NLL, derivative with respect to margin).

Arrays have positional semantics. Series indexes must agree when multiple
Series are provided. Invalid inputs raise ValueError; unrepresentable numerical
results raise FloatingPointError instead of quietly returning NaN or infinity.
"""
    lo, hi, raw, scale = _inputs(lower, upper, margin, scale)
    nll, gradient = np.empty(len(lo)), np.empty(len(lo))
    with np.errstate(divide='raise', invalid='raise', over='raise', under='ignore'):
        log_lower = np.log(lo)
        exact = lo == hi
        if exact.any():
            z = (log_lower[exact] - raw[exact]) / scale
            nll[exact] = (0.5 * z) * z + log_lower[exact] + np.log(scale) + _LOG_SQRT_2PI
            gradient[exact] = -z / scale
        censored = np.isposinf(hi)
        if censored.any():
            z = (log_lower[censored] - raw[censored]) / scale
            log_survival = log_ndtr(-z)
            nll[censored] = -log_survival
            nonnegative = z >= 0
            hazard = np.zeros(len(z))
            hazard[nonnegative] = _SQRT_2_OVER_PI / erfcx(z[nonnegative] / _SQRT_2)
            # Beyond this limit the negative-tail density is already far below
            # representable precision; do not overflow an unnecessary square.
            negative = (z < 0) & (z >= -np.sqrt(np.finfo(float).max))
            hazard[negative] = np.exp(-(0.5 * z[negative]) * z[negative] - _LOG_SQRT_2PI
                                      - log_survival[negative])
            gradient[censored] = -hazard / scale
        interval = ~exact & ~censored
        if interval.any():
            nll[interval], gradient[interval] = _finite_interval(
                log_lower[interval], lo[interval], hi[interval], raw[interval], scale)
    if not np.isfinite(nll).all() or not np.isfinite(gradient).all():
        raise FloatingPointError('Lognormal interval terms are not finite')
    return nll, gradient


def linear_aft_diagnostics(matrix, lower, upper, margin, weights, reg_lambda=1.0, scale=1.2):
    """Evaluate an unweighted linear AFT objective and its full gradient.

matrix is n by p, weights contains p slopes, and margin already includes the
unpenalized intercept. The penalty is lambda/2 * sum(weights**2). Callers must
verify that margin is the affine prediction represented by matrix and weights.
success means the diagnostic computation succeeded, not that a model converged
or generalized; convergence is assessed separately from gradient_inf_norm.
"""
    if isinstance(matrix, pd.DataFrame):
        _check_indexes([lower, upper, margin], matrix.index)
        if isinstance(weights, pd.Series) and not weights.index.equals(matrix.columns):
            raise ValueError('Coefficient Series index must equal matrix columns in order')
    lo, hi, raw, scale = _inputs(lower, upper, margin, scale)
    design = np.asarray(matrix, dtype=float)
    coefficients = np.asarray(weights, dtype=float)
    if (design.ndim != 2 or design.shape[0] != len(lo)
            or coefficients.shape != (design.shape[1],)):
        raise ValueError('Require an aligned two-dimensional matrix and one coefficient per column')
    if not np.isfinite(design).all() or not np.isfinite(coefficients).all():
        raise ValueError('Matrix and coefficients must be finite')
    penalty = _scalar(reg_lambda, 'reg_lambda', strictly_positive=False)
    result = {
        'rows': len(lo), 'features': design.shape[1], 'mean_nll': None,
        'penalized_objective': None, 'gradient_inf_norm': None, 'intercept_gradient': None,
        'finite': False, 'success': False, 'failure_reason': None,
    }
    try:
        nll, gradient = lognormal_interval_terms(lo, hi, raw, scale)
        with np.errstate(divide='raise', invalid='raise', over='raise', under='ignore'):
            mean_nll = float(np.sum(nll / len(lo)))
            penalty_value = float(0.5 * penalty * np.dot(coefficients, coefficients)) if penalty else 0.
            objective = mean_nll + penalty_value
            intercept_gradient = float(np.sum(gradient / len(lo)))
            slope_gradient = design.T @ (gradient / len(lo)) + penalty * coefficients
            norm = float(np.max(np.abs(np.r_[intercept_gradient, slope_gradient])))
        if not np.isfinite([mean_nll, objective, intercept_gradient, norm]).all():
            raise FloatingPointError('Linear diagnostics are not finite')
    except FloatingPointError as exc:
        result['failure_reason'] = str(exc)
        return result
    result.update({
        'mean_nll': mean_nll, 'penalized_objective': float(objective),
        'gradient_inf_norm': norm, 'intercept_gradient': intercept_gradient,
        'finite': True, 'success': True,
    })
    return result
