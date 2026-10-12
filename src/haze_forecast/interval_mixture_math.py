"""True observation likelihood and median of two same-scale lognormal experts."""

import numpy as np
import pandas as pd
from scipy.special import log_expit, log_ndtr, logsumexp

from haze_forecast.lognormal_aft_audit import lognormal_interval_terms


def _check_indexes(values):
    expected = None
    for value in values:
        if isinstance(value, (pd.Series, pd.DataFrame)):
            if expected is None:
                expected = value.index
            elif not value.index.equals(expected):
                raise ValueError('Pandas index must match other inputs in order')


def _real_array(value, name):
    try:
        if np.iscomplexobj(value):
            raise ValueError(f'{name} must be real numeric values')
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be real numeric values') from exc


def _prediction_inputs(locations, gate_margin, scale):
    _check_indexes([locations, gate_margin])
    raw = _real_array(locations, 'locations')
    gate = _real_array(gate_margin, 'gate_margin')
    sigma = _real_array(scale, 'scale')
    if (raw.ndim != 2 or raw.shape[1] != 2 or not len(raw)
            or gate.shape != (len(raw),)):
        raise ValueError('Require nonempty locations [n, 2] and gate_margin [n]')
    if not np.isfinite(raw).all() or not np.isfinite(gate).all():
        raise ValueError('Locations and gate_margin must be finite')
    if (sigma.shape != () or isinstance(scale, (bool, np.bool_))
            or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError('scale must be a finite positive scalar')
    return raw, gate, float(sigma)


def _log_weights(gate):
    return np.column_stack([log_expit(-gate), log_expit(gate)])


def mixture_observation_terms(lower, upper, locations, gate_margin, scale=1.2):
    """Return row NLL [n] and responsibilities [n, 2], without probability floors.

The raw gate logit belongs to component 1. Exact bounds use a density; finite
intervals use their integrated probability; infinite upper bounds use survival.
Arrays are positional; indexes of all pandas inputs must match in order.
"""
    _check_indexes([lower, upper, locations, gate_margin])
    raw, gate, sigma = _prediction_inputs(locations, gate_margin, scale)
    lo, hi = _real_array(lower, 'lower'), _real_array(upper, 'upper')
    component_nll = np.column_stack([
        lognormal_interval_terms(lo, hi, raw[:, component], sigma)[0]
        for component in range(2)
    ])
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        log_joint = _log_weights(gate) - component_nll
        log_probability = logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - log_probability[:, None])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    if not np.isfinite(log_probability).all() or not np.isfinite(responsibilities).all():
        raise FloatingPointError('Mixture observation terms are not finite')
    return -log_probability, responsibilities


def mixture_median(locations, gate_margin, scale=1.2):
    """Return positive finite median hours, with 64 log-space bisection steps.

The raw gate logit belongs to component 1. No time cap is applied. Invalid
inputs raise ValueError; unrepresentable numerical results raise
FloatingPointError rather than returning zero, NaN, or infinity.
"""
    raw, gate, sigma = _prediction_inputs(locations, gate_margin, scale)
    left_index = np.argmin(raw, axis=1)
    rows = np.arange(len(raw))
    left = raw[rows, left_index]
    right = raw[rows, 1 - left_index]
    low, high = left.copy(), right.copy()
    log_weights = _log_weights(gate)
    log_left = log_weights[rows, left_index]
    log_right = log_weights[rows, 1 - left_index]
    delta_sign = np.where(left_index == 1, np.sign(gate), -np.sign(gate))
    log_delta = np.full(len(raw), -np.inf)
    nonzero = gate != 0

    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        magnitude = np.abs(gate[nonzero])
        # log(abs(w_left - 0.5)); expm1 preserves even subnormal gate logits.
        log_delta[nonzero] = (np.log(-np.expm1(-magnitude)) - np.log(2.)
                              - np.log1p(np.exp(-magnitude)))
        signs = np.column_stack([delta_sign, np.ones(len(raw)), -np.ones(len(raw))])
        for _ in range(64):
            midpoint = low * 0.5 + high * 0.5
            # F(mid)-0.5 = (w_left-0.5) - w_left*SF_left + w_right*CDF_right.
            # Signed log tails avoid the rounded CDF=0.5 plateau far apart.
            log_terms = np.column_stack([
                log_delta,
                log_right + log_ndtr((midpoint - right) / sigma),
                log_left + log_ndtr((left - midpoint) / sigma),
            ])
            _, sign = logsumexp(log_terms, b=signs, axis=1, return_sign=True)
            low = np.where(sign <= 0, midpoint, low)
            high = np.where(sign >= 0, midpoint, high)
        result = np.exp(low * 0.5 + high * 0.5)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError('Mixture median is not representable as positive finite hours')
    return result
