"""Pseudo-Huber distance to an observed interval, in hours, not log time."""

import numpy as np


def interval_loss(prediction, lower, upper, delta=24.):
    p, lo, hi = [np.asarray(x, dtype=float) for x in (prediction, lower, upper)]
    if (p.shape != lo.shape or p.shape != hi.shape or p.ndim != 1
            or not np.isfinite(p).all() or not np.isfinite(lo).all()
            or np.isnan(hi).any() or (hi < lo).any() or (lo <= 0).any()
            or not np.isfinite(delta) or delta <= 0):
        raise ValueError('Invalid prediction, observed bounds, or scale')
    distance = p - np.clip(p, lo, hi)
    root = np.hypot(1., distance / delta)
    loss = delta ** 2 * (root - 1)
    gradient = distance / root
    # Interior interval points exert no force. Endpoints have a one-sided Hessian.
    active = (p <= lo) | (p >= hi)
    hessian = active.astype(float) / root ** 3
    return loss, gradient, hessian
