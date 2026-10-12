"""Differentiable, unclipped lognormal likelihood for duration intervals."""

import math

import torch


_LOG_SQRT_2PI = 0.5 * math.log(2. * math.pi)
_QUAD_NODES = (0.0198550717512319, 0.101666761293187, 0.237233795041836,
               0.408282678752175, 0.591717321247825, 0.762766204958164,
               0.898333238706813, 0.980144928248768)
_QUAD_WEIGHTS = (0.0506142681451881, 0.111190517226687, 0.156853322938944,
                 0.181341891689181, 0.181341891689181, 0.156853322938944,
                 0.111190517226687, 0.0506142681451881)


def interval_aft_nll(mu, lower, upper, scale=1.2):
    """Return one NLL per row, with gradients with respect to log-time ``mu``.

    Exact labels use continuous density (including the log-time Jacobian),
    finite intervals use integrated mass, and +inf upper bounds use survival.
    Computation uses float64 on the input device without detaching ``mu``.
    Inputs must be nonempty, aligned one-dimensional real floating tensors.
    Nonrepresentable results raise rather than clipping the likelihood.
    """
    values = (mu, lower, upper)
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
           for value in values):
        raise ValueError('Require real floating-point tensors')
    if (mu.ndim != 1 or not mu.numel()
            or any(value.shape != mu.shape or value.device != mu.device for value in values)):
        raise ValueError('Require nonempty aligned one-dimensional tensors on one device')
    if not bool(torch.isfinite(mu).all()):
        raise ValueError('mu must be finite')
    if not bool((torch.isfinite(lower) & (lower > 0)).all()):
        raise ValueError('Lower bounds must be finite and strictly positive')
    if not bool(((torch.isfinite(upper) | torch.isposinf(upper)) & (upper >= lower)).all()):
        raise ValueError('Upper bounds must be >= lower and finite or positive infinity')
    try:
        if isinstance(scale, torch.Tensor) and scale.ndim != 0:
            raise ValueError('scale must be a finite positive scalar')
        sigma = float(scale)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('scale must be a finite positive scalar') from exc
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('scale must be a finite positive scalar')

    raw, lo, hi = (value.to(dtype=torch.float64) for value in values)
    log_lo = lo.log()
    result = torch.empty_like(raw)
    exact, censored = lo == hi, torch.isposinf(hi)
    # Index each branch before evaluation: unused invalid expressions must not
    # enter autograd, as they would with a final torch.where over all branches.
    if bool(exact.any()):
        z = (log_lo[exact] - raw[exact]) / sigma
        result[exact] = (0.5 * z) * z + log_lo[exact] + math.log(sigma) + _LOG_SQRT_2PI
    if bool(censored.any()):
        z = (log_lo[censored] - raw[censored]) / sigma
        result[censored] = -torch.special.log_ndtr(-z)
    interval = ~exact & ~censored
    if bool(interval.any()):
        left, right = lo[interval], hi[interval]
        close = (right - left) <= left
        log_width = torch.empty_like(left)
        log_width[close] = torch.log1p((right[close] - left[close]) / left[close])
        log_width[~close] = right[~close].log() - left[~close].log()
        width = log_width / sigma
        a = (log_lo[interval] - raw[interval]) / sigma
        b = a + width
        if not bool((width > 0).all()):
            raise FloatingPointError('An observation interval lost its numerical width')
        terms = torch.empty_like(a)
        narrow = width * torch.maximum(torch.ones_like(a), torch.maximum(a.abs(), b.abs())) <= 1e-3
        # Integrate the density ratio over the full interval when CDF subtraction
        # loses precision. Eight-point quadrature is not a midpoint label proxy.
        if bool(narrow.any()):
            start, span = a[narrow], width[narrow]
            nodes = a.new_tensor(_QUAD_NODES)
            log_weights = a.new_tensor(_QUAD_WEIGHTS).log()
            offset = span[:, None] * nodes
            log_integral = torch.logsumexp(log_weights - start[:, None] * offset
                                           - 0.5 * offset.square(), dim=1)
            terms[narrow] = (0.5 * start) * start + _LOG_SQRT_2PI - span.log() - log_integral
        regular = ~narrow
        if bool(regular.any()):
            ar, br = a[regular], b[regular]
            positive = ar >= 0
            log_large, log_small = torch.empty_like(ar), torch.empty_like(ar)
            log_large[positive] = torch.special.log_ndtr(-ar[positive])
            log_small[positive] = torch.special.log_ndtr(-br[positive])
            log_large[~positive] = torch.special.log_ndtr(br[~positive])
            log_small[~positive] = torch.special.log_ndtr(ar[~positive])
            terms[regular] = -log_large - torch.log(-torch.expm1(log_small - log_large))
        result[interval] = terms
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Lognormal interval NLL is not finite')
    return result
