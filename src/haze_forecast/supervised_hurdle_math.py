"""Fixed-support truncated lognormal duration mixture, sigma=1.2."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm

from haze_forecast.lognormal_aft_audit import lognormal_interval_terms


def check(p, mu):
    p, mu = np.asarray(p, float), np.asarray(mu, float)
    if (p.ndim != 1 or not len(p) or not np.isfinite(p).all()
            or (p <= 0).any() or (p >= 1).any()
            or mu.shape != (2,) or not np.isfinite(mu).all()):
        raise ValueError('Require finite open probabilities and two locations')
    return p, mu


def nll(lower, upper, p, mu):
    p, mu = check(p, mu)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if (lo.shape != p.shape or hi.shape != p.shape or not np.isfinite(lo).all()
            or (lo < 1).any() or np.isnan(hi).any() or (hi < lo).any()):
        raise ValueError('Invalid observation bounds')
    terms = np.full((len(p), 2), -np.inf)
    exact = lo == hi
    for j, (left, right) in enumerate(((1., 24.), (24., np.inf))):
        a, b = np.maximum(lo, left), np.minimum(hi, right)
        mask = (~exact & (b > a)) | (exact & (lo >= left) & (lo < right))
        if not mask.any():
            continue
        normalizer = lognormal_interval_terms(
            np.array([left]), np.array([right]), np.array([mu[j]]), 1.2)[0][0]
        terms[mask, j] = -lognormal_interval_terms(
            a[mask], b[mask], np.full(mask.sum(), mu[j]), 1.2)[0] + normalizer
    result = -logsumexp(terms + np.column_stack((np.log1p(-p), np.log(p))), axis=1)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite mixture likelihood')
    return result


def median(p, mu):
    p, mu = check(p, mu)
    result = np.full(len(p), 24.)
    short, long = p < .5, p > .5
    a, b = -mu[0]/1.2, (np.log(24)-mu[0])/1.2
    c = (np.log(24)-mu[1])/1.2
    result[short] = np.exp(truncnorm.ppf(.5/(1-p[short]), a, b, loc=mu[0], scale=1.2))
    result[long] = np.exp(truncnorm.isf(.5/p[long], c, np.inf, loc=mu[1], scale=1.2))
    if not np.isfinite(result).all() or (result < 1).any():
        raise FloatingPointError('Invalid median')
    return result
