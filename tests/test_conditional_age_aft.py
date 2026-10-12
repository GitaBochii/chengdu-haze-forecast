import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import lognorm

from haze_forecast.conditional_age_aft import conditional_age_terms, remaining_median
from haze_forecast.lognormal_aft_audit import lognormal_interval_terms


def test_zero_age_reduces_to_original_distribution():
    lo = np.array([1., 3., 5.])
    hi = np.array([1., 9., np.inf])
    mu = np.array([2., 1., 3.])
    expected = lognormal_interval_terms(lo, hi, mu)
    actual = conditional_age_terms(lo, hi, mu, np.zeros(3))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(remaining_median(mu, np.zeros(3)), np.exp(mu))


@pytest.mark.parametrize('upper', [3., 8., np.inf])
def test_likelihood_and_gradient(upper):
    lo, hi, age, mu = np.array([3.]), np.array([upper]), np.array([12.]), np.array([2.5])
    loss, grad = conditional_age_terms(lo, hi, mu, age)
    dist = lognorm(s=1.2, scale=np.exp(mu[0]))
    numerator = dist.pdf(15) if upper == 3 else (dist.sf(15) - dist.sf(12+upper))
    np.testing.assert_allclose(loss[0], -np.log(numerator / dist.sf(12)), atol=1e-12)
    eps = 1e-5
    numerical = (conditional_age_terms(lo, hi, mu+eps, age)[0]
                 - conditional_age_terms(lo, hi, mu-eps, age)[0]) / (2*eps)
    np.testing.assert_allclose(grad, numerical, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('age,mu', [(0., 1.), (12., 2.5), (1000., -3.), (3., 8.)])
def test_median_matches_independent_root(age, mu):
    result = remaining_median(np.array([mu]), np.array([age]))[0]
    distribution = lognorm(s=1.2, scale=np.exp(mu))
    log_survival = distribution.logsf(age)
    root = brentq(lambda r: distribution.logsf(age+r)-log_survival+np.log(2), 0., 1e7)
    np.testing.assert_allclose(result, root, rtol=1e-8, atol=1e-8)


def test_invalid_age_rejected():
    with pytest.raises(ValueError):
        conditional_age_terms([1.], [2.], [1.], [-1.])
    with pytest.raises(ValueError):
        remaining_median([1.], [np.nan])
