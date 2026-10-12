import numpy as np
import pytest
from scipy.stats import truncnorm

from haze_forecast.supervised_hurdle_math import median, nll


def test_density_interval_and_boundary():
    mu = np.log([8., 48.])
    p = np.array([.2, .8, .6, .4])
    lo, hi = np.array([3., 24., 10., 30.]), np.array([3., 24., 60., np.inf])
    s = 1.2
    a, b = (0-mu[0])/s, (np.log(24)-mu[0])/s
    c = (np.log(24)-mu[1])/s
    short = truncnorm(a, b, loc=mu[0], scale=s)
    long = truncnorm(c, np.inf, loc=mu[1], scale=s)
    expected = [(1-p[0])*short.pdf(np.log(3))/3,
                p[1]*long.pdf(np.log(24))/24,
                (1-p[2])*short.sf(np.log(10))+p[2]*long.cdf(np.log(60)),
                p[3]*long.sf(np.log(30))]
    np.testing.assert_allclose(nll(lo, hi, p, mu), -np.log(expected), rtol=1e-10)


def test_median_cdf_and_support():
    mu = np.log([8., 48.])
    p = np.array([.001, .2, .5, .8, .999])
    pred = median(p, mu)
    short = truncnorm(-mu[0]/1.2, (np.log(24)-mu[0])/1.2, loc=mu[0], scale=1.2)
    long = truncnorm((np.log(24)-mu[1])/1.2, np.inf, loc=mu[1], scale=1.2)
    np.testing.assert_allclose((1-p)*short.cdf(np.log(pred))+p*long.cdf(np.log(pred)), .5, atol=1e-12)
    assert (pred >= 1).all() and pred[2] == 24.


def test_full_support_and_single_class():
    mu = np.log([8., 48.])
    np.testing.assert_allclose(nll([1.], [np.inf], [.3], mu), 0., atol=1e-12)
    assert np.isfinite(nll([2.], [2.], [.3], mu)).all()


@pytest.mark.parametrize('p', [np.array([0.]), np.array([1.]), np.array([np.nan])])
def test_invalid_probability(p):
    with pytest.raises(ValueError):
        median(p, np.log([8., 48.]))
