import numpy as np
import pytest
from aft_binned_recovery import binned_bounds, survival_decode


def test_integer_bin_bounds_and_censoring():
    lo,hi = binned_bounds(np.array([1,2,19,20]))
    np.testing.assert_array_equal(lo,[0,1.5,18.5,19.5])
    np.testing.assert_array_equal(hi,[1.5,2.5,19.5,np.inf])
    with pytest.raises(ValueError):
        binned_bounds(np.array([0,21]))


def test_probability_is_cdf_not_exponentiated_location():
    mu = np.log(np.array([2.,19.5,100.]))
    start,p = survival_decode(mu,.7,.5)
    assert p[1] == .5
    assert np.all(np.diff(p)<0)
    assert start[2] == 20
    assert (start[:2]>=1).all() and (start[:2]<=19).all()
    abstain,_ = survival_decode(mu,.7,1.000001)
    np.testing.assert_array_equal(abstain,[20,20,20])


def test_extreme_margin_is_finite_or_censored_not_dropped():
    start,p = survival_decode(np.array([-100.,100.]),.35,.5)
    np.testing.assert_array_equal(start,[1,20])
    assert np.isfinite(p).all()
