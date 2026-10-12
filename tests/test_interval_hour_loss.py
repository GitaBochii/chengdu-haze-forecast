import numpy as np
import pytest

from haze_forecast.interval_hour_loss import interval_loss


def test_derivatives_match_finite_differences_away_from_endpoints():
    p = np.array([2., 8., 30., 10.])
    lo = np.array([5., 5., 5., 4.])
    hi = np.array([5., 12., 12., np.inf])
    loss, gradient, hessian = interval_loss(p, lo, hi)
    step = 1e-4
    plus = interval_loss(p + step, lo, hi)
    minus = interval_loss(p - step, lo, hi)
    np.testing.assert_allclose((plus[0] - minus[0]) / (2 * step), gradient, atol=1e-7)
    np.testing.assert_allclose((plus[1] - minus[1]) / (2 * step), hessian, atol=1e-7)
    assert loss[1] == loss[3] == gradient[1] == hessian[1] == 0


def test_exact_label_does_not_lose_curvature_at_zero_error():
    loss, gradient, hessian = interval_loss([5.], [5.], [5.])
    assert loss[0] == gradient[0] == 0
    assert hessian[0] == 1


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        interval_loss([3.], [5.], [2.])
