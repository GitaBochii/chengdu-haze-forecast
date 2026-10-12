import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import lognorm

from haze_forecast.lognormal_aft_audit import lognormal_interval_terms
from haze_forecast.torch_interval_aft import interval_aft_nll


def tensors(mu, lower, upper, dtype=torch.float64):
    return (torch.tensor(mu, dtype=dtype, requires_grad=True),
            torch.tensor(lower, dtype=dtype), torch.tensor(upper, dtype=dtype))


def test_mixed_likelihood_matches_scipy_and_audit_gradient():
    mu, lo, hi = tensors([0., 2., 3., -2., 1.], [0.1, 2., 4., 7., 1.],
                         [0.1, 8., np.inf, 20., np.inf])
    loss = interval_aft_nll(mu, lo, hi)
    expected = [-lognorm.logpdf(0.1, 1.2, scale=np.exp(0.)),
                -np.log(lognorm.cdf(8., 1.2, scale=np.exp(2.))
                        - lognorm.cdf(2., 1.2, scale=np.exp(2.))),
                -lognorm.logsf(4., 1.2, scale=np.exp(3.)),
                -np.log(lognorm.cdf(20., 1.2, scale=np.exp(-2.))
                        - lognorm.cdf(7., 1.2, scale=np.exp(-2.))),
                -lognorm.logsf(1., 1.2, scale=np.exp(1.))]
    np.testing.assert_allclose(loss.detach(), expected, rtol=1e-11, atol=1e-11)
    loss.sum().backward()
    _, gradient = lognormal_interval_terms(lo.numpy(), hi.numpy(), mu.detach().numpy())
    np.testing.assert_allclose(mu.grad, gradient, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('location', [-1200., -60., 0., 60., 1200.])
@pytest.mark.parametrize('upper', [1.01, 1.000000000001, np.nextafter(1., 2.)])
def test_extreme_and_narrow_intervals_match_scaled_scipy_quadrature(location, upper):
    mu, lo, hi = tensors([location], [1.], [upper])
    a, width = -location / 1.2, np.log1p(upper - 1.) / 1.2
    mode = min(max(0., a), a + width)
    def density_ratio(s):
        offset = width * s
        return np.exp(-0.5 * (a - mode) * (a + mode) - a * offset - 0.5 * offset**2)
    mass = quad(density_ratio, 0., 1., epsabs=1e-12, epsrel=1e-12)[0]
    moment = quad(lambda s: (a + width * s) * density_ratio(s), 0., 1.,
                  epsabs=1e-12, epsrel=1e-12)[0]
    expected = 0.5 * mode**2 + 0.5 * np.log(2. * np.pi) - np.log(width * mass)
    loss = interval_aft_nll(mu, lo, hi)
    np.testing.assert_allclose(loss.detach(), [expected], rtol=1e-11, atol=1e-8)
    loss.sum().backward()
    np.testing.assert_allclose(mu.grad, [-moment / mass / 1.2], rtol=2e-8, atol=1e-8)


def test_autograd_matches_finite_difference_with_mixed_branches():
    mu, lo, hi = tensors([-50., 0., 50., 2., 1., 3.], [1.] * 6,
                         [1.01, 1.000000001, np.inf, 1., 8., np.inf])
    assert torch.autograd.gradcheck(lambda value: interval_aft_nll(value, lo, hi),
                                    (mu,), eps=1e-5, atol=1e-6, rtol=1e-5)


def test_float32_network_receives_finite_gradients_from_float64_work():
    mu, lo, hi = tensors([-60., 0., 60.], [1., 2., 3.], [1.01, 2., np.inf], torch.float32)
    loss = interval_aft_nll(mu, lo, hi)
    assert loss.dtype == torch.float64
    loss.mean().backward()
    assert mu.grad.dtype == torch.float32
    assert torch.isfinite(mu.grad).all()


@pytest.mark.parametrize('upper', [1., 2., np.inf])
def test_single_branch_batches_with_custom_scale(upper):
    mu, lo, hi = tensors([-1200., -60., 0., 60., 1200.], [1.] * 5, [upper] * 5)
    expected, gradient = lognormal_interval_terms(lo.numpy(), hi.numpy(),
                                                 mu.detach().numpy(), scale=0.8)
    loss = interval_aft_nll(mu, lo, hi, scale=0.8)
    np.testing.assert_allclose(loss.detach(), expected, rtol=1e-11, atol=1e-9)
    loss.sum().backward()
    np.testing.assert_allclose(mu.grad, gradient, rtol=2e-8, atol=1e-8)


def test_integer_and_nontensor_inputs_rejected():
    with pytest.raises(ValueError):
        interval_aft_nll(torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
    with pytest.raises(ValueError):
        interval_aft_nll([0.], [1.], [2.])


def test_unrepresentable_density_rejected_without_clipping():
    with pytest.raises(FloatingPointError):
        interval_aft_nll(*tensors([1e308], [1.], [1.]))


@pytest.mark.parametrize('mu,lo,hi', [([], [], []), ([0.], [0.], [1.]),
    ([0.], [-1.], [1.]), ([0.], [np.inf], [np.inf]), ([0.], [np.nan], [1.]),
    ([0.], [1.], [np.nan]), ([0.], [1.], [-np.inf]), ([0.], [2.], [1.]),
    ([np.inf], [1.], [2.]), ([np.nan], [1.], [2.]),
    ([0., 1.], [1.], [2.]), ([[0.]], [[1.]], [[2.]])])
def test_invalid_arrays_rejected(mu, lo, hi):
    with pytest.raises(ValueError):
        interval_aft_nll(*tensors(mu, lo, hi))


@pytest.mark.parametrize('scale', [0., -1., np.inf, np.nan, [1.2]])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError):
        interval_aft_nll(*tensors([0.], [1.], [2.]), scale=scale)
