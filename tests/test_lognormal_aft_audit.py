import json

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import log_ndtr
from scipy.stats import norm

from haze_forecast import lognormal_aft_audit as audit


def test_exact_likelihood_and_margin_gradient_match_formula():
    lower = np.array([0.1, 1., 8., 100.])
    margin = np.array([-1., 0., 3., -20.])
    scale = 1.2
    nll, gradient = audit.lognormal_interval_terms(lower, lower, margin, scale)
    z = (np.log(lower) - margin) / scale
    expected = 0.5 * z**2 + np.log(lower * scale) + 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(nll, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gradient, -z / scale, rtol=1e-13, atol=1e-13)
    assert nll[-1] > -np.log(1e-12)


def test_moderate_intervals_match_scipy_probability_and_density():
    lower = np.array([1., 2., 4., 7.])
    upper = np.array([3., 8., 6., 20.])
    margin = np.array([0., 2., 3., 1.])
    scale = 0.8
    a = (np.log(lower) - margin) / scale
    b = (np.log(upper) - margin) / scale
    probability = norm.cdf(b) - norm.cdf(a)
    nll, gradient = audit.lognormal_interval_terms(lower, upper, margin, scale)
    np.testing.assert_allclose(nll, -np.log(probability), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, (norm.pdf(b) - norm.pdf(a)) / (scale * probability),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('margin', [-1200., -60., 60., 1200.])
def test_extreme_interval_tails_match_scaled_quadrature(margin):
    lower, upper, scale = 1., 1.01, 1.2
    a = (np.log(lower) - margin) / scale
    width = np.log(upper / lower) / scale
    mode = min(max(0., a), a + width)

    def density_ratio(s):
        z = a + width * s
        return np.exp(-0.5 * (z - mode) * (z + mode))

    mass, _ = quad(density_ratio, 0., 1., epsabs=1e-12, epsrel=1e-12)
    first_moment, _ = quad(lambda s: (a + width * s) * density_ratio(s),
                           0., 1., epsabs=1e-12, epsrel=1e-12)
    expected_nll = -np.log(width * mass) + 0.5 * mode**2 + 0.5 * np.log(2 * np.pi)
    expected_gradient = -first_moment / mass / scale
    nll, gradient = audit.lognormal_interval_terms([lower], [upper], [margin], scale)
    np.testing.assert_allclose(nll, [expected_nll], rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(gradient, [expected_gradient], rtol=2e-9, atol=1e-8)


@pytest.mark.parametrize('margin', [-60., 0., 60.])
def test_next_float_width_is_not_rounded_into_exact_or_zero_mass(margin):
    lower = 1.
    upper = np.nextafter(lower, np.inf)
    scale = 1.2
    width = np.log1p((upper - lower) / lower) / scale
    a = -margin / scale
    nll, gradient = audit.lognormal_interval_terms([lower], [upper], [margin], scale)
    expected = -np.log(width) + 0.5 * a * a + 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(nll, [expected], rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(gradient, [-a / scale], rtol=1e-13, atol=1e-12)


def test_symmetric_interval_has_zero_gradient():
    nll, gradient = audit.lognormal_interval_terms([np.exp(-1.)], [np.exp(1.)], [0.], 1.)
    assert nll[0] == pytest.approx(-np.log(norm.cdf(1.) - norm.cdf(-1.)))
    assert gradient[0] == pytest.approx(0., abs=1e-15)


def test_right_censor_matches_survival_with_extreme_mills_ratio():
    lower = np.ones(5)
    a = np.array([-1000., -3., 0., 3., 1000.])
    scale = 1.2
    nll, gradient = audit.lognormal_interval_terms(lower, np.full(5, np.inf), -scale * a, scale)
    np.testing.assert_allclose(nll, -log_ndtr(-a), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gradient[1:4], -norm.pdf(a[1:4]) / norm.sf(a[1:4]) / scale,
                               rtol=1e-12, atol=1e-12)
    assert gradient[0] == 0.
    assert gradient[-1] == pytest.approx(-(1000. + 1 / 1000.) / scale, rel=1e-11)
    assert np.isfinite(nll).all() and np.isfinite(gradient).all()


def test_survival_far_below_location_returns_zero_without_spurious_overflow():
    nll, gradient = audit.lognormal_interval_terms([1.], [np.inf], [1e200])
    np.testing.assert_array_equal(nll, [0.])
    np.testing.assert_array_equal(gradient, [0.])


def test_time_rescaling_preserves_interval_terms_and_shifts_exact_density():
    lower = np.array([2., 1., 5.])
    upper = np.array([2., 4., np.inf])
    margin = np.array([1., 2., 1.])
    original_nll, original_gradient = audit.lognormal_interval_terms(lower, upper, margin)
    multiplier = 1e150
    nll, gradient = audit.lognormal_interval_terms(lower * multiplier, upper * multiplier,
                                                  margin + np.log(multiplier))
    np.testing.assert_allclose(nll, original_nll + np.array([np.log(multiplier), 0., 0.]),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, original_gradient, rtol=1e-12, atol=1e-12)


def test_mixed_penalized_gradient_matches_objective_finite_difference():
    matrix = np.array([[-1., 2.], [0., -1.], [1., 0.], [2., 1.]])
    lower = np.array([2., 1., 5., 3.])
    upper = np.array([2., 4., np.inf, 9.])
    parameters = np.array([1., 0.3, -0.2])
    penalty = 0.8
    step = 1e-5

    def evaluate(params):
        margin = params[0] + matrix @ params[1:]
        return audit.linear_aft_diagnostics(matrix, lower, upper, margin, params[1:], penalty)

    numeric = []
    for i in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += step
        minus[i] -= step
        numeric.append((evaluate(plus)['penalized_objective']
                        - evaluate(minus)['penalized_objective']) / (2 * step))
    result = evaluate(parameters)
    assert result['intercept_gradient'] == pytest.approx(numeric[0], abs=1e-9)
    assert result['gradient_inf_norm'] == pytest.approx(max(abs(np.asarray(numeric))), abs=1e-9)


def test_mixed_terms_gradient_matches_central_finite_difference():
    lower = np.array([2., 1., 2., 4., 1., 1., 1.])
    upper = np.array([2., 3., 8., np.inf, 1.01, 1.01, np.inf])
    margin = np.array([1., 1., 3., 1., -60., 60., -60.])
    step = 1e-4
    _, gradient = audit.lognormal_interval_terms(lower, upper, margin)
    plus, _ = audit.lognormal_interval_terms(lower, upper, margin + step)
    minus, _ = audit.lognormal_interval_terms(lower, upper, margin - step)
    np.testing.assert_allclose(gradient, (plus - minus) / (2 * step), rtol=2e-7, atol=2e-7)


def test_analytic_ridge_optimum_has_zero_full_gradient():
    rng = np.random.default_rng(1701)
    matrix = rng.normal(size=(40, 4))
    matrix -= matrix.mean(axis=0)
    log_time = 2. + matrix @ np.array([0.4, -0.2, 0.1, 0.7]) + rng.normal(scale=0.2, size=40)
    lower = np.exp(log_time)
    scale, penalty = 1.2, 0.7
    weights = np.linalg.solve(matrix.T @ matrix + len(matrix) * penalty * scale**2 * np.eye(4),
                              matrix.T @ (log_time - log_time.mean()))
    margin = log_time.mean() + matrix @ weights
    result = audit.linear_aft_diagnostics(matrix, lower, lower, margin, weights, penalty, scale)
    nll, _ = audit.lognormal_interval_terms(lower, lower, margin, scale)
    assert result['mean_nll'] == pytest.approx(nll.mean())
    assert result['penalized_objective'] == pytest.approx(nll.mean() + penalty * np.dot(weights, weights) / 2)
    assert result['gradient_inf_norm'] < 1e-12
    assert abs(result['intercept_gradient']) < 1e-12
    assert result['finite'] is True and result['success'] is True
    json.dumps(result, allow_nan=False)


def test_diagnostics_gradient_includes_unpenalized_intercept():
    matrix = np.array([[-1.], [0.], [1.]])
    lower = np.array([2., 3., 5.])
    upper = np.array([2., 7., np.inf])
    weights = np.array([0.2])
    margin = 1. + matrix @ weights
    result = audit.linear_aft_diagnostics(matrix, lower, upper, margin, weights, reg_lambda=2.)
    _, gradient = audit.lognormal_interval_terms(lower, upper, margin)
    intercept_gradient = gradient.mean()
    slope_gradient = (matrix.T @ gradient / len(matrix))[0] + 2. * weights[0]
    assert result['intercept_gradient'] == pytest.approx(intercept_gradient)
    assert result['gradient_inf_norm'] == pytest.approx(max(abs(intercept_gradient), abs(slope_gradient)))


def test_intercept_only_model_needs_no_slope_penalty():
    lower = np.array([2., 3., 5.])
    margin = np.full(3, np.log(lower).mean())
    result = audit.linear_aft_diagnostics(np.empty((3, 0)), lower, lower, margin, np.empty(0))
    assert result['gradient_inf_norm'] < 1e-12
    assert result['mean_nll'] == result['penalized_objective']


@pytest.mark.parametrize(('lower', 'upper', 'margin', 'scale'), [
    ([], [], [], 1.2),
    ([1.], [2., 3.], [0.], 1.2),
    ([[1.]], [2.], [0.], 1.2),
    ([1.], [2.], [[0.]], 1.2),
    ([0.], [2.], [0.], 1.2),
    ([np.nan], [2.], [0.], 1.2),
    ([1.], [np.nan], [0.], 1.2),
    ([1.], [-np.inf], [0.], 1.2),
    ([2.], [1.], [0.], 1.2),
    ([1.], [2.], [np.inf], 1.2),
    ([1.], [2.], [0.], 0.),
    ([1.], [2.], [0.], np.inf),
    ([1.], [2.], [0.], [1.2]),
])
def test_invalid_term_inputs_are_rejected(lower, upper, margin, scale):
    with pytest.raises(ValueError):
        audit.lognormal_interval_terms(lower, upper, margin, scale)


@pytest.mark.parametrize(('matrix', 'weights', 'penalty'), [
    ([1., 2.], [0.], 1.),
    ([[1.], [2.]], [[0.]], 1.),
    ([[1.], [2.]], [0., 1.], 1.),
    ([[1.]], [0.], 1.),
    ([[1.], [np.nan]], [0.], 1.),
    ([[1.], [2.]], [np.inf], 1.),
    ([[1.], [2.]], [0.], -1.),
    ([[1.], [2.]], [0.], np.inf),
])
def test_invalid_linear_diagnostic_inputs_are_rejected(matrix, weights, penalty):
    with pytest.raises(ValueError):
        audit.linear_aft_diagnostics(matrix, [1., 2.], [1., 2.], [0., 0.], weights, penalty)


def test_series_and_dataframe_index_mismatches_are_rejected():
    lower = pd.Series([1., 2.], index=[5, 3])
    upper = pd.Series([1., 2.], index=[3, 5])
    with pytest.raises(ValueError, match='index'):
        audit.lognormal_interval_terms(lower, upper, np.zeros(2))
    matrix = pd.DataFrame({'a': [0., 1.]}, index=[3, 5])
    with pytest.raises(ValueError, match='index'):
        audit.linear_aft_diagnostics(matrix, lower, lower, np.zeros(2), [0.])
    with pytest.raises(ValueError, match='index'):
        audit.linear_aft_diagnostics(matrix, [1., 2.], [1., 2.], np.zeros(2),
                                      pd.Series([0.], index=['wrong']))


def test_unrepresentable_terms_raise_instead_of_returning_nan_or_inf():
    with pytest.raises(FloatingPointError):
        audit.lognormal_interval_terms([1.], [1.], [1e308])
    result = audit.linear_aft_diagnostics([[0.]], [1.], [1.], [1e308], [0.])
    assert result['success'] is False and result['finite'] is False
    assert result['mean_nll'] is None and result['gradient_inf_norm'] is None
    assert result['failure_reason']
    json.dumps(result, allow_nan=False)
