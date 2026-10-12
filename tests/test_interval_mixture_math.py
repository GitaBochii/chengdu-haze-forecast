"""Independent probability and quantile checks for two lognormal components."""

import importlib

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import brentq
from scipy.special import expit, ndtri_exp
from scipy.stats import norm


def _math():
    return importlib.import_module('haze_forecast.interval_mixture_math')


def test_observation_terms_match_independent_density_interval_and_survival():
    lower = np.array([2., 3., 9.])
    upper = np.array([2., 7., np.inf])
    locations = np.array([[0.5, 2.], [1., 2.5], [0., 3.]])
    gate = np.array([-0.7, 0.4, 1.3])
    scale = 1.2
    zlo = (np.log(lower[:, None]) - locations) / scale
    zhi = (np.log(upper[:, None]) - locations) / scale
    likelihood = np.stack([
        norm.pdf(zlo[0]) / (lower[0] * scale),
        norm.cdf(zhi[1]) - norm.cdf(zlo[1]),
        norm.sf(zlo[2]),
    ])
    joint = likelihood * np.column_stack([expit(-gate), expit(gate)])
    expected_probability = joint.sum(axis=1)
    nll, responsibilities = _math().mixture_observation_terms(lower, upper, locations, gate)
    assert_allclose(nll, -np.log(expected_probability), rtol=2e-14, atol=2e-14)
    assert_allclose(responsibilities, joint / expected_probability[:, None], rtol=2e-14)
    assert nll.dtype == responsibilities.dtype == np.dtype('float64')
    assert_allclose(responsibilities.sum(axis=1), 1., rtol=0, atol=2e-16)


def test_extreme_gate_selects_component_one_for_positive_raw_margin_without_floor():
    locations = np.array([[0., 2.], [0., 2.]])
    gate = np.array([-1000., 1000.])
    nll, responsibilities = _math().mixture_observation_terms([1., 1.], [1., 1.],
                                                            locations, gate)
    expected = -norm.logpdf(np.array([0., -2.]) / 1.2) + np.log(1.2)
    assert_allclose(nll, expected, rtol=2e-14)
    assert_array_equal(responsibilities, [[1., 0.], [0., 1.]])
    assert_allclose(_math().mixture_median(locations, gate), np.exp([0., 2.]), rtol=2e-14)


def test_observation_can_underflow_responsibility_to_exact_zero():
    nll, responsibilities = _math().mixture_observation_terms(
        [1.], [np.inf], [[-100., 100.]], [0.])
    assert_allclose(nll, [np.log(2.)], rtol=2e-14)
    assert_array_equal(responsibilities, [[0., 1.]])


def test_identical_components_reduce_to_one_lognormal_for_any_gate():
    lower = np.array([1., 5., 20.])
    upper = np.array([1., 9., np.inf])
    mu = np.array([-1., 1., 4.])
    gate = np.array([-3., 0.5, 900.])
    locations = np.column_stack([mu, mu])
    zlo = (np.log(lower) - mu) / 1.2
    zhi = (np.log(upper) - mu) / 1.2
    likelihood = np.array([norm.pdf(zlo[0]) / (lower[0] * 1.2),
                           norm.cdf(zhi[1]) - norm.cdf(zlo[1]), norm.sf(zlo[2])])
    nll, responsibilities = _math().mixture_observation_terms(lower, upper, locations, gate)
    assert_allclose(nll, -np.log(likelihood), rtol=2e-14)
    assert_allclose(responsibilities, np.column_stack([expit(-gate), expit(gate)]), rtol=2e-14)
    assert_allclose(_math().mixture_median(locations, gate), np.exp(mu), rtol=2e-14)


@pytest.mark.parametrize('locations', [[[1., 5.]], [[-100., 100.]], [[100., -100.]]])
def test_symmetric_median_is_log_midpoint_even_across_a_rounded_cdf_plateau(locations):
    result = _math().mixture_median(locations, [0.])
    assert_allclose(np.log(result), np.mean(locations, axis=1), rtol=0, atol=2e-14)


@pytest.mark.parametrize('gate', [-3., -0.2, 0.2, 3.])
def test_biased_median_matches_independent_scipy_root(gate):
    locations = np.array([[np.log(2.), np.log(120.)]])
    weights = np.array([expit(-gate), expit(gate)])

    def objective(raw):
        return np.dot(weights, norm.cdf((raw - locations[0]) / 1.2)) - 0.5

    expected = brentq(objective, *locations[0], xtol=1e-14)
    result = _math().mixture_median(locations, [gate])
    assert_allclose(np.log(result), [expected], rtol=2e-14, atol=2e-14)
    assert_allclose(objective(np.log(result[0])), 0., rtol=0, atol=2e-15)


@pytest.mark.parametrize('gate', [1e-12, -1e-12, 1e-15, -1e-15, 1e-300, -1e-300,
                                  np.nextafter(0., 1.), -np.nextafter(0., 1.)])
def test_wide_mixture_preserves_even_tiny_gate_sign(gate):
    # At this separation, the opposite tail changes the root by < 1e-100.
    # expm1 gives an independent expression for the dominant tail probability.
    magnitude = abs(gate)
    log_tail_probability = np.log(np.expm1(magnitude)) - np.log(2.) - magnitude
    expected = np.sign(gate) * (100. + 1.2 * ndtri_exp(log_tail_probability))
    result = _math().mixture_median([[-100., 100.]], [gate])
    assert_allclose(np.log(result), [expected], rtol=0, atol=2e-12)
    reversed_result = _math().mixture_median([[100., -100.]], [-gate])
    assert_allclose(result, reversed_result, rtol=2e-14)


def test_median_has_no_hour_cap_and_does_not_change_inputs():
    locations = np.log(np.array([[1000., 2000.], [0.01, 0.02]]))
    gate = np.array([1., -1.])
    saved_locations, saved_gate = locations.copy(), gate.copy()
    result = _math().mixture_median(locations, gate)
    assert 1000. < result[0] < 2000.
    assert 0.01 < result[1] < 0.02
    assert_array_equal(locations, saved_locations)
    assert_array_equal(gate, saved_gate)


@pytest.mark.parametrize('location', [-1000., 1000., np.finfo(float).max])
def test_unrepresentable_median_is_explicit_failure(location):
    with pytest.raises(FloatingPointError):
        _math().mixture_median([[location, location]], [0.])


@pytest.mark.parametrize('locations, gate', [
    ([], []), ([1., 2.], [0.]), ([[1.]], [0.]), ([[1., 2., 3.]], [0.]),
    ([[1., 2.]], [0., 1.]), ([[1., 2.]], [[0.]]),
    ([[np.nan, 2.]], [0.]), ([[np.inf, 2.]], [0.]),
    ([[1., 2.]], [np.nan]), ([[1., 2.]], [np.inf]),
    ([[1. + 1j, 2.]], [0.]), ([[1., 2.]], [1j]),
    ([['invalid', 2.]], [0.]),
])
def test_invalid_prediction_inputs_raise_value_error(locations, gate):
    with pytest.raises(ValueError):
        _math().mixture_median(locations, gate)
    with pytest.raises(ValueError):
        _math().mixture_observation_terms([1.], [2.], locations, gate)


@pytest.mark.parametrize('scale', [0., -1., np.nan, np.inf, [1.2], True, 1.2 + 1j])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        _math().mixture_median([[0., 1.]], [0.], scale=scale)
    with pytest.raises(ValueError):
        _math().mixture_observation_terms([1.], [2.], [[0., 1.]], [0.], scale=scale)


@pytest.mark.parametrize('lower, upper', [
    ([0.], [1.]), ([-1.], [1.]), ([np.nan], [2.]), ([np.inf], [np.inf]),
    ([2.], [1.]), ([1.], [np.nan]), ([1.], [-np.inf]),
    ([], []), ([1., 2.], [2., 3.]), ([[1.]], [[2.]]),
    ([1. + 1j], [2.]), ([1.], [2. + 1j]),
])
def test_invalid_observation_bounds_raise_value_error(lower, upper):
    with pytest.raises(ValueError):
        _math().mixture_observation_terms(lower, upper, [[0., 1.]], [0.])


def test_matching_pandas_indexes_and_array_inputs_are_supported():
    index = pd.Index(['second', 'first'])
    lower = pd.Series([1., 2.], index=index)
    upper = pd.Series([3., np.inf], index=index)
    locations = pd.DataFrame([[0., 1.], [1., 2.]], index=index)
    gate = pd.Series([0.2, -0.4], index=index)
    actual = _math().mixture_observation_terms(lower, upper, locations, gate)
    expected = _math().mixture_observation_terms(lower.to_numpy(), upper.to_numpy(),
                                               locations.to_numpy(), gate.to_numpy())
    for result, reference in zip(actual, expected):
        assert_array_equal(result, reference)
    assert_array_equal(_math().mixture_median(locations, gate),
                       _math().mixture_median(locations.to_numpy(), gate.to_numpy()))


@pytest.mark.parametrize('position', [0, 1, 2, 3])
def test_observation_rejects_pandas_index_reordering(position):
    index = pd.Index(['first', 'second'])
    values = [pd.Series([1., 2.], index=index), pd.Series([3., 4.], index=index),
              pd.DataFrame([[0., 1.], [1., 2.]], index=index),
              pd.Series([0.2, -0.4], index=index)]
    values[position] = values[position].iloc[::-1]
    with pytest.raises(ValueError, match='index'):
        _math().mixture_observation_terms(*values)


def test_median_rejects_location_and_gate_index_reordering():
    locations = pd.DataFrame([[0., 1.], [1., 2.]], index=['first', 'second'])
    gate = pd.Series([0., 0.], index=['second', 'first'])
    with pytest.raises(ValueError, match='index'):
        _math().mixture_median(locations, gate)
