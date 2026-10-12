import numpy as np
import pytest

from haze_forecast.interval_hazard_likelihood import expectation, median_hours


def test_finite_interval_enumeration():
    logprob, events, risk = expectation([[.2, .3, .4]], [2], [3])
    mass = np.array([0., .8 * .3, .8 * .7 * .4])
    np.testing.assert_allclose(logprob, np.log(mass.sum()))
    np.testing.assert_allclose(events[0], mass / mass.sum())
    np.testing.assert_allclose(risk[0], [1., 1., mass[2] / mass.sum()])


def test_exact_and_right_censored_off_by_one():
    logprob, events, risk = expectation([[.2, .3, .4]] * 2, [2, 3], [2, np.inf])
    np.testing.assert_allclose(logprob, np.log([.8 * .3, .8 * .7]))
    np.testing.assert_allclose(events, [[0, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(risk, [[1, 1, 0], [1, 1, 0]])


def test_right_censor_at_first_hour_is_uninformative():
    logprob, events, risk = expectation([[.2, .3]], [1], [np.inf])
    np.testing.assert_array_equal(logprob, [0])
    np.testing.assert_array_equal(events, [[0, 0]])
    np.testing.assert_array_equal(risk, [[0, 0]])


def test_underflow_does_not_destroy_likelihood_or_posterior():
    logprob, events, risk = expectation(np.full((1, 2000), .5), [1999], [2000])
    assert np.isfinite(logprob).all()
    np.testing.assert_allclose(events[0, -2:], [2 / 3, 1 / 3])
    np.testing.assert_allclose(risk[0, :1999], 1)


def test_unresolved_median_is_not_horizon_cap():
    result = median_hours([[.2, .3, .4], [.01, .01, .01], [.5, .1, .1]])
    np.testing.assert_allclose(result, [3., np.nan, 1.], equal_nan=True)


@pytest.mark.parametrize('lo,hi', [([0], [2]), ([1.5], [2]), ([2], [1]),
                                   ([1], [4]), ([5], [np.inf]), ([1], [np.nan])])
def test_invalid_bounds_rejected(lo, hi):
    with pytest.raises(ValueError):
        expectation([[.2, .3, .4]], lo, hi)


@pytest.mark.parametrize('hazards', [[[0., .3]], [[1., .3]], [[np.nan, .3]], [], [.2, .3]])
def test_invalid_hazard_rejected(hazards):
    with pytest.raises(ValueError):
        expectation(hazards, [1], [1])


def test_expected_complete_log_likelihood_identity():
    h = np.array([[.2, .3, .4]])
    logprob, events, risk = expectation(h, [2], [3])
    positive = events > 0
    entropy = -np.sum(events[positive] * np.log(events[positive]))
    complete = np.sum(events * np.log(h) + (risk - events) * np.log1p(-h))
    assert complete + entropy == pytest.approx(logprob[0])


def test_em_counts_match_observed_likelihood_gradient():
    from scipy.special import expit
    logits = np.array([[-1., .3, -.4], [.5, -.7, .2]])
    h = expit(logits)
    _, events, risk = expectation(h, [2, 3], [3, np.inf])
    analytical = risk * h - events
    numerical = np.zeros_like(logits)
    for i, j in np.ndindex(logits.shape):
        step = np.zeros_like(logits)
        step[i, j] = 1e-5
        plus = expectation(expit(logits + step), [2, 3], [3, np.inf])[0]
        minus = expectation(expit(logits - step), [2, 3], [3, np.inf])[0]
        numerical[i, j] = -(plus.sum() - minus.sum()) / 2e-5
    np.testing.assert_allclose(analytical, numerical, atol=1e-9)


def test_right_censor_just_after_horizon():
    logprob, events, risk = expectation([[.2, .3]], [3], [np.inf])
    np.testing.assert_allclose(logprob, np.log([.8 * .7]))
    np.testing.assert_array_equal(events, [[0, 0]])
    np.testing.assert_array_equal(risk, [[1, 1]])
