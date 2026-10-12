import itertools

import numpy as np
import pytest

from haze_forecast.run_length_recovery import recovery_start_distribution


def test_six_clean_hours_start_at_one_not_confirmation_six():
    result = recovery_start_distribution(np.ones((1, 8, 6)))
    np.testing.assert_array_equal(result, [[1, 0, 0, 0]])


def test_five_clean_then_rebound_resets_counter():
    clean = np.array([1] * 5 + [0] + [1] * 6, dtype=float)
    result = recovery_start_distribution(np.broadcast_to(clean[None, :, None], (1, 12, 6)))
    assert result.shape == (1, 8)
    assert result[0, 6] == 1
    assert result.sum() == 1


def test_no_clean_path_remains_in_tail():
    np.testing.assert_array_equal(recovery_start_distribution(np.zeros((1, 6, 6))), [[0, 1]])


def test_dynamic_program_matches_exhaustive_state_dependent_paths():
    q = np.random.default_rng(42).uniform(.05, .95, (2, 8, 6))
    expected = np.zeros((2, 4))
    for batch in range(2):
        for path in itertools.product((0, 1), repeat=8):
            probability, run, first_start = 1., 0, None
            for t, clean in enumerate(path):
                if first_start is not None:
                    # Absorbed suffix is marginalized with an arbitrary fair coin.
                    probability *= .5
                    continue
                p = q[batch, t, run]
                probability *= p if clean else 1 - p
                run = run + 1 if clean else 0
                if run == 6:
                    first_start = t - 5
            expected[batch, -1 if first_start is None else first_start] += probability
    actual = recovery_start_distribution(q)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-12)


@pytest.mark.parametrize('value', [np.nan, np.inf, -.1, 1.1])
def test_invalid_or_unknown_probabilities_are_not_imputed(value):
    q = np.full((1, 6, 6), .5)
    q[0, 0, 0] = value
    with pytest.raises(ValueError):
        recovery_start_distribution(q)


@pytest.mark.parametrize('shape', [(6, 6), (0, 6, 6), (1, 5, 6), (1, 6, 5)])
def test_invalid_dimensions(shape):
    with pytest.raises(ValueError):
        recovery_start_distribution(np.zeros(shape))


def test_long_horizon_probability_conservation():
    q = np.random.default_rng(7).uniform(0, 1, (3, 359, 6))
    result = recovery_start_distribution(q)
    assert result.shape == (3, 355)
    assert (result >= 0).all()
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-12)
