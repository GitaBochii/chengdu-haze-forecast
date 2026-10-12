import numpy as np
import pytest

from haze_forecast.run_length_training import observed_transitions, recovery_median


def test_transition_state_is_before_observation_and_absorbs_at_six():
    h, state, target = observed_transitions([90] * 5 + [130] + [90] * 8)
    assert list(h) == list(range(1, 13))
    assert list(state) == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    assert list(target) == [1] * 5 + [0] + [1] * 6


def test_missing_stops_prefix_without_filling_or_resuming():
    h, state, target = observed_transitions([120, 90, np.nan, 90, 120])
    assert list(h) == [1, 2]
    assert list(state) == [0, 0]
    assert list(target) == [0, 1]
    assert len(observed_transitions([np.nan])[0]) == 0


def test_median_never_assigns_tail_a_fake_hour():
    result = recovery_median(np.array([[.2, .4, .4], [.1, .1, .8], [.5, 0, .5]]))
    np.testing.assert_allclose(result, [2, np.nan, 1], equal_nan=True)


@pytest.mark.parametrize('bad', [[np.inf], [[90, 120]]])
def test_invalid_observations(bad):
    with pytest.raises(ValueError):
        observed_transitions(bad)


@pytest.mark.parametrize('bad', [[[.2, .2]], [[-.1, 1.1]], [[np.nan, 1]], [1, 0]])
def test_invalid_distribution(bad):
    with pytest.raises(ValueError):
        recovery_median(bad)
