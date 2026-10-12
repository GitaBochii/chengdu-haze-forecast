import numpy as np
import pytest

from haze_forecast.regional_air_scoring import (
    active_node_mask,
    aggregate_active_predictions,
    verified_recovery_mask,
)


def test_activity_only_uses_current_and_past_per_channel():
    raw = np.full((5, 2, 2), np.nan)
    raw[0, 0, 0] = 10
    raw[1, 0, 1] = 20
    raw[3, 1] = 99
    result = active_node_mask(raw, np.array([1, 2]), lookback=2)
    np.testing.assert_array_equal(result[0], [[True, True], [False, False]])
    np.testing.assert_array_equal(result[1], [[False, True], [False, False]])


def test_aggregation_excludes_offline_nodes_and_retains_channel_masks():
    prediction = np.array([[[[10, 50], [30, 70], [999, 999]],
                             [[20, 60], [40, 80], [999, 999]]]], dtype=float)
    active = np.array([[[True, False], [True, False], [False, False]]])
    result = aggregate_active_predictions(prediction, active)
    np.testing.assert_array_equal(result[0, :, 0], [20, 30])
    assert np.isnan(result[0, :, 1]).all()


def test_verified_recovery_needs_confirmation_not_later_observations():
    future = np.ones((6, 72))
    future[0, 6] = np.nan  # After hour 1's six-hour confirmation.
    future[1, 5] = np.nan  # Inside hour 1's confirmation.
    future[2, 71] = np.nan  # Hour 67 confirmation needs hour 72.
    future[3, 71] = np.nan  # Long event requires the complete capped horizon.
    actual = np.array([1, 1, 67, 68, 100, np.nan])
    np.testing.assert_array_equal(
        verified_recovery_mask(actual, future), [True, False, False, False, True, False],
    )


def test_shape_and_bounds_fail_closed():
    with pytest.raises(ValueError):
        active_node_mask(np.ones((3, 2, 2)), np.array([-1]))
    with pytest.raises(ValueError):
        active_node_mask(np.ones((3, 2, 2)), np.array([3]))
    with pytest.raises(ValueError):
        aggregate_active_predictions(np.ones((1, 3, 2, 2)), np.ones((1, 3, 2)))
    with pytest.raises(ValueError):
        verified_recovery_mask(np.array([1]), np.ones((1, 5)))
    assert not verified_recovery_mask(np.array([0]), np.ones((1, 72))).item()
