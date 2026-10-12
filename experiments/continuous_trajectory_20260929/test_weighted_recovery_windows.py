import numpy as np
import pytest
from weighted_recovery_windows import regime_weights


def test_equal_mass_preserves_all_dates_and_mean_one():
    last = np.array([50.,90.,100.,101.])
    weights, info = regime_weights(last)
    np.testing.assert_allclose(weights, [2/3,2/3,2/3,2.])
    assert len(weights) == len(last)
    assert weights.mean() == 1
    assert info['clean_weight_mass'] == info['polluted_weight_mass'] == 2


def test_missing_or_negative_history_retains_neutral_weight():
    weights, info = regime_weights(np.array([50.,120.,np.nan,-1.,np.inf]))
    np.testing.assert_array_equal(weights,np.ones(5))
    assert info['invalid_count'] == 3
    assert info['finite_count'] == 2


def test_weights_fitted_only_to_supplied_training_population():
    training = np.array([50.,90.,120.])
    original, _ = regime_weights(training)
    evaluation = np.array([999.]*100)
    evaluation[:] = 10
    again, _ = regime_weights(training)
    np.testing.assert_array_equal(original,again)
    with pytest.raises(ValueError):
        regime_weights(np.array([20.,30.,np.nan]))
