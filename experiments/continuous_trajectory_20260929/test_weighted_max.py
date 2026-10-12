import numpy as np
from weighted_max import training_weights


def test_weights_ignore_development_and_test_values():
    x = np.zeros((5, 323))
    x[:, 71] = [80, 90, 120, 500, 600]
    train = np.array([True, True, True, False, False])
    weights, info = training_weights(x, train)
    np.testing.assert_array_equal(weights, [.75, .75, 1.5])
    x[~train, 71] = 1
    changed, _ = training_weights(x, train)
    np.testing.assert_array_equal(changed, weights)
    assert info['clean_weight_mass'] == info['polluted_weight_mass']


def test_invalid_last_observation_keeps_training_origin():
    x = np.zeros((4, 323))
    x[:, 71] = [80, 120, np.nan, -1]
    weights, info = training_weights(x, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(weights, [1, 1, 1, 1])
    assert info['invalid_count'] == 2
