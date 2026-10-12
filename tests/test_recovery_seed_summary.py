import numpy as np
import pytest

from scripts.summarize_recovery_seeds import ensemble_probabilities, validate_protocols


def test_equal_weight_distribution_not_average_of_point_predictions():
    first = np.array([[.6, .0, .4]])
    second = np.array([[.0, .6, .4]])
    probability, median = ensemble_probabilities([first, second])
    np.testing.assert_allclose(probability, [[.3, .3, .4]])
    assert median.tolist() == [2]


def test_tail_and_invalid_probabilities():
    _, median = ensemble_probabilities([np.array([[.1, .1, .8]]), np.array([[.0, .0, 1.]])])
    assert median.tolist() == [3]
    for arrays in ([np.array([[.1, .1]])], [np.array([[np.nan, .5]])],
                   [np.array([[-.1, 1.1]])], [np.ones((1, 2)) / 2, np.ones((2, 2)) / 2]):
        with pytest.raises(ValueError):
            ensemble_probabilities(arrays)


def test_seed_protocols_must_differ_only_in_seed_and_output():
    first = {'configuration': {'seed': 42, 'output': 'a', 'auxiliary_weight': .1},
             'inputs_before': {'x': 'hash'}, 'scaler': {'mean': [0]}, 'nodes': ['a'],
             'feature_names': ['aqi'], 'partitions': {'training': 10},
             'coordinates_lon_lat': [[1, 2]], 'reference_before': {'model': 'hash'},
             'deployment_before': {'model': 'deployed'}}
    second = {**first, 'configuration': {**first['configuration'], 'seed': 43, 'output': 'b'}}
    validate_protocols([first, second])
    with pytest.raises(ValueError):
        validate_protocols([first, first])
    with pytest.raises(ValueError):
        validate_protocols([first, {**second, 'configuration': {**second['configuration'], 'auxiliary_weight': .2}}])
    with pytest.raises(ValueError):
        validate_protocols([first, {**second, 'inputs_before': {'x': 'changed'}}])
