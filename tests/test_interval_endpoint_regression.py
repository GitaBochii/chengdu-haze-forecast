import numpy as np
import pandas as pd
import pytest

from haze_forecast.interval_endpoint_regression import (
    expand_interval_endpoints,
    make_endpoint_model,
)


def test_pairs_preserve_features_order_and_unit_total_weight():
    frame = pd.DataFrame({'x': [2.0, np.nan], 'z': [3.0, 4.0]}, index=[7, 9])
    original = frame.copy(deep=True)
    lower = pd.Series([10, 20], index=frame.index)
    upper = pd.Series([10, 40], index=frame.index)
    features, labels, weights = expand_interval_endpoints(frame, lower, upper)
    pd.testing.assert_frame_equal(features, frame.iloc[[0, 0, 1, 1]].reset_index(drop=True))
    np.testing.assert_array_equal(labels, [10, 10, 20, 40])
    np.testing.assert_array_equal(weights, [0.5] * 4)
    np.testing.assert_array_equal(weights.reshape(-1, 2).sum(axis=1), [1, 1])
    pd.testing.assert_frame_equal(frame, original)


def test_weighted_endpoint_loss_equals_distance_plus_constant():
    rng = np.random.default_rng(42)
    lower = rng.uniform(0.1, 100, 500)
    upper = lower + rng.uniform(0, 100, 500)
    upper[:100] = lower[:100]
    prediction = rng.uniform(-10, 250, 500)
    _, labels, weights = expand_interval_endpoints(pd.DataFrame({'x': lower}), lower, upper)
    loss = (np.abs(np.repeat(prediction, 2) - labels) * weights).reshape(-1, 2).sum(axis=1)
    distance = np.maximum.reduce([lower - prediction, prediction - upper, np.zeros(500)])
    np.testing.assert_allclose(loss - (upper - lower) / 2, distance, atol=1e-12)


@pytest.mark.parametrize(('lower', 'upper'), [
    ([0, 2], [3, 4]), ([-1, 2], [3, 4]), ([np.nan, 2], [3, 4]),
    ([1, 2], [np.inf, 4]), ([1, 2], [np.nan, 4]), ([1, 2], [0.5, 4]),
    ([1], [3, 4]), ([[1], [2]], [3, 4]),
])
def test_invalid_bounds_are_not_silently_dropped(lower, upper):
    with pytest.raises(ValueError):
        expand_interval_endpoints(pd.DataFrame({'x': [1, 2]}), lower, upper)


@pytest.mark.parametrize('which', ['lower', 'upper'])
def test_series_must_align_to_frame(which):
    frame = pd.DataFrame({'x': [1, 2]}, index=[10, 20])
    values = {'lower': [1, 2], 'upper': [3, 4]}
    values[which] = pd.Series(values[which], index=[20, 10])
    with pytest.raises(ValueError, match='align'):
        expand_interval_endpoints(frame, **values)


@pytest.mark.parametrize('frame', [
    pd.DataFrame({'x': []}), pd.DataFrame(index=[0, 1]),
    pd.DataFrame({'x': [1, np.inf]}), pd.DataFrame({'x': [1, -np.inf]}),
    pd.DataFrame([[1, 2], [3, 4]], columns=['x', 'x']),
    pd.DataFrame({'x': [1, 2]}, index=[0, 0]),
])
def test_invalid_features_are_rejected(frame):
    with pytest.raises(ValueError):
        expand_interval_endpoints(frame, [1] * len(frame), [2] * len(frame))


def test_fixed_factory_preserves_endpoint_pairs():
    model = make_endpoint_model(3484)
    params = model.get_params()
    assert params['objective'] == 'regression_l1'
    assert params['boosting_type'] == 'gbdt'
    assert params['subsample'] == 1.0 and params['subsample_freq'] == 0
    assert params['subsample_for_bin'] >= 2 * 3484
    assert params['min_child_samples'] == 32
    assert params['min_child_weight'] == 0.001
    assert params['n_estimators'] == 700
    assert params['deterministic'] and params['force_col_wise']


@pytest.mark.parametrize('rows', [0, -1, 1.5, True])
def test_factory_requires_positive_integer_population(rows):
    with pytest.raises(ValueError):
        make_endpoint_model(rows)


@pytest.mark.parametrize(('lower', 'upper'), [
    ([10, 30], [10, 30]), ([10, 20, 40], [60, 30, 40]),
])
def test_native_constant_leaf_belongs_to_interval_l1_optimal_set(lower, upper):
    frame = pd.DataFrame({'x': np.ones(len(lower))})
    features, labels, weights = expand_interval_endpoints(frame, lower, upper)
    model = make_endpoint_model(len(frame)).set_params(n_estimators=2)
    model.fit(features, labels, sample_weight=weights)
    prediction = model.predict(frame)
    ordered = np.sort(labels)
    left, right = ordered[len(lower) - 1:len(lower) + 1]
    assert np.all(prediction >= left - 1e-10)
    assert np.all(prediction <= right + 1e-10)
    loss = np.average(np.abs(labels - prediction[0]), weights=weights)
    optimum = np.average(np.abs(labels - (left + right) / 2), weights=weights)
    assert loss == pytest.approx(optimum)
