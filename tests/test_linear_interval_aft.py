import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from haze_forecast.linear_interval_aft import LinearIntervalAFT


def test_exact_labels_match_analytical_log_ridge():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(120, 3)), columns=['x', 'y', 'z'])
    log_time = 3 + frame.to_numpy() @ np.array([0.8, -0.3, 0.1])
    time = np.exp(log_time)
    model = LinearIntervalAFT().fit_bounds(frame, time, time)
    matrix = model.transform(frame).to_numpy()
    weights = np.linalg.solve(
        matrix.T @ matrix + len(frame) * 1.2 ** 2 * np.eye(matrix.shape[1]),
        matrix.T @ (log_time - log_time.mean()),
    )
    expected = matrix @ weights + log_time.mean()
    np.testing.assert_allclose(model.predict_margin(frame), expected, atol=2e-5)
    np.testing.assert_allclose(model.predict(frame), np.exp(expected), rtol=2e-5)
    assert len(model.training_trace) == 51
    assert model.training_trace[0]['iteration'] == 949
    assert model.training_trace[-1]['iteration'] == 999
    assert model.parameters()['lambda'] == 1
    assert model.parameters()['updater'] == 'coord_descent'


def test_mixed_interval_balanced_inputs_match_scalar_likelihood():
    frame = pd.DataFrame({'x': [-1.] * 4 + [1.] * 4})
    lower = np.tile([5., 10., 20., 30.], 2)
    upper = np.tile([5., 20., 20., np.inf], 2)

    def objective(mu):
        return (-norm.logpdf((np.log(5) - mu) / 1.2)
                - np.log(norm.cdf((np.log(20) - mu) / 1.2)
                         - norm.cdf((np.log(10) - mu) / 1.2))
                - norm.logpdf((np.log(20) - mu) / 1.2)
                - norm.logsf((np.log(30) - mu) / 1.2))

    optimum = minimize_scalar(objective, bounds=(1, 5), method='bounded')
    assert optimum.success
    model = LinearIntervalAFT().fit_bounds(frame, lower, upper)
    np.testing.assert_allclose(model.predict_margin(frame), optimum.x, atol=2e-5)


def test_preprocessing_is_train_only_and_deduplicates_masks(tmp_path):
    frame = pd.DataFrame({
        'x': [1., np.nan, 3., np.nan],
        'existing_mask': [0., 1., 0., 1.],
        'all_missing': np.nan, 'constant': 2.,
    }, index=[4, 5, 6, 7])
    lower = pd.Series([10., 20., 30., 40.], index=frame.index)
    model = LinearIntervalAFT().fit_bounds(frame, lower, lower)
    assert model.kept_features == ['x', 'existing_mask']
    assert model.all_missing_features == ['all_missing']
    assert 'missingindicator_x' in model.dropped_design_features
    before = model.imputer.statistics_.copy()
    scaler_mean = model.scaler.mean_.copy()
    query = frame.copy()
    query['x'] = [np.nan, 100., 2., 4.]
    query['all_missing'] = 1000.
    transformed = model.transform(query)
    assert transformed.loc[4, 'x'] == pytest.approx(0.)
    np.testing.assert_array_equal(model.imputer.statistics_, before)
    np.testing.assert_array_equal(model.scaler.mean_, scaler_mean)
    path = tmp_path / 'linear.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(model.predict(query), joblib.load(path).predict(query))


@pytest.mark.parametrize(('lower', 'upper'), [
    ([0., 2.], [3., 4.]), ([-1., 2.], [3., 4.]),
    ([np.nan, 2.], [3., 4.]), ([1., 2.], [np.nan, 4.]),
    ([1., 2.], [0.5, 4.]), ([1.], [3., 4.]),
    ([1e40, 2.], [1e40, 4.]), ([1e-50, 2.], [3., 4.]),
])
def test_reject_invalid_or_unrepresentable_labels(lower, upper):
    with pytest.raises(ValueError):
        LinearIntervalAFT().fit_bounds(pd.DataFrame({'x': [1., 2.]}), lower, upper)


def test_series_alignment_is_required():
    frame = pd.DataFrame({'x': [1., 2.]}, index=[7, 8])
    with pytest.raises(ValueError, match='index'):
        LinearIntervalAFT().fit_bounds(frame, pd.Series([1., 2.], index=[8, 7]), [3., 4.])


@pytest.mark.parametrize('frame', [
    pd.DataFrame({'x': []}), pd.DataFrame({'x': [np.nan, np.nan]}),
    pd.DataFrame({'x': [1., np.inf]}), pd.DataFrame({'x': [2., 2.]}),
    pd.DataFrame([[1, 2], [3, 4]], columns=['x', 'x']),
])
def test_reject_unusable_features(frame):
    with pytest.raises(ValueError):
        LinearIntervalAFT().fit_bounds(frame, [1.] * len(frame), [2.] * len(frame))


def test_prediction_requires_fit_and_same_feature_order():
    frame = pd.DataFrame({'x': [1., 2.], 'y': [2., 3.]})
    with pytest.raises(ValueError, match='fitted'):
        LinearIntervalAFT().predict(frame)
    model = LinearIntervalAFT().fit_bounds(frame, [10., 20.], [10., 20.])
    with pytest.raises(ValueError, match='columns'):
        model.predict(frame[['y', 'x']])
