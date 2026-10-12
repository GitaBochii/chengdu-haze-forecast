import joblib
import numpy as np
import pandas as pd
import pytest

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor


def model_class():
    from haze_forecast.two_expert_aft import TwoExpertAFT

    return TwoExpertAFT


def example():
    x = np.linspace(-2., 2., 80)
    frame = pd.DataFrame({'x': x, 'z': np.where(x > 0, np.nan, x**2)})
    lower = pd.Series(np.exp(2. + .4 * x), index=frame.index)
    upper = lower.copy()
    upper.iloc[::4] *= 1.5
    return frame, lower, upper


def test_training_preserves_inputs_and_frozen_baseline_and_replays(tmp_path):
    frame, lower, upper = example()
    original = frame.copy(deep=True)
    baseline = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(frame, lower, upper)
    before = baseline.booster.save_raw(raw_format='json')
    model = model_class()(max_updates=2).fit_bounds(frame, lower, upper, baseline)
    pd.testing.assert_frame_equal(frame, original)
    assert before == baseline.booster.save_raw(raw_format='json')
    assert model.feature_names == list(frame.columns)
    assert 1 <= len(model.states) <= 3
    assert len(model.trace) == len(model.states)
    accepted = [row['mean_nll'] for row in model.trace if row['accepted']]
    assert np.all(np.diff(accepted) < 0)
    assert model.selected_index == max(i for i, row in enumerate(model.trace) if row['accepted'])
    assert np.allclose(model.predict_state(frame, 0), baseline.predict(frame), rtol=2e-7, atol=1e-6)
    prediction = model.predict(frame)
    assert prediction.shape == (80,) and np.isfinite(prediction).all() and (prediction > 0).all()
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(frame), prediction)
    assert all(np.isfinite(row['mean_nll']) for row in model.trace)
    for row in model.trace:
        assert np.isclose(sum(row['responsibility_mass']), len(frame))
    for state in model.states[1:]:
        assert len(state['experts']) == 2
        assert state['gate'].num_boosted_rounds() == 100
        assert all(expert.num_boosted_rounds() == 500 for expert in state['experts'])
        assert len(state['expert_training_configs']) == 2


@pytest.mark.parametrize('current,proposed,expected', [(3., 2., True), (3., 3., False), (3., 4., False)])
def test_update_acceptance_uses_only_strict_training_nll(current, proposed, expected):
    from haze_forecast.two_expert_aft import accept_training_update

    assert accept_training_update(current, proposed) is expected


@pytest.mark.parametrize('current,proposed', [(np.nan, 1.), (1., np.inf), (-np.inf, 0.)])
def test_update_acceptance_rejects_invalid_losses(current, proposed):
    from haze_forecast.two_expert_aft import accept_training_update

    with pytest.raises(ValueError):
        accept_training_update(current, proposed)


@pytest.mark.parametrize('updates', [0, -1, True, 1.5, '5'])
def test_invalid_update_count(updates):
    with pytest.raises(ValueError):
        model_class()(max_updates=updates)


def test_prediction_before_fit_is_rejected():
    with pytest.raises(ValueError):
        model_class()().predict(example()[0])


@pytest.mark.parametrize('invalid', ['lower_zero', 'reversed', 'wrong_index', 'infinite_feature',
                                   'duplicate_column', 'duplicate_index', 'empty', 'unfitted_baseline'])
def test_fit_does_not_repair_invalid_inputs(invalid):
    frame, lower, upper = example()
    baseline = ObservedAFTSurvivalRegressor(nthread=1).fit_bounds(frame, lower, upper)
    if invalid == 'lower_zero':
        lower.iloc[0] = 0
    elif invalid == 'reversed':
        upper.iloc[0] = lower.iloc[0] - 1
    elif invalid == 'wrong_index':
        lower.index = lower.index[::-1]
    elif invalid == 'infinite_feature':
        frame.loc[0, 'x'] = np.inf
    elif invalid == 'duplicate_column':
        frame.columns = ['x', 'x']
    elif invalid == 'duplicate_index':
        frame.index = [0] * len(frame)
    elif invalid == 'empty':
        frame = frame.iloc[:0]
    else:
        baseline = ObservedAFTSurvivalRegressor()
    with pytest.raises(ValueError):
        model_class()().fit_bounds(frame, lower, upper, baseline)


def test_responsibility_support_reports_mass_concentration_not_independent_samples():
    from scripts.benchmark_two_expert_aft import responsibility_support

    responsibility = np.array([[.5, .5], [.5, .5], [1., 0.]])
    result = responsibility_support([10, 10, 20], responsibility)
    np.testing.assert_allclose(result['effective_blocks'], [2., 1.])
    np.testing.assert_allclose(result['largest_block_fraction'], [.5, 1.])


def test_zero_responsibility_component_support_is_unavailable_not_nan():
    from scripts.benchmark_two_expert_aft import responsibility_support

    result = responsibility_support([10, 20], np.array([[1., 0.], [1., 0.]]))
    assert result['effective_blocks'] == [2., None]
    assert result['largest_block_fraction'] == [.5, None]


@pytest.mark.parametrize('responsibility,expected', [
    ([[.5, .5], [1., 0.]], True), ([[1., 0.], [1., 0.]], False),
    ([[1., 1e-100], [1., 1e-100]], False),
])
def test_component_weights_must_survive_native_float32_conversion(responsibility, expected):
    from haze_forecast.two_expert_aft import trainable_components

    assert trainable_components(np.array(responsibility)) is expected
