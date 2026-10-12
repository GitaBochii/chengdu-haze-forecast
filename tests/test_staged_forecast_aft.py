import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from scripts.benchmark_staged_forecast_aft import head_inputs


def test_weather_control_preserves_mask_and_margin():
    frame = pd.DataFrame({'weather': [3., np.nan], 'age': [4., 6.]})
    before = frame.copy(deep=True)
    actual, supported = head_inputs(frame, ['weather', 'age'], np.array([1., 2.]), True)
    control, other = head_inputs(frame, ['weather', 'age'], np.array([1., 2.]), False)
    pd.testing.assert_frame_equal(frame, before)
    np.testing.assert_array_equal(supported, [True, False])
    np.testing.assert_array_equal(supported, other)
    np.testing.assert_array_equal(actual.isna(), control.isna())
    np.testing.assert_array_equal(control.base_log_hours, [1., 2.])
    assert control.weather.iloc[0] == control.age.iloc[0] == 0
    assert actual.weather.iloc[0] == 3


@pytest.mark.parametrize('margin', [[1.], [1., np.nan], [[1.], [2.]]])
def test_invalid_margins_rejected(margin):
    with pytest.raises(ValueError):
        head_inputs(pd.DataFrame({'w': [1., 2.]}), ['w'], margin, True)


def test_infinite_forecast_rejected():
    with pytest.raises(ValueError):
        head_inputs(pd.DataFrame({'w': [np.inf]}), ['w'], [1.], True)


def test_zero_round_aft_keeps_log_margin():
    features, _ = head_inputs(pd.DataFrame({'w': [1., 2.]}), ['w'], [1., 2.], True)
    matrix = xgb.DMatrix(features, base_margin=np.array([1., 2.]))
    matrix.set_float_info('label_lower_bound', [2., 3.])
    matrix.set_float_info('label_upper_bound', [2., 4.])
    model = xgb.train({'objective': 'survival:aft', 'nthread': 1}, matrix, num_boost_round=0)
    np.testing.assert_allclose(model.predict(matrix), np.exp([1., 2.]), rtol=1e-6)
