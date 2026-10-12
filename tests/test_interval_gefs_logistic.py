import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_interval_gefs_logistic import fit_model


def test_logistic_uses_interval_bounds_and_positive_outputs():
    frame = pd.DataFrame({'x': np.arange(8.), 'lower_bound_hours': np.arange(1., 9.),
                          'upper_bound_hours': np.arange(2., 10.)})
    model = fit_model(frame, ['x'], 'logistic', rounds=2)
    import xgboost as xgb
    matrix = xgb.DMatrix(frame[['x']])
    result = model.predict(matrix)
    assert np.isfinite(result).all() and (result > 0).all()
    np.testing.assert_allclose(result, np.exp(model.predict(matrix, output_margin=True)), rtol=2e-7)


def test_invalid_distribution_is_rejected():
    with pytest.raises(ValueError):
        fit_model(pd.DataFrame(), [], 'extreme')
