import numpy as np
import pandas as pd
import pytest

from haze_forecast.observed_hazard import ObservedHazardRegressor, expand_bounds


def test_exact_and_censor_expansion():
    x = pd.DataFrame({'x': [10., 20., 30.]})
    expanded, target, owner = expand_bounds(x, [3, 3, 1], [3, np.inf, np.inf])
    assert expanded.horizon_hours.tolist() == [1, 2, 3, 1, 2]
    assert expanded.x.tolist() == [10, 10, 10, 20, 20]
    assert target.tolist() == [0, 0, 1, 0, 0]
    assert owner.tolist() == [0, 0, 0, 1, 1]


@pytest.mark.parametrize('lo,hi', [([0], [np.inf]), ([1.5], [np.inf]), ([2], [3]), ([2], [np.nan])])
def test_invalid_bounds(lo, hi):
    with pytest.raises(ValueError):
        expand_bounds(pd.DataFrame({'x': [1.]}), lo, hi)


def test_index_mismatch():
    with pytest.raises(ValueError, match='index'):
        expand_bounds(pd.DataFrame({'x': [1.]}), pd.Series([1], index=[2]), [1])


def test_unresolved_is_not_fabricated_median():
    model = ObservedHazardRegressor()
    model.predict_with_censoring = lambda x: (np.array([5., 5.]), np.array([False, True]))
    out = model.predict(pd.DataFrame({'x': [1, 2]}))
    assert out[0] == 5 and np.isnan(out[1])


def test_requires_observed_recovery():
    with pytest.raises(ValueError, match='recovery'):
        ObservedHazardRegressor().fit_bounds(pd.DataFrame({'x': [1.]}), [3], [np.inf])
