from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from haze_forecast.interval_hazard_em import IntervalHazardEM, soft_risk_targets


def test_soft_target_preserves_fractional_event_mass():
    target, weight = soft_risk_targets(np.array([.25, .2, 0]), np.array([1., .4, 0]))
    np.testing.assert_allclose(target, [.25, .5, 0])
    np.testing.assert_allclose(weight, [1., .4, 0])


def test_invalid_expected_counts_rejected():
    with pytest.raises(ValueError):
        soft_risk_targets(np.array([.6]), np.array([.5]))


def test_fit_and_median_are_valid():
    x = pd.DataFrame({'x': np.arange(20, dtype=float)})
    lo = np.tile([1., 2., 3., 4.], 5)
    hi = lo + np.tile([0., 1., 0., 2.], 5)
    model = IntervalHazardEM(rounds=10, max_em_steps=2).fit_bounds(x, lo, hi)
    assert model.horizon == 6
    accepted = [row['nll'] for row in model.trace if row['accepted']]
    assert all(b <= a for a, b in pairwise(accepted))
    hazards = model.predict_hazards(x)
    assert hazards.shape == (20, 6)
    assert np.isfinite(hazards).all() and (hazards > 0).all() and (hazards < 1).all()
    prediction = model.predict(x)
    assert ((prediction >= 1) | np.isnan(prediction)).all()


def test_cross_entropy_keeps_soft_labels():
    import lightgbm as lgb
    dataset = lgb.Dataset(np.arange(20.).reshape(-1, 1), label=np.full(20, .25))
    model = lgb.train({'objective': 'cross_entropy', 'verbosity': -1, 'num_threads': 1,
                       'min_data_in_leaf': 30}, dataset, num_boost_round=2)
    np.testing.assert_allclose(model.predict([[3.]], num_threads=1), [.25], atol=1e-7)


def test_rejected_update_preserves_previously_accepted_model(monkeypatch):
    import haze_forecast.interval_hazard_em as module

    class Candidate:
        def __init__(self, good):
            self.good = good

        def predict(self, values, num_threads=1):
            if self.good:
                return np.where(values[:, -1] == 10, .9, .01)
            return np.full(len(values), .99)

    first, second = Candidate(True), Candidate(False)
    candidates = iter([first, second])
    monkeypatch.setattr(module.lgb, 'train', lambda *args, **kwargs: next(candidates))
    features = pd.DataFrame({'x': [1., 2.]})
    model = IntervalHazardEM(max_em_steps=2).fit_bounds(features, [10., 10.], [10., 10.])
    assert [step['accepted'] for step in model.trace] == [True, True, False]
    assert model.model is first
    np.testing.assert_array_equal(model.predict(features), [10., 10.])
