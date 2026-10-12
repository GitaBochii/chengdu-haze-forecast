"""Discrete hazards trained on exact events and observed censored prefixes."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from haze_forecast.discrete_survival import DiscreteHazardRegressor, _numeric_feature_matrix


def expand_bounds(features, lower, upper):
    if features.empty or not features.columns.is_unique:
        raise ValueError('nonempty unique feature columns required')
    x = _numeric_feature_matrix(features, list(features.columns))
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.shape != (len(x),) or hi.shape != lo.shape:
        raise ValueError('bounds shape mismatch')
    for series in (lower, upper):
        if isinstance(series, pd.Series) and not series.index.equals(features.index):
            raise ValueError('bounds index mismatch')
    if not np.isfinite(lo).all() or (lo < 1).any() or not np.equal(lo, np.rint(lo)).all():
        raise ValueError('positive integer lower bounds required')
    exact = np.isfinite(hi) & (hi == lo)
    if not (exact | np.isposinf(hi)).all():
        raise ValueError('only exact or right-censored bounds supported')
    # A lower bound L only establishes survival through hour L-1.
    count = np.where(exact, lo, lo - 1).astype(int)
    owner = np.repeat(np.arange(len(x)), count)
    start = np.repeat(np.cumsum(count) - count, count)
    horizon = np.arange(len(owner)) - start + 1
    result = x.iloc[owner].reset_index(drop=True)
    result['horizon_hours'] = horizon.astype(float)
    target = (exact[owner] & (horizon == lo[owner])).astype(int)
    return result, target, owner


class ObservedHazardRegressor(DiscreteHazardRegressor):
    def fit_bounds(self, features, lower, upper):
        expanded, target, _ = expand_bounds(features, lower, upper)
        if not len(target) or not np.any(target == 1):
            raise ValueError('at least one observed recovery is required')
        self.feature_names = list(features.columns)
        self.max_horizon_hours = int(expanded.horizon_hours.max())
        self.model = LGBMClassifier(objective='binary', n_estimators=300, learning_rate=.04,
            num_leaves=15, min_child_samples=40, subsample=.9, colsample_bytree=.8,
            reg_lambda=2., random_state=self.seed, n_jobs=1, verbosity=-1)
        self.model.fit(expanded, target)
        return self

    def predict(self, features):
        value, unresolved = self.predict_with_censoring(features)
        return np.where(unresolved, np.nan, value)
