"""Finite-interval absolute risk via paired endpoints and native LightGBM L1."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def expand_interval_endpoints(frame, lower, upper):
    """Return identical feature pairs, endpoint labels, and half-row weights.

    The weighted endpoint L1 loss is distance to [lower, upper] plus half its
    width. This is not a likelihood or an imputation of two exact observations.
    Infinite upper bounds are unsupported, rather than silently discarded.
    """
    if (not isinstance(frame, pd.DataFrame) or frame.empty
            or not frame.columns.is_unique or not frame.index.is_unique):
        raise ValueError('Require nonempty features with unique columns and row index')
    features = frame.astype(float)
    if np.isinf(features.to_numpy()).any():
        raise ValueError('Infinite features are not supported; NaN remains missing')
    bounds = []
    for value in (lower, upper):
        if isinstance(value, pd.Series) and not value.index.equals(frame.index):
            raise ValueError('Label Series must align with the feature index')
        array = np.asarray(value, dtype=float)
        if array.shape != (len(frame),) or not np.isfinite(array).all():
            raise ValueError('Require aligned finite interval bounds; do not drop rows')
        bounds.append(array)
    lo, hi = bounds
    if (lo <= 0).any() or (hi < lo).any():
        raise ValueError('Require 0 < lower <= upper')
    paired = features.iloc[np.repeat(np.arange(len(frame)), 2)].reset_index(drop=True)
    labels = np.column_stack((lo, hi)).ravel()
    return paired, labels, np.full(len(labels), 0.5)


def make_endpoint_model(original_rows):
    """Fixed project L1 configuration, adjusted only to keep row pairs intact."""
    if (isinstance(original_rows, (bool, np.bool_))
            or not isinstance(original_rows, (int, np.integer)) or original_rows <= 0):
        raise ValueError('Require a positive integer original row count')
    return LGBMRegressor(
        objective='regression_l1', boosting_type='gbdt',
        n_estimators=700, learning_rate=0.04, num_leaves=24,
        min_child_samples=32, min_child_weight=0.001,
        subsample=1.0, subsample_freq=0, subsample_for_bin=2 * int(original_rows),
        colsample_bytree=0.8, reg_lambda=3.0, random_state=42,
        n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True,
    )
