"""Explicit observed bounds for AFT without changing the legacy fit path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.training import AFTSurvivalRegressor


class ObservedAFTSurvivalRegressor(AFTSurvivalRegressor):
    """Fit exact, interval, or right-censored labels with legacy AFT settings.

    Inherited ``fit`` still has the original exact-label/clamping behavior.
    ``fit_bounds`` does not infer censoring or enforce publication-time safety;
    callers must construct observations and filter label availability upstream.
    Array inputs are positional; Series must match the feature index exactly.
    """

    def fit_bounds(
        self,
        features: pd.DataFrame,
        lower_bound_hours: np.ndarray | pd.Series,
        upper_bound_hours: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> ObservedAFTSurvivalRegressor:
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("features must be a nonempty DataFrame with columns")
        numeric = features.astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("features must be finite or NaN, not infinite")

        def vector(values, name):
            if isinstance(values, pd.Series) and not values.index.equals(features.index):
                raise ValueError(f"{name} index must match features index")
            result = np.asarray(values, dtype=float)
            if result.shape != (len(features),):
                raise ValueError(f"{name} must have one entry per row")
            return result

        lower = vector(lower_bound_hours, "lower_bound_hours")
        upper = vector(upper_bound_hours, "upper_bound_hours")
        if not np.isfinite(lower).all() or (lower <= 0).any():
            raise ValueError("lower_bound_hours must be finite and strictly positive")
        if np.isnan(upper).any() or np.isneginf(upper).any() or (upper < lower).any():
            raise ValueError("upper_bound_hours must be >= lower, finite or positive infinity")
        # DMatrix stores bounds as float32; overflow must not create censoring.
        with np.errstate(over="ignore", under="ignore"):
            lower32 = lower.astype(np.float32)
            upper32 = upper.astype(np.float32)
        if (
            not np.isfinite(lower32).all()
            or (lower32 <= 0).any()
            or not np.array_equal(np.isfinite(upper32), np.isfinite(upper))
        ):
            raise ValueError("bounds must remain positive and finite where required in float32")
        weight = None
        if sample_weight is not None:
            weight = vector(sample_weight, "sample_weight")
            if not np.isfinite(weight).all() or (weight < 0).any() or not (weight > 0).any():
                raise ValueError("sample_weight must be finite, non-negative with positive sum")

        matrix = xgb.DMatrix(numeric, missing=np.nan)
        matrix.set_float_info("label_lower_bound", lower)
        matrix.set_float_info("label_upper_bound", upper)
        if weight is not None:
            matrix.set_weight(weight)
        params: dict[str, float | int | str] = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.2,
            "eta": 0.05,
            "max_depth": 4,
            "min_child_weight": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "lambda": 3.0,
            "seed": self.seed,
            "tree_method": "hist",
            "verbosity": 0,
        }
        if self.nthread is not None:
            params["nthread"] = self.nthread
        self.booster = xgb.train(params, matrix, num_boost_round=self.num_boost_round)
        return self
