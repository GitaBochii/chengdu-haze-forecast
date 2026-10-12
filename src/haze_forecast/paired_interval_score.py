"""Bounds on candidate-minus-baseline absolute error with a shared true label.

Each row has sharp bounds over its own observation interval. Averaging these
bounds is conservative for any jointly compatible true recovery times. It need
not be sharp when origins from one process share additional timing constraints.
These are label-compatible bounds, not confidence intervals or exact MAE.
"""

import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions


def _validated_prediction(frame, prediction, name):
    if isinstance(prediction, pd.Series) and not prediction.index.equals(frame.index):
        raise ValueError(f'{name} Series index must equal the cohort index in order')
    score_predictions(frame, prediction)
    return np.asarray(prediction, dtype=float)


def paired_error_bounds(frame, baseline, candidate):
    """Return per-row lower/upper error differences; positive means worse.

Series must have exactly the cohort index and order; arrays use positions.
Both predictions must cover every row and satisfy the existing scorecard's
label, origin-identity, and prediction constraints. No rows are dropped.
"""
    baseline = _validated_prediction(frame, baseline, 'baseline')
    candidate = _validated_prediction(frame, candidate, 'candidate')
    lo = frame.lower_bound_hours.to_numpy(dtype=float)
    hi = frame.upper_bound_hours.to_numpy(dtype=float)
    prediction_min = np.minimum(baseline, candidate)
    prediction_max = np.maximum(baseline, candidate)

    # The difference is monotone in truth and constant outside the prediction
    # range. Clamping endpoints therefore also handles +inf without inf - inf.
    lo = np.clip(lo, prediction_min, prediction_max)
    hi = np.clip(hi, prediction_min, prediction_max)
    at_lo = abs(candidate - lo) - abs(baseline - lo)
    at_hi = abs(candidate - hi) - abs(baseline - hi)
    return pd.DataFrame({
        'lower': np.minimum(at_lo, at_hi),
        'upper': np.maximum(at_lo, at_hi),
    }, index=frame.index)


def paired_error_summary(frame, baseline, candidate):
    """Aggregate conservative paired bounds with row and process weighting.

Uncertain-subgroup process means include only uncertain origins within each
represented process. Empty subgroups return None. The shared-label convention
holds within each origin; joint timing restrictions between origins are not
used to tighten the aggregate bounds.
"""
    bounds = paired_error_bounds(frame, baseline, candidate)
    lower = bounds.lower.to_numpy(dtype=float)
    upper = bounds.upper.to_numpy(dtype=float)
    blocks = frame.continuity_block.to_numpy()
    hi = frame.upper_bound_hours.to_numpy(dtype=float)
    exact = frame.exact_hours.notna().to_numpy()
    uncertain = ~exact
    all_rows = np.ones(len(frame), dtype=bool)

    def mean(values, mask):
        return float(np.mean(values[mask])) if mask.any() else None

    def block_mean(values, mask):
        if not mask.any():
            return None
        return float(pd.Series(values[mask]).groupby(blocks[mask]).mean().mean())

    result = {
        'rows': len(frame),
        'blocks': int(pd.Series(blocks).nunique()),
        'prediction_coverage': 1.0,
        'exact_rows': int(exact.sum()),
        'exact_blocks': int(pd.Series(blocks[exact]).nunique()),
        'uncertain_rows': int(uncertain.sum()),
        'uncertain_blocks': int(pd.Series(blocks[uncertain]).nunique()),
        'finite_interval_rows': int((uncertain & np.isfinite(hi)).sum()),
        'right_censored_rows': int(np.isposinf(hi).sum()),
        'exact_row_delta': mean(lower, exact),
        'exact_block_delta': block_mean(lower, exact),
    }
    for name, mask in [('all', all_rows), ('uncertain', uncertain)]:
        for bound, values in [('lower', lower), ('upper', upper)]:
            result[f'{name}_row_delta_{bound}'] = mean(values, mask)
            result[f'{name}_block_delta_{bound}'] = block_mean(values, mask)
    return result
