"""Paired-cohort scores without inventing exact times for interval labels."""

import numpy as np
import pandas as pd


def score_predictions(frame, prediction):
    p = np.asarray(prediction, dtype=float)
    if frame.empty or p.shape != (len(frame),):
        raise ValueError('Require a nonempty aligned cohort')
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Missing or invalid predictions: do not drop origins')
    if frame.timestamp.isna().any() or not frame.timestamp.is_unique:
        raise ValueError('Require unique nonmissing origin timestamps')
    if frame.continuity_block.isna().any():
        raise ValueError('Missing process identity')
    lo = frame.lower_bound_hours.to_numpy(dtype=float)
    hi = frame.upper_bound_hours.to_numpy(dtype=float)
    exact = frame.exact_hours.to_numpy(dtype=float)
    is_exact = ~np.isnan(exact)
    if not np.isfinite(lo).all() or (lo <= 0).any() or np.isnan(hi).any() or (hi < lo).any():
        raise ValueError('Invalid observation bounds')
    if not np.array_equal(is_exact, lo == hi) or not np.array_equal(exact[is_exact], lo[is_exact]):
        raise ValueError('Exact labels must equal both bounds')
    blocks = frame.continuity_block.to_numpy()
    lower = np.maximum.reduce([lo - p, p - hi, np.zeros(len(p))])
    upper = np.maximum(abs(lo - p), abs(hi - p))
    tail = is_exact & (exact >= 24)
    uncertain = ~is_exact

    def mean(values):
        return float(np.mean(values)) if len(values) else None

    def block_mean(values, mask):
        if not mask.any():
            return None
        return float(pd.Series(values[mask]).groupby(blocks[mask]).mean().mean())

    all_rows = np.ones(len(p), dtype=bool)
    finite_upper = np.isfinite(hi).all()
    return {
        'rows': len(p), 'blocks': int(pd.Series(blocks).nunique()),
        'prediction_coverage': 1.0, 'exact_rows': int(is_exact.sum()),
        'exact_blocks': int(pd.Series(blocks[is_exact]).nunique()),
        'finite_interval_rows': int((uncertain & np.isfinite(hi)).sum()),
        'right_censored_rows': int(np.isposinf(hi).sum()),
        'exact_row_mae': mean(lower[is_exact]),
        'exact_block_mae': block_mean(lower, is_exact),
        'tail24_rows': int(tail.sum()), 'tail24_block_mae': block_mean(lower, tail),
        'all_row_error_lower': mean(lower),
        'all_row_error_upper': mean(upper) if finite_upper else None,
        'all_block_error_lower': block_mean(lower, all_rows),
        'all_block_error_upper': block_mean(upper, all_rows) if finite_upper else None,
        'uncertain_blocks': int(pd.Series(blocks[uncertain]).nunique()),
        'uncertain_row_distance': mean(lower[uncertain]),
        'uncertain_block_distance': block_mean(lower, uncertain),
        'uncertain_below_fraction': mean((p < lo)[uncertain]),
    }
