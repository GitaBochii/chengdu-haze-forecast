"""Train-only spatial compression and interval-aware pilot summaries."""
from collections import Counter

import numpy as np
import torch


def fit_spatial_pca(values, train):
    if (values.ndim != 2 or values.shape[1] < 4 or train.shape != values.shape[:1]
            or train.dtype != torch.bool or int(train.sum()) < 5
            or not torch.isfinite(values).all()):
        raise ValueError('finite features and at least five training rows required')
    x = values[train].double().cpu()
    mean = x.mean(0)
    _, _, vectors = torch.linalg.svd(x - mean, full_matrices=False)
    projection = vectors[:4].T.contiguous()
    # Fix the arbitrary SVD sign for reproducible saved transforms.
    pivots = projection.abs().argmax(0)
    signs = projection[pivots, torch.arange(4)].sign()
    projection *= signs
    scores = (x - mean) @ projection
    scale = scores.std(0, correction=0)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return dict(mean=mean, projection=projection, scale=scale)


def transform_spatial(values, state):
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError('finite feature matrix required')
    return (((values.double().cpu() - state['mean']) @ state['projection'])
            / state['scale']).float()


def block_weights(blocks):
    if not len(blocks):
        raise ValueError('at least one block required')
    count = Counter(blocks)
    return torch.tensor([1 / (len(count) * count[b]) for b in blocks], dtype=torch.float32)


def error_summary(rows, predictions):
    p = np.asarray(predictions, dtype=float)
    if not rows or p.shape != (len(rows),) or not np.isfinite(p).all() or ((p < 1) | (p > 68)).any():
        raise ValueError('one finite capped prediction per row required')
    low = np.array([r['lower_bound_hours'] for r in rows], dtype=float)
    high = np.array([r['upper_bound_hours'] for r in rows], dtype=float)
    if (not np.isfinite(low).all() or (low < 1).any() or np.isnan(high).any()
            or (high < low).any()):
        raise ValueError('invalid original label bounds')
    low, high = np.minimum(low, 68), np.minimum(high, 68)
    lower = np.maximum.reduce([low - p, p - high, np.zeros_like(p)])
    upper = np.maximum(np.abs(p-low), np.abs(p-high))
    blocks = [r['continuity_block'] for r in rows]
    weights = block_weights(blocks).double().numpy()
    weights /= weights.sum()
    exact = np.array([r['reason'] == 'exact' for r in rows])
    return dict(row_count=len(rows), block_count=len(set(blocks)),
                original_exact_count=int(exact.sum()),
                exact_row_mae_capped_hours=float(lower[exact].mean()) if exact.any() else None,
                row_error_lower_hours=float(lower.mean()), row_error_upper_hours=float(upper.mean()),
                block_error_lower_hours=float(weights @ lower),
                block_error_upper_hours=float(weights @ upper))
