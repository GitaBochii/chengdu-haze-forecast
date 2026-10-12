"""Support projection for eligible hourly-grid recovery predictions only."""

import numpy as np


def enforce_hourly_support(predictions, eligible):
    """Enforce T >= 1 only when the caller establishes hourly origin eligibility.

    Not applicable to subhourly targets or already-clean/noneligible snapshots.
    No labels, learned threshold, upper cap, or rounding are used.
    """
    p = np.asarray(predictions, dtype=float)
    mask = np.asarray(eligible)
    if (p.ndim != 1 or mask.shape != p.shape or mask.dtype != np.bool_
            or not np.isfinite(p).all() or (p < 0).any()):
        raise ValueError('Require finite nonnegative predictions and aligned boolean eligibility')
    return np.where(mask, np.maximum(p, 1.), p)
