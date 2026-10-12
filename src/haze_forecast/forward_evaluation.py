"""Small helpers shared by strictly forward model experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def recency_sample_weights(
    timestamps: pd.Series,
    reference_time: pd.Timestamp,
    half_life_years: float,
) -> np.ndarray:
    """Down-weight older observations without using any future measurements.

    ``reference_time`` is the first timestamp of the held-out prediction year,
    rather than the end of the test set.  This keeps the weighting rule usable
    when a model is deployed at the beginning of that year.
    """
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    observed = pd.to_datetime(timestamps, errors="raise")
    reference = pd.Timestamp(reference_time)
    if observed.dt.tz is not None and reference.tz is None:
        reference = reference.tz_localize(observed.dt.tz)
    if observed.dt.tz is None and reference.tz is not None:
        reference = reference.tz_localize(None)
    age_years = np.maximum((reference - observed).dt.total_seconds().to_numpy() / 31_556_952, 0.0)
    return np.exp(-np.log(2.0) * age_years / half_life_years)
