"""Causal observation-gap descriptors without imputing weather measurements."""

import numpy as np
import pandas as pd

from haze_forecast.weather_source_ablation import WEATHER_BASE


def weather_observation_age(frame):
    times = frame.timestamp
    if (frame.empty or not isinstance(times.dtype, pd.DatetimeTZDtype) or times.isna().any()
            or times.duplicated().any() or not times.is_monotonic_increasing
            or not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()):
        raise ValueError('Require a unique, ordered, contiguous timezone-aware hourly grid')
    result = frame[['timestamp']].copy()
    for column in WEATHER_BASE:
        values = frame[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('Infinite measurement')
        observed = frame[column].notna()
        # Carry observation timestamps only, never weather values or future rows.
        latest = times.where(observed).ffill()
        result[f'{column}_observation_age_h'] = (times - latest).dt.total_seconds() / 3600
        result[f'{column}_unobserved'] = (~observed).astype(float)
    return result
