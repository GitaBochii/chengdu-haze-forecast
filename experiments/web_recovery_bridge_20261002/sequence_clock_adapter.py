"""Isolated 87-feature sequence adapter; does not replace production inference."""

import numpy as np
import pandas as pd

from haze_forecast.features import MEASUREMENT_COLUMNS, add_missingness_indicators
from haze_forecast.observed_hourly_features import build_observed_hourly_features


def sequence_feature_row(frame, origin, feature_names):
    if frame.empty or not frame.columns.is_unique or 'timestamp' not in frame:
        raise ValueError('nonempty unique measurement columns required')
    if len(feature_names) != 87 or len(set(feature_names)) != 87:
        raise ValueError('87 unique ordered features required')
    times = frame.timestamp
    if not isinstance(times.dtype, pd.DatetimeTZDtype):
        raise ValueError('timezone-aware measurements required')
    if times.isna().any() or times.duplicated().any():
        raise ValueError('unique nonmissing timestamps required')
    if not (times.dt.minute.eq(0) & times.dt.second.eq(0)
            & times.dt.microsecond.eq(0) & times.dt.nanosecond.eq(0)).all():
        raise ValueError('timestamps must be exact hours')
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin.tz is None or origin != origin.floor('h'):
        raise ValueError('timezone-aware hourly origin required')
    origin = origin.tz_convert('Asia/Shanghai')
    local_times = times.dt.tz_convert('Asia/Shanghai')
    if (local_times > origin).any():
        raise ValueError('future observations cannot enter inference')
    if local_times.max() != origin:
        raise ValueError('origin must be the latest observed timestamp')
    start = origin - pd.Timedelta(hours=24)
    if local_times.min() > start:
        raise ValueError('24-hour history span requires 25 hourly grid positions')
    required = {'aqi', 'pm2_5', 'pm10'}
    if not required.issubset(frame):
        raise ValueError('AQI and particle measurements required')
    measurements = frame[['timestamp', *[c for c in MEASUREMENT_COLUMNS if c in frame]]].copy()
    measurements['timestamp'] = local_times
    for name in MEASUREMENT_COLUMNS:
        if name not in measurements:
            measurements[name] = np.nan
        measurements[name] = pd.to_numeric(measurements[name], errors='raise')
        if np.isinf(measurements[name].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError('infinite measurements are invalid')
    # Reindex to actual clock positions before any lag or rolling calculation.
    grid = pd.date_range(start, origin, freq='h')
    measurements = measurements.set_index('timestamp').reindex(grid.rename('timestamp')).reset_index()
    full = build_observed_hourly_features(measurements)
    pbl = [name[:-11] for name in feature_names
           if name.startswith('boundary_layer_height_m') and name.endswith('_is_missing')
           and '_change_' not in name]
    full = add_missingness_indicators(full, pbl)
    full['boundary_layer_height_m_change_3_is_missing'] = (
        full.boundary_layer_height_m.isna() | full.boundary_layer_height_m_lag_3.isna()
    ).astype(float)
    if not set(feature_names).issubset(full):
        raise ValueError('unavailable model features')
    return full[feature_names].tail(1).reset_index(drop=True).astype(float)
