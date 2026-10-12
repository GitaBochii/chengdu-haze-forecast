"""Causal lengths of consecutive confirmed particle-pollution observations."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from haze_forecast.observed_business import build_business_states


def build_observed_pollution_run(frame):
    """Return observed run length and whether its hourly-grid start is known.

    Unknown observations break the observed run without establishing a clean
    hour. A left-boundary or post-unknown run therefore has an unknown start.
    Only the four state inputs are read; future labels and event IDs are ignored.
    """
    measurements = ['aqi', 'pm2_5_iaqi', 'pm10_iaqi']
    required = ['timestamp', *measurements]
    if (not isinstance(frame, pd.DataFrame) or frame.empty
            or not frame.columns.is_unique or not set(required).issubset(frame)):
        raise ValueError('Require nonempty state inputs with unique columns')
    timestamps = frame.timestamp
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype) or timestamps.isna().any():
        raise ValueError('Require timezone-aware nonmissing timestamps')
    differences = timestamps.diff().iloc[1:].to_numpy(dtype='timedelta64[ns]')
    if (not timestamps.is_unique or not timestamps.is_monotonic_increasing
            or not np.all(differences == np.timedelta64(1, 'h'))):
        raise ValueError('Require a complete ordered hourly grid without duplicates')
    if not (timestamps.dt.minute.eq(0) & timestamps.dt.second.eq(0)
            & timestamps.dt.microsecond.eq(0) & timestamps.dt.nanosecond.eq(0)).all():
        raise ValueError('Require whole-hour timestamps')
    for name in measurements:
        dtype = frame[name].dtype
        if not is_numeric_dtype(dtype) or is_complex_dtype(dtype) or is_bool_dtype(dtype):
            raise ValueError('Require real numeric measurements or missing values')
    values = frame[measurements].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError('Infinite measurements are invalid')
    inputs = frame[required].copy()
    inputs[measurements] = values
    states = build_business_states(inputs).pollution_state.to_numpy()
    lengths = np.full(len(frame), np.nan)
    known_start = np.full(len(frame), np.nan)
    previous, length, start_observed = None, 0, 0.0
    for position, state in enumerate(states):
        if state == 'confirmed_pollution':
            if previous == 'confirmed_pollution':
                length += 1
            else:
                length = 1
                start_observed = float(previous == 'confirmed_not_pollution')
            lengths[position] = length
            known_start[position] = start_observed
        else:
            length = 0
            if state == 'confirmed_not_pollution':
                lengths[position] = 0.0
        previous = state
    result = frame[['timestamp']].copy()
    result['observed_particle_run_hours'] = lengths
    result['particle_run_start_observed'] = known_start
    return result
