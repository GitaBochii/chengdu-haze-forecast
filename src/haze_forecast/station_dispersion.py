"""Contemporaneous station AQI distribution, with no future coverage selection."""

import numpy as np
import pandas as pd

SPATIAL_COLUMNS = ['station_aqi_q25_offset', 'station_aqi_q75_offset',
                   'station_aqi_polluted_fraction', 'station_aqi_count']


def station_dispersion(frame):
    if frame.empty or not {'timestamp', 'node_id', 'aqi'}.issubset(frame.columns):
        raise ValueError('Require nonempty station observations')
    if (not isinstance(frame.timestamp.dtype, pd.DatetimeTZDtype) or frame.timestamp.isna().any()
            or frame.node_id.isna().any() or frame.duplicated(['timestamp', 'node_id']).any()):
        raise ValueError('Require unique station/time observations with timezone')
    data = frame[['timestamp', 'node_id', 'aqi']].copy()
    data['aqi'] = pd.to_numeric(data.aqi, errors='raise')
    if np.isinf(data.aqi.to_numpy(float)).any():
        raise ValueError('Infinite AQI')
    grouped = data.groupby('timestamp', sort=True).aqi
    result = pd.DataFrame(index=grouped.count().index)
    median = grouped.median()
    result[SPATIAL_COLUMNS[0]] = grouped.quantile(.25) - median
    result[SPATIAL_COLUMNS[1]] = grouped.quantile(.75) - median
    count = grouped.count()
    polluted = data.aqi.gt(100).groupby(data.timestamp).sum()
    result[SPATIAL_COLUMNS[2]] = polluted / count.replace(0, np.nan)
    result[SPATIAL_COLUMNS[3]] = count
    return result.reset_index()
