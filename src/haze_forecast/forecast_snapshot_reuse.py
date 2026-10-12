"""Reuse already-known forecast snapshots, never observations or future releases."""

import numpy as np
import pandas as pd


def reuse_forecast_snapshots(snapshots, origins, fields):
    if not fields or len(fields) != len(set(fields)):
        raise ValueError('unique forecast fields required')
    times = ['known_at', 'valid_at', 'run_at']
    if not set(times + fields).issubset(snapshots.columns):
        raise ValueError('snapshot metadata or fields missing')
    if set(fields) & set(times + ['timestamp']):
        raise ValueError('forecast fields cannot overwrite metadata')
    data = snapshots[times + fields].copy()
    for name in times:
        if not isinstance(data[name].dtype, pd.DatetimeTZDtype) or data[name].isna().any():
            raise ValueError('timezone-aware nonmissing snapshot times required')
        data[name] = data[name].dt.tz_convert('UTC')
    requested = pd.DatetimeIndex(origins)
    if requested.tz is None or requested.hasnans or not requested.is_unique:
        raise ValueError('unique timezone-aware nonmissing origins required')
    if data.known_at.duplicated().any():
        raise ValueError('ambiguous snapshots at the same known time')
    if (data.run_at > data.known_at).any() or (data.valid_at <= data.run_at).any():
        raise ValueError('invalid forecast chronology')
    numeric = data[fields].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('complete finite forecast snapshots required')
    data[fields] = numeric
    data = data.sort_values('known_at').reset_index(drop=True)
    known = pd.DatetimeIndex(data.known_at)
    records = []
    for origin in requested:
        selected = None
        stop = known.searchsorted(origin, side='right')
        for position in range(stop - 1, -1, -1):
            candidate = data.iloc[position]
            if candidate.valid_at > origin:
                selected = candidate
                break
        record = {'timestamp': origin, **{field: np.nan for field in fields},
                  **{name: pd.NaT for name in times}, 'forecast_known_age_hours': np.nan,
                  'forecast_run_age_hours': np.nan, 'forecast_remaining_lead_hours': np.nan}
        if selected is not None:
            record.update(selected.to_dict())
            record['forecast_known_age_hours'] = (origin - selected.known_at).total_seconds() / 3600
            record['forecast_run_age_hours'] = (origin - selected.run_at).total_seconds() / 3600
            record['forecast_remaining_lead_hours'] = (selected.valid_at - origin).total_seconds() / 3600
        records.append(record)
    result = pd.DataFrame(records, columns=['timestamp', *fields, *times,
        'forecast_known_age_hours', 'forecast_run_age_hours', 'forecast_remaining_lead_hours'])
    for name in times:
        result[name] = pd.to_datetime(result[name], utc=True)
    return result
