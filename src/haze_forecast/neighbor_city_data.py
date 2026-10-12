"""Unfilled historical city observations for isolated transport experiments."""

import io

import pandas as pd

CITIES = ['成都', '德阳', '眉山', '资阳', '雅安', '绵阳', '遂宁', '乐山']
POLLUTANTS = ['AQI', 'PM2.5', 'PM10']


def required_days(origins):
    if origins.dt.tz is None or origins.isna().any():
        raise ValueError('origins must have complete timezone-aware timestamps')
    source = origins.dt.tz_convert('Asia/Shanghai') - pd.Timedelta(hours=48)
    return sorted(source.dt.strftime('%Y%m%d').unique().tolist())


def parse_city_day(raw, date, cities=None):
    cities = CITIES if cities is None else cities
    frame = pd.read_csv(io.BytesIO(raw))
    if frame.empty or not {'date', 'hour', 'type'}.issubset(frame):
        raise ValueError('missing daily observations or schema')
    if set(frame.date.astype(str)) != {date}:
        raise ValueError('unexpected observation date')
    hour = pd.to_numeric(frame.hour, errors='raise')
    if not hour.between(0, 23).all() or not hour.mod(1).eq(0).all():
        raise ValueError('invalid observation hour')
    if frame.duplicated(['date', 'hour', 'type']).any():
        raise ValueError('duplicate hourly pollutant')
    result = pd.DataFrame({'timestamp': pd.date_range(
        pd.Timestamp(date), periods=24, freq='h', tz='Asia/Shanghai')})
    for city in cities:
        for pollutant in POLLUTANTS:
            selected = frame.loc[frame.type.eq(pollutant)].set_index('hour')
            values = (pd.to_numeric(selected[city], errors='raise')
                      if city in selected else pd.Series(dtype=float))
            if (values.dropna() < 0).any():
                raise ValueError('negative concentration')
            result[f'{city}_{pollutant}'] = values.reindex(range(24)).to_numpy()
    return result
