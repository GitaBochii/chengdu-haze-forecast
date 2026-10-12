"""Matched source ablations, without rebuilding event labels or pollution inputs."""

from __future__ import annotations

import pandas as pd

from haze_forecast.features import build_features

WEATHER_BASE = ('temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms')
WEATHER_FIELDS = tuple(name + suffix for name in WEATHER_BASE for suffix in
                       ('', '_lag_1', '_lag_3', '_lag_24', '_roll_3', '_roll_24', '_change_3'))
WEATHER_FIELDS += ('wind_speed_ms',)


def feature_arms(base_features: list[str]) -> dict[str, list[str]]:
    four = [name for name in base_features if not name.startswith('boundary_layer_height_m')]
    return {'baseline': list(base_features), 'mixed_four': four,
            'uniform_four': list(four),
            'pollution_only': [name for name in four if name not in WEATHER_FIELDS]}


def _times(values) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(values)
    if times.tz is None or times.hasnans:
        raise ValueError('weather alignment requires nonmissing timezone-aware timestamps')
    return times.tz_convert('UTC')


def uniform_weather_rows(rows: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """Generate causal weather on the full hourly grid, then match existing rows."""
    weather = proxy[['timestamp', *WEATHER_BASE]].copy()
    weather['timestamp'] = _times(weather.timestamp)
    weather = weather.sort_values('timestamp').reset_index(drop=True)
    if weather.empty or weather.timestamp.duplicated().any():
        raise ValueError('weather grid must be nonempty and unique')
    if not weather.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError('weather grid must be contiguous hourly; retain missing-hour rows')
    target_times = _times(rows.timestamp)
    if not target_times.isin(weather.timestamp).all():
        raise ValueError('weather grid does not cover every event timestamp')
    aligned = build_features(weather).set_index('timestamp').reindex(target_times)
    result = rows.copy()
    for field in WEATHER_FIELDS:
        result[field] = aligned[field].to_numpy()
    return result
