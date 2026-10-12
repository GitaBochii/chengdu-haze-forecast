"""Conservative GHCNh source-223 adapter for Shuangliu weather continuity."""

import numpy as np
import pandas as pd

from haze_forecast.cnemc_weather import _relative_humidity_percent


def prepare_ghcnh_weather(raw):
    if raw.empty or not raw.STATION.eq('CHI0000ZUUU').all():
        raise ValueError('nonempty Shuangliu station data required')
    observed = pd.to_datetime(raw.DATE, utc=True, errors='raise').dt.tz_convert('Asia/Shanghai')
    if observed.isna().any() or observed.duplicated().any():
        raise ValueError('unique valid observation times required')

    def value(name, low, high):
        numeric = pd.to_numeric(raw[name], errors='coerce')
        trusted = raw[name + '_Source_Code'].astype('string').eq('223')
        trusted &= raw[name + '_Quality_Code'].astype('string').eq('1')
        return numeric.where(trusted.fillna(False) & numeric.between(low, high))

    temperature = value('temperature', -93.2, 61.8)
    dewpoint = value('dew_point_temperature', -100, 70)
    direction = value('wind_direction', 0, 360)
    speed = value('wind_speed', 0, 100)
    radians = np.radians(direction)
    result = pd.DataFrame({'timestamp': observed.dt.ceil('h'), 'observed_at': observed,
        'temperature_c': temperature,
        'relative_humidity_pct': _relative_humidity_percent(temperature, dewpoint),
        'wind_u_ms': -speed * np.sin(radians), 'wind_v_ms': -speed * np.cos(radians)})
    columns = ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']
    result['valid_count'] = result[columns].notna().sum(axis=1)
    result = result.sort_values(['timestamp', 'valid_count', 'observed_at']).drop_duplicates('timestamp', keep='last')
    return result[['timestamp', 'observed_at', *columns]].reset_index(drop=True)
