import numpy as np
import pandas as pd
import pytest

from haze_forecast.ghcnh_weather import prepare_ghcnh_weather


def raw():
    f = pd.DataFrame({'STATION': ['CHI0000ZUUU'], 'DATE': ['2025-09-01T00:30:00'],
                      'temperature': [20.], 'dew_point_temperature': [10.],
                      'wind_direction': [90.], 'wind_speed': [2.]})
    for c in ['temperature', 'dew_point_temperature', 'wind_direction', 'wind_speed']:
        f[c + '_Source_Code'] = '223'
        f[c + '_Quality_Code'] = '1'
    return f


def test_units_time_and_components():
    f = prepare_ghcnh_weather(raw())
    assert f.timestamp.iloc[0] == pd.Timestamp('2025-09-01 09:00', tz='Asia/Shanghai')
    assert f.temperature_c.iloc[0] == 20
    assert 50 < f.relative_humidity_pct.iloc[0] < 55
    np.testing.assert_allclose([f.wind_u_ms.iloc[0], f.wind_v_ms.iloc[0]], [-2, 0], atol=1e-12)


def test_bad_quality_and_sentinel_not_real_wind():
    f = raw()
    f['temperature_Quality_Code'] = '2'
    f['wind_direction'] = 999
    out = prepare_ghcnh_weather(f)
    assert out[['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']].isna().all().all()


def test_unhandled_source_fails_closed():
    f = raw()
    f['temperature_Source_Code'] = '999'
    assert prepare_ghcnh_weather(f).temperature_c.isna().all()


def test_wrong_station_rejected():
    f = raw()
    f['STATION'] = 'OTHER'
    with pytest.raises(ValueError, match='station'):
        prepare_ghcnh_weather(f)


def test_no_future_hour_backfill():
    f = raw()
    f['DATE'] = '2025-09-01T00:59:00'
    assert prepare_ghcnh_weather(f).timestamp.dt.hour.iloc[0] == 9
