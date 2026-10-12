import numpy as np
import pandas as pd
import pytest

from haze_forecast.cnemc_weather import add_noaa_weather


def test_noaa_weather_merge_derives_relative_humidity_and_wind_vector():
    pollution = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-05-13 00:00:00+08:00"]),
            "aqi": [120.0],
            "pm2_5": [80.0],
            "pm10": [100.0],
            "so2": [10.0],
            "no2": [20.0],
            "co": [1.0],
            "o3": [30.0],
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-05-13 00:00:00+08:00"]),
            "noaa_temperature_c": [20.0],
            "noaa_dewpoint_c": [10.0],
            "noaa_wind_direction_deg": [180.0],
            "noaa_wind_speed_ms": [5.0],
        }
    )

    merged = add_noaa_weather(pollution, weather)

    assert merged.loc[0, "temperature_c"] == 20.0
    assert merged.loc[0, "relative_humidity_pct"] == pytest.approx(52.5, abs=0.2)
    assert merged.loc[0, "wind_u_ms"] == pytest.approx(0.0, abs=1e-12)
    assert merged.loc[0, "wind_v_ms"] == pytest.approx(5.0, abs=1e-12)
    assert np.isnan(merged.loc[0, "boundary_layer_height_m"])


def test_noaa_weather_merge_keeps_pollution_hour_when_station_weather_is_missing():
    pollution = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2014-05-13 00:00:00+08:00"]), "aqi": [120.0]}
    )
    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-05-13 01:00:00+08:00"]),
            "noaa_temperature_c": [20.0],
        }
    )

    merged = add_noaa_weather(pollution, weather)

    assert len(merged) == 1
    assert np.isnan(merged.loc[0, "temperature_c"])
