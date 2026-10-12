import pandas as pd
import pytest

from haze_forecast.constants import REQUIRED_COLUMNS
from haze_forecast.validation import normalize_source_frame, validate_hourly_frame


def test_required_columns_include_timestamp_and_core_pollutants():
    assert REQUIRED_COLUMNS == {
        "timestamp",
        "station_id",
        "aqi",
        "pm2_5",
        "pm10",
        "so2",
        "no2",
        "co",
        "o3",
        "temperature_c",
        "relative_humidity_pct",
        "wind_u_ms",
        "wind_v_ms",
        "boundary_layer_height_m",
    }


def test_validate_hourly_frame_rejects_duplicate_station_timestamp():
    row = {
        "timestamp": "2021-01-01 00:00:00",
        "station_id": "1431A",
        "aqi": 120,
        "pm2_5": 90,
        "pm10": 110,
        "so2": 12,
        "no2": 30,
        "co": 0.8,
        "o3": 50,
        "temperature_c": 8,
        "relative_humidity_pct": 70,
        "wind_u_ms": 1,
        "wind_v_ms": 0,
        "boundary_layer_height_m": 300,
    }
    duplicated = pd.DataFrame([row, row])

    with pytest.raises(ValueError, match="duplicate station_id/timestamp"):
        validate_hourly_frame(duplicated)


def test_normalize_source_frame_converts_kelvin_to_celsius():
    source = pd.DataFrame(
        {
            "year": [2021],
            "month": [1],
            "day": [1],
            "hour": [0],
            "AQI": [120],
            "PM2.5": [90],
            "PM10": [110],
            "SO2": [12],
            "NO2": [30],
            "O3-8h": [50],
            "CO": [0.8],
            "U10": [1],
            "V10": [0],
            "T2": [279.15],
            "RH2": [70],
            "PBLH": [300],
        }
    )

    result = normalize_source_frame(source, "1431A")

    assert result.loc[0, "temperature_c"] == pytest.approx(6.0)
