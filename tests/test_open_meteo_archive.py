import pandas as pd

from haze_forecast.open_meteo_archive import (
    add_boundary_layer_height,
    fill_weather_gaps,
    parse_boundary_layer_height,
    parse_hourly_weather,
)


def test_parse_boundary_layer_height_uses_requested_local_timestamps():
    payload = {
        "hourly": {
            "time": ["2014-05-13T00:00", "2014-05-13T01:00"],
            "boundary_layer_height": [75.0, 85.0],
        }
    }

    result = parse_boundary_layer_height(payload)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2014-05-13 00:00:00+08:00"),
        pd.Timestamp("2014-05-13 01:00:00+08:00"),
    ]
    assert result["boundary_layer_height_m"].tolist() == [75.0, 85.0]


def test_add_boundary_layer_height_left_joins_without_changing_pollution_rows():
    pollution = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2014-05-13 00:00", "2014-05-13 01:00"], utc=True
            ).tz_convert("Asia/Shanghai"),
            "aqi": [120.0, 130.0],
        }
    )
    pbl = pd.DataFrame(
        {
            "timestamp": [pollution.loc[0, "timestamp"]],
            "boundary_layer_height_m": [75.0],
        }
    )

    result = add_boundary_layer_height(pollution, pbl)

    assert result["aqi"].tolist() == [120.0, 130.0]
    assert result["boundary_layer_height_m"].tolist()[0] == 75.0
    assert pd.isna(result["boundary_layer_height_m"].tolist()[1])


def test_parse_hourly_weather_returns_model_weather_columns_and_wind_vector():
    payload = {
        "hourly": {
            "time": ["2025-12-30T00:00"],
            "temperature_2m": [8.3],
            "relative_humidity_2m": [91],
            "wind_speed_10m": [5.0],
            "wind_direction_10m": [180],
            "boundary_layer_height": [55.0],
        }
    }

    result = parse_hourly_weather(payload)

    assert result.loc[0, "timestamp"] == pd.Timestamp("2025-12-30 00:00:00+08:00")
    assert result.loc[0, "temperature_c"] == 8.3
    assert result.loc[0, "relative_humidity_pct"] == 91.0
    assert result.loc[0, "wind_u_ms"] == 0.0
    assert result.loc[0, "wind_v_ms"] == 5.0
    assert result.loc[0, "boundary_layer_height_m"] == 55.0


def test_fill_weather_gaps_preserves_observed_values_and_uses_archive_for_missing_values():
    observed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-30 00:00:00+08:00"]),
            "temperature_c": [None],
            "relative_humidity_pct": [88.0],
            "wind_u_ms": [None],
            "wind_v_ms": [None],
            "boundary_layer_height_m": [None],
        }
    )
    archive = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-30 00:00:00+08:00"]),
            "temperature_c": [8.3],
            "relative_humidity_pct": [91.0],
            "wind_u_ms": [0.0],
            "wind_v_ms": [5.0],
            "boundary_layer_height_m": [55.0],
        }
    )

    result = fill_weather_gaps(observed, archive)

    assert result.loc[0, "temperature_c"] == 8.3
    assert result.loc[0, "relative_humidity_pct"] == 88.0
    assert result.loc[0, "wind_u_ms"] == 0.0
    assert result.loc[0, "wind_v_ms"] == 5.0
    assert result.loc[0, "boundary_layer_height_m"] == 55.0
