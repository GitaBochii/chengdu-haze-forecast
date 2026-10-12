import importlib.util

import numpy as np
import pandas as pd
import pytest


def test_spatial_features_use_only_the_current_timestamp_and_follow_wind_direction():
    assert importlib.util.find_spec("haze_forecast.spatial") is not None
    from haze_forecast.spatial import build_city_spatial_features

    timestamps = pd.date_range("2021-01-01", periods=2, freq="h", tz="Asia/Shanghai")
    station_hourly = pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[0], timestamps[1], timestamps[1]],
            "station_id": ["west", "east", "west", "east"],
            "pm2_5": [10.0, 30.0, 999.0, 50.0],
            "pm10": [20.0, 40.0, 999.0, 60.0],
            "wind_u_ms": [0.0, 1.0, 0.0, 1.0],
            "wind_v_ms": [0.0, 0.0, 0.0, 0.0],
        }
    )
    coordinates = {"west": [103.0, 30.0], "east": [104.0, 30.0]}

    result = build_city_spatial_features(station_hourly, coordinates)

    first = result.loc[result["timestamp"].eq(timestamps[0])].iloc[0]
    assert first["spatial_station_count"] == 2
    assert first["spatial_pm2_5_std"] == 10.0
    assert first["spatial_pm2_5_range"] == 20.0
    assert first["spatial_upwind_pm2_5_mean"] == 10.0


def test_city_spatial_enrichment_preserves_city_rows_and_adds_hourly_features():
    from haze_forecast.spatial import enrich_city_with_spatial_features

    timestamp = pd.Timestamp("2021-01-01 00:00", tz="Asia/Shanghai")
    city = pd.DataFrame({"timestamp": [timestamp], "aqi": [120.0]})
    stations = pd.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "station_id": ["west", "east"],
            "pm2_5": [10.0, 30.0],
            "pm10": [20.0, 40.0],
            "wind_u_ms": [0.0, 1.0],
            "wind_v_ms": [0.0, 0.0],
        }
    )

    enriched = enrich_city_with_spatial_features(city, stations, {"west": [103.0, 30.0], "east": [104.0, 30.0]})

    assert enriched.shape[0] == 1
    assert enriched.loc[0, "aqi"] == 120.0
    assert enriched.loc[0, "spatial_upwind_pm10_mean"] == 20.0


def test_spatial_features_report_eastward_transport_and_station_pair_coverage():
    from haze_forecast.spatial import build_city_spatial_features

    timestamp = pd.Timestamp("2021-01-01", tz="Asia/Shanghai")
    stations = pd.DataFrame(
        {
            "timestamp": [timestamp] * 3,
            "station_id": ["west", "middle", "east"],
            "pm2_5": [100.0, 40.0, 10.0],
            "pm10": [150.0, 60.0, 20.0],
            "wind_u_ms": [2.0, 2.0, 2.0],
            "wind_v_ms": [0.0, 0.0, 0.0],
        }
    )

    row = build_city_spatial_features(
        stations, {"west": [103, 30], "middle": [104, 30], "east": [105, 30]}
    ).iloc[0]

    assert row["spatial_wind_direction_consistency"] == pytest.approx(1.0)
    assert row["spatial_pm2_5_transport_pair_count"] == 6.0
    assert row["spatial_pm2_5_upwind_downwind_difference"] > 0
    assert row["spatial_pm2_5_gradient_median"] > 0


def test_directional_features_keep_missing_for_zero_wind_or_one_valid_station():
    from haze_forecast.spatial import build_city_spatial_features

    timestamp = pd.Timestamp("2021-01-01", tz="Asia/Shanghai")
    stations = pd.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "station_id": ["west", "east"],
            "pm2_5": [50.0, np.nan],
            "pm10": [80.0, np.nan],
            "wind_u_ms": [0.0, 0.0],
            "wind_v_ms": [0.0, 0.0],
        }
    )

    row = build_city_spatial_features(stations, {"west": [103, 30], "east": [104, 30]}).iloc[0]

    assert pd.isna(row["spatial_pm2_5_upwind_downwind_difference"])
    assert row["spatial_pm2_5_transport_pair_count"] == 0.0
