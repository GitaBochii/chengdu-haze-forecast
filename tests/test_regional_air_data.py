import json

import pandas as pd
import pytest

from haze_forecast.regional_air_data import (
    aggregate_regional_city_hourly,
    complete_station_hours,
    load_station_metadata,
    parse_station_daily,
)


def metadata():
    return pd.DataFrame({
        "station_id": ["a", "b", "c"], "city": ["chengdu"] * 3,
        "longitude": [104.0, 104.1, 104.2], "latitude": [30.0, 30.1, 30.2],
    })


def test_hourly_pollutants_timezone_and_missing_station():
    raw = pd.DataFrame({
        "date": [20240101, 20240101], "hour": [0, 0],
        "type": ["PM2.5", "PM2.5_24h"], "a": [12, 999], "b": [None, 999],
    })
    actual = parse_station_daily(raw, metadata())
    assert len(actual) == 3
    assert actual.loc[actual.station_id == "a", "pm2_5"].item() == 12
    assert actual.loc[actual.station_id == "c", "pm2_5"].isna().all()
    assert str(actual.timestamp.dt.tz) == "Asia/Shanghai"


def test_city_uses_existing_median_not_mean():
    raw = pd.DataFrame({
        "date": [20240101], "hour": [0], "type": ["PM2.5"],
        "a": [1], "b": [3], "c": [100],
    })
    actual = aggregate_regional_city_hourly(parse_station_daily(raw, metadata()))
    assert actual.pm2_5.item() == 3
    assert actual.pm2_5_station_count.item() == 3


def test_complete_grid_does_not_fill_future_or_past_values():
    raw = pd.DataFrame({
        "date": [20240101, 20240101], "hour": [0, 2], "type": ["PM2.5"] * 2,
        "a": [1, 9], "b": [2, 8], "c": [3, 7],
    })
    frame = parse_station_daily(raw, metadata())
    actual = complete_station_hours(frame, metadata(), "2024-01-01", "2024-01-01")
    assert len(actual) == 24 * 3
    assert actual.loc[actual.timestamp.dt.hour == 1, "pm2_5"].isna().all()
    assert actual.loc[actual.timestamp.dt.hour == 23, "pm2_5"].isna().all()


def test_reject_duplicate_daily_measurements_and_invalid_hour():
    raw = pd.DataFrame({
        "date": [20240101, 20240101], "hour": [0, 0], "type": ["PM2.5"] * 2,
        "a": [1, 2],
    })
    with pytest.raises(ValueError, match="duplicate"):
        parse_station_daily(raw, metadata())
    with pytest.raises(ValueError, match="hour"):
        parse_station_daily(raw.iloc[:1].assign(hour=24), metadata())


def test_metadata_json_longitude_latitude_order(tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(json.dumps({"1431A": [103.9728, 30.7236]}))
    actual = load_station_metadata(path)
    assert actual.longitude.item() == 103.9728
    assert actual.latitude.item() == 30.7236
    assert actual.city.item() == "chengdu"
