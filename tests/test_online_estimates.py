import csv
import io
import json
from datetime import UTC, datetime, timedelta

import pytest
import requests

from haze_forecast import online_estimates as online

NOW = datetime(2026, 9, 11, 8, 37, tzinfo=UTC)


def payload(kind):
    hours = [NOW.replace(minute=0) - timedelta(hours=72 - i) for i in range(96)]
    values = (
        {
            "pm2_5": 75,
            "pm10": 150,
            "sulphur_dioxide": 20,
            "nitrogen_dioxide": 30,
            "carbon_monoxide": 1000,
            "ozone": 60,
        }
        if kind == "air"
        else {
            "temperature_2m": 20,
            "relative_humidity_2m": 60,
            "wind_speed_10m": 2,
            "wind_direction_10m": 90,
        }
    )
    units = {
        k: (
            "μg/m³"
            if kind == "air"
            else {
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "wind_speed_10m": "m/s",
                "wind_direction_10m": "°",
            }[k]
        )
        for k in values
    }
    return {
        "utc_offset_seconds": 0,
        "hourly_units": {"time": "iso8601", **units},
        "hourly": {
            "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
            **{k: [v] * 96 for k, v in values.items()},
        },
    }


class Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        pass


@pytest.fixture
def network(monkeypatch):
    air, weather = payload("air"), payload("weather")
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        return Response(air if "air-quality" in url else weather)

    monkeypatch.setattr(online.requests, "get", get)
    monkeypatch.setattr(online, "_now", lambda: NOW)
    return air, weather, calls


def test_fetch_converts_and_excludes_present_future(tmp_path, network):
    result = online.fetch_online_estimate(tmp_path)
    assert len(result["rows"]) == 24
    assert result["end_at"] == "2026-09-11T15:00:00+08:00"
    assert result["start_at"] == "2026-09-10T16:00:00+08:00"
    row = result["rows"][-1]
    assert row["co"] == 1
    assert row["aqi"] == 100
    assert row["wind_u_ms"] == pytest.approx(-2)
    assert row["wind_v_ms"] == pytest.approx(0, abs=1e-8)
    assert row["boundary_layer_height_m"] is None
    records = list(csv.DictReader(io.StringIO(result["csv"])))
    assert len(records) == 25
    assert records[1]["timestamp"] == result["start_at"]
    assert result["model_input_start_at"] == records[0]["timestamp"]
    assert result["model_input_hours"] == 25
    assert result["missing_fields"] == ["boundary_layer_height_m"]
    json.dumps(result, allow_nan=False)
    assert len(list((tmp_path / "artifacts/online_estimates").glob("*/air.json"))) == 1
    assert network[2][0]["timeout"] == (3, 8)


@pytest.mark.parametrize("change", ["missing", "unit", "timezone", "duplicate", "nonfinite"])
def test_bad_source_refused(tmp_path, network, change):
    air = network[0]
    if change == "missing":
        air["hourly"]["ozone"][60] = None
    if change == "unit":
        air["hourly_units"]["carbon_monoxide"] = "mg/m³"
    if change == "timezone":
        air["utc_offset_seconds"] = 28800
    if change == "duplicate":
        air["hourly"]["time"][60] = air["hourly"]["time"][61]
    if change == "nonfinite":
        air["hourly"]["pm2_5"][60] = float("inf")
    with pytest.raises(online.OnlineEstimateError):
        online.fetch_online_estimate(tmp_path)


def test_no_us_aqi_and_o3_rolling(tmp_path, network):
    air = network[0]
    air["hourly"]["us_aqi"] = [999] * 96
    air["hourly"]["ozone"][71] = 160
    result = online.fetch_online_estimate(tmp_path)
    assert result["rows"][-1]["o3"] == 72.5
    assert result["rows"][-1]["aqi"] == 100
    assert "2012" in result["aqi_method"]


def test_network_error_sanitized(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("secret-url-token")

    monkeypatch.setattr(online.requests, "get", fail)
    with pytest.raises(online.OnlineEstimateError) as error:
        online.fetch_online_estimate(tmp_path)
    assert "secret" not in str(error.value)


def test_aqi_old_breakpoints_and_fallback():
    assert online._iaqi(75, online.BREAKPOINTS["pm2_5"]) == 100
    assert online._iaqi(76, online.BREAKPOINTS["pm2_5"]) == 102
    with pytest.raises(online.OnlineEstimateError):
        online._iaqi(501, online.BREAKPOINTS["pm2_5"])


def test_extra_prefix_required_and_no_future_values(tmp_path, network):
    air = network[0]
    for field in online.AIR_FIELDS:
        air["hourly"][field][72:] = [999999] * 24
    assert online.fetch_online_estimate(tmp_path)["rows"][-1]["aqi"] == 100
    air["hourly"]["pm2_5"][25] = None
    with pytest.raises(online.OnlineEstimateError):
        online.fetch_online_estimate(tmp_path)


def test_pm_24h_not_hourly_and_so2_high_fallback(tmp_path, network):
    air = network[0]
    air["hourly"]["pm2_5"] = [0] * 96
    air["hourly"]["pm10"] = [0] * 96
    air["hourly"]["pm2_5"][71] = 480
    result = online.fetch_online_estimate(tmp_path)
    assert result["rows"][-1]["pm2_5"] == 480
    assert result["rows"][-1]["aqi"] == 30
    air["hourly"]["sulphur_dioxide"] = [900] * 96
    assert online.fetch_online_estimate(tmp_path)["rows"][-1]["aqi"] == 213


def test_o3_over_800_uses_hourly_only(tmp_path, network):
    network[0]["hourly"]["ozone"] = [900] * 96
    assert online.fetch_online_estimate(tmp_path)["rows"][-1]["aqi"] == 350


def test_response_limit(tmp_path, monkeypatch):
    response = Response({})
    response.body = b" " * (2 * 1024 * 1024 + 1)
    monkeypatch.setattr(online.requests, "get", lambda *a, **k: response)
    with pytest.raises(online.OnlineEstimateError):
        online.fetch_online_estimate(tmp_path)


def test_model_receives_complete_lag_window(tmp_path, network):
    import pandas as pd

    from haze_forecast.app import build_history_prediction_input

    network[0]["hourly"]["pm2_5"] = list(range(96))
    result = online.fetch_online_estimate(tmp_path)
    frame = pd.read_csv(io.StringIO(result["csv"]))
    _, features = build_history_prediction_input(frame, ["pm2_5_lag_24", "pm2_5_roll_24"])
    assert features.iloc[0]["pm2_5_lag_24"] == 47
    assert features.iloc[0]["pm2_5_roll_24"] == pytest.approx(sum(range(47, 71)) / 24)


@pytest.mark.parametrize("source,index,field", [(0, 24, "pm2_5"), (1, 47, "temperature_2m")])
def test_missing_warmup_refused(tmp_path, network, source, index, field):
    network[source]["hourly"][field][index] = None
    with pytest.raises(online.OnlineEstimateError):
        online.fetch_online_estimate(tmp_path)
