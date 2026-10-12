import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_regional_weather_proxy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("regional_weather_proxy", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload():
    return {
        "timezone": "Asia/Shanghai", "utc_offset_seconds": 28800,
        "hourly_units": {"wind_speed_10m": "m/s"},
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T02:00"],
            "temperature_2m": [1.0, 3.0], "relative_humidity_2m": [50, 60],
            "wind_speed_10m": [2.0, 4.0], "wind_direction_10m": [90, 180],
            "boundary_layer_height": [100, None],
        },
    }


def test_source_units_and_local_time_are_validated():
    module = load_module()
    value = payload()
    result = module.parse_proxy_payload(value)
    assert str(result.timestamp.dt.tz) == "Asia/Shanghai"
    assert result.wind_u_ms.iloc[0] == -2.0
    assert result.wind_v_ms.iloc[1] == 4.0
    value["hourly_units"]["wind_speed_10m"] = "km/h"
    with pytest.raises(ValueError, match="m/s"):
        module.parse_proxy_payload(value)
    value = payload()
    value["utc_offset_seconds"] = 0
    with pytest.raises(ValueError, match="timezone"):
        module.parse_proxy_payload(value)


def test_missing_pbl_stays_missing():
    module = load_module()
    value = payload()
    del value["hourly"]["boundary_layer_height"]
    result = module.parse_proxy_payload(value)
    assert result.boundary_layer_height_m.isna().all()


def test_complete_grid_preserves_missing_hours_and_rejects_duplicates():
    module = load_module()
    frame = module.parse_proxy_payload(payload())
    grid = pd.date_range("2023-01-01", periods=4, freq="h", tz="Asia/Shanghai")
    result = module.align_proxy([frame], grid)
    assert len(result) == 4
    assert result.temperature_c.tolist()[0] == 1
    assert result.loc[[1, 3], "temperature_c"].isna().all()
    with pytest.raises(ValueError, match="duplicate"):
        module.align_proxy([frame, frame], grid)


def test_year_ranges_cover_partial_years():
    module = load_module()
    ranges = module.year_ranges("2014-05-13", "2026-09-03")
    assert len(ranges) == 13
    assert ranges[0] == (2014, "2014-05-13", "2014-12-31")
    assert ranges[-1] == (2026, "2026-01-01", "2026-09-03")
    with pytest.raises(ValueError):
        module.year_ranges("2026-09-03", "2014-05-13")


def test_cache_requires_matching_identity_and_hash(tmp_path):
    module = load_module()
    path = tmp_path / "weather.json"
    path.write_text(json.dumps(payload()))
    params = {"models": "era5"}
    entry = {"parameters": params, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "status": "validated"}
    with pytest.raises(ValueError, match="identity"):
        module.validate_cache(path, {"models": "other"}, entry)
    entry["sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        module.validate_cache(path, params, entry)


def test_failed_http_is_not_written_as_cache(tmp_path, monkeypatch):
    module = load_module()
    class Response:
        status_code = 500
        content = b'{"error":true}'
        ok = False
        def raise_for_status(self):
            raise module.requests.HTTPError("500")
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: Response())
    path = tmp_path / "failed.json"
    with pytest.raises(module.requests.HTTPError):
        module.fetch_response({}, path, [])
    assert not path.exists()


def daily_payload():
    value = payload()
    value["hourly_units"].update({"temperature_2m": "\u00b0C", "relative_humidity_2m": "%",
                                   "wind_direction_10m": "\u00b0"})
    value["hourly"]["time"] = [str(t.strftime("%Y-%m-%dT%H:%M")) for t in
                               pd.date_range("2023-01-01", periods=24, freq="h")]
    for name in value["hourly"]:
        if name != "time":
            value["hourly"][name] = [1.0] * 24
    return value


def test_response_requires_exact_requested_dates_and_units():
    module = load_module()
    value = daily_payload()
    params = {"start_date": "2023-01-01", "end_date": "2023-01-01"}
    module.validate_response(value, params)
    with pytest.raises(ValueError, match="timestamp grid"):
        module.validate_response(value, {**params, "end_date": "2023-01-02"})
    value["hourly_units"]["temperature_2m"] = "F"
    with pytest.raises(ValueError, match="unit"):
        module.validate_response(value, params)


def test_verified_cache_copy_preserves_original_and_never_fetches(tmp_path, monkeypatch):
    module = load_module()
    source = tmp_path / "original"
    destination = tmp_path / "new"
    source.mkdir()
    destination.mkdir()
    path = source / "era5_2023.json"
    path.write_text(json.dumps(daily_payload()))
    params = {"start_date": "2023-01-01", "end_date": "2023-01-01"}
    entry = {"parameters": params, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "status": "validated", "response_path": str(path)}
    audit_path = source / "audit.json"
    audit_path.write_text(json.dumps({"requests": [entry]}))
    original_audit = audit_path.read_bytes()
    def no_network(*args, **kwargs):
        pytest.fail("verified cache must not access network")
    monkeypatch.setattr(module.requests, "get", no_network)
    copied = destination / path.name
    audit = []
    module.fetch_response(params, copied, audit, source)
    assert copied.read_bytes() == path.read_bytes()
    assert audit_path.read_bytes() == original_audit
    assert audit[0]["reused_from"] == str(path.resolve())
    module.fetch_response(params, copied, [])
