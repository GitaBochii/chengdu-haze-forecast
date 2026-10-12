import time

import pandas as pd
import pytest

from haze_forecast.showcase_server import InputError, ShowcaseService


@pytest.fixture
def online(monkeypatch):
    from haze_forecast import online_estimates

    service = ShowcaseService()
    values = {field["name"]: field["default"] for field in service.schema()["fields"]}
    frame = pd.DataFrame([values] * 25)
    frame.insert(
        0, "timestamp", pd.date_range("2026-09-10", periods=25, freq="h", tz="Asia/Shanghai")
    )
    data = {
        "csv": frame.to_csv(index=False),
        "rows": [],
        "start_at": frame.iloc[1]["timestamp"].isoformat(),
        "model_input_start_at": frame.iloc[0]["timestamp"].isoformat(),
        "model_input_hours": 25,
        "end_at": frame.iloc[-1]["timestamp"].isoformat(),
        "fetched_at": "2026-09-11T00:00:00+08:00",
        "notice": "测试模型估算，不是实测。",
        "missing_fields": [],
        "source_urls": ["https://open-meteo.com/en/docs/air-quality-api"],
    }
    calls = []

    def fetch(root):
        calls.append(root)
        return data

    monkeypatch.setattr(online_estimates, "fetch_online_estimate", fetch)
    return service, calls


def test_import_cache_and_private_csv(online):
    service, calls = online
    first = service.import_online()
    assert first["can_predict"] and "csv" not in first
    assert service.import_online()["token"] == first["token"]
    assert len(calls) == 1


def test_online_prediction_is_explicitly_experimental(online):
    service, _ = online
    preview = service.import_online()
    result = service.predict(
        {"mode": "online", "token": preview["token"], "acknowledge_estimates": True}
    )
    assert result["mode"] == "online"
    assert result["actual_hours"] is None and result["error_hours"] is None
    assert "非实测" in result["model_label"]
    assert "独立精度" in result["notice"]
    assert result["provenance"]["fetched_at"] == preview["fetched_at"]
    assert result["provenance"]["model_input_hours"] == 25
    assert len(result["trend"]) == 24
    assert result["trend"][0]["timestamp"] == preview["start_at"]


@pytest.mark.parametrize("case", ["no_import", "no_consent", "wrong_token", "expired"])
def test_online_prediction_gates(online, case):
    service, _ = online
    preview = service.import_online()
    payload = {"mode": "online", "token": preview["token"], "acknowledge_estimates": True}
    if case == "no_import":
        service.online_snapshot = None
    elif case == "no_consent":
        payload["acknowledge_estimates"] = False
    elif case == "wrong_token":
        payload["token"] = "invalid"
    else:
        service.online_snapshot = (time.monotonic() - 901, service.online_snapshot[1])
    with pytest.raises(InputError):
        service.predict(payload)
