import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from haze_forecast.showcase_server import ShowcaseService, create_server


@pytest.fixture(scope="module")
def service():
    return ShowcaseService()


def manual(service):
    return {
        "mode": "manual",
        "timestamp": "2026-09-11T12:00",
        "values": {field["name"]: field["default"] for field in service.schema()["fields"]},
    }


def sequence(service):
    defaults = manual(service)["values"]
    data = pd.DataFrame([defaults] * 25)
    data.insert(0, "timestamp", pd.date_range("2026-09-10", periods=25, freq="h"))
    return data


def test_real_models(service):
    result = service.predict(manual(service))
    assert result["eligible"] and result["prediction_hours"] > 0
    assert result["actual_hours"] is None
    result = service.predict({"mode": "sequence", "csv": sequence(service).to_csv(index=False)})
    assert result["eligible"] and result["prediction_hours"] > 0
    assert len(result["trend"]) == 25


def test_non_polluted_is_not_zero_hour_claim(service):
    payload = manual(service)
    payload["values"]["aqi"] = 50
    result = service.predict(payload)
    assert not result["eligible"] and result["prediction_hours"] is None
    data = sequence(service)
    data.loc[24, "aqi"] = 50
    result = service.predict({"mode": "sequence", "csv": data.to_csv(index=False)})
    assert not result["eligible"] and result["prediction_hours"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "bad"])
def test_invalid_manual(service, value):
    payload = manual(service)
    payload["values"]["aqi"] = value
    with pytest.raises(ValueError):
        service.predict(payload)


def test_invalid_sequence(service):
    data = sequence(service)
    data.loc[24, "aqi"] = float("nan")
    with pytest.raises(ValueError):
        service.predict({"mode": "sequence", "csv": data.to_csv(index=False)})
    with pytest.raises(ValueError):
        service.predict(
            {"mode": "sequence", "csv": sequence(service).iloc[::2].to_csv(index=False)}
        )


def test_real_replay(service):
    date = service.dates()["dates"][0]
    record = service.history(date)["records"][0]
    result = service.replay(record["id"])
    assert result["error_hours"] == abs(result["prediction_hours"] - result["actual_hours"])
    assert len(result["trend"]) > 1
    assert all(
        pd.Timestamp(row["timestamp"]) <= pd.Timestamp(record["id"]) for row in result["trend"]
    )


def test_http_boundaries(service):
    server = create_server(port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        assert json.load(urlopen(base + "/api/status"))["ready"]
        assert urlopen(base + "/").status == 200
        request = Request(
            base + "/api/predict",
            data=json.dumps(manual(service)).encode(),
            headers={"Content-Type": "application/json", "Origin": base},
        )
        assert json.load(urlopen(request))["prediction_hours"] > 0
        request = Request(
            base + "/api/predict", data=b"{broken", headers={"Content-Type": "application/json"}
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
        for path in ["/../START_HERE.md", "/%2e%2e/START_HERE.md", "/assets/../../START_HERE.md"]:
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
        request = Request(
            base + "/api/predict",
            data=json.dumps(manual(service)).encode(),
            headers={"Content-Type": "application/json", "Origin": "https://evil.example"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        request = Request(
            base + "/api/predict", data=b"{}", headers={"Content-Type": "application/json"}
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_schema_and_missing_weather(service):
    assert set(service.schema()["columns"]) == set(sequence(service).columns)
    data = sequence(service)
    data["boundary_layer_height_m"] = float("nan")
    assert (
        service.predict({"mode": "sequence", "csv": data.to_csv(index=False)})["prediction_hours"]
        > 0
    )


@pytest.mark.parametrize("value", [True, False])
def test_sequence_rejects_boolean_measurements(service, value):
    data = sequence(service)
    data["relative_humidity_pct"] = value
    with pytest.raises(ValueError, match="无效数值"):
        service.predict({"mode": "sequence", "csv": data.to_csv(index=False)})


@pytest.mark.parametrize("mode", ["manual", "sequence"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_model_output_is_not_a_recovery_time(service, monkeypatch, mode, value):
    class InvalidModel:
        def predict(self, frame):
            return [value]

    metadata = service.models[0]
    monkeypatch.setattr(service, "models", (metadata, InvalidModel(), InvalidModel()))
    payload = (
        manual(service)
        if mode == "manual"
        else {"mode": "sequence", "csv": sequence(service).to_csv(index=False)}
    )
    with pytest.raises(RuntimeError, match="not finite"):
        service.predict(payload)


@pytest.mark.parametrize("edit", ["duplicate", "gap", "infinite", "bad_number", "missing_column"])
def test_sequence_validation_cases(service, edit):
    data = sequence(service)
    if edit == "duplicate":
        data.loc[24, "timestamp"] = data.loc[23, "timestamp"]
    elif edit == "gap":
        data.loc[24, "timestamp"] = data.loc[24, "timestamp"] + pd.Timedelta(hours=1)
    elif edit == "infinite":
        data.loc[24, "temperature_c"] = float("inf")
    elif edit == "bad_number":
        data["so2"] = "oops"
    else:
        data = data.drop(columns="co")
    with pytest.raises(ValueError):
        service.predict({"mode": "sequence", "csv": data.to_csv(index=False)})
