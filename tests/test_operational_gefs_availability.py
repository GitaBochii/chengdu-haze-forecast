import json
from datetime import UTC, datetime
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from haze_forecast.operational_gefs import gefs_data_url
from haze_forecast.operational_gefs_availability import audit_frame, head_evidence
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES


def sample():
    run = datetime(2022, 1, 2, tzinfo=UTC)
    messages = []
    fields = ("temperature_2m", "relative_humidity_2m", "u_wind_10m", "v_wind_10m")
    for lead in (6, 24):
        for product in ("geavg", "gespr", "gec00"):
            url = gefs_data_url(run, product=product, lead_hours=lead)
            for field in fields if product != "gec00" else ("pblh",):
                messages.append({"product": product, "lead_hours": lead, "field": field,
                                 "source_url": url, "index_url": url + ".idx",
                                 "byte_range": [0, 20], "sha256": "a" * 64})
    row = {name: 1.0 for name in OPERATIONAL_GEFS_FEATURES}
    row.update(timestamp=pd.Timestamp("2022-01-02T04:00Z"), event_id=1,
               gefs_run_utc=run, gefs_forecast_valid_utc_f006=pd.Timestamp("2022-01-02T06:00Z"),
               gefs_forecast_valid_utc_f024=pd.Timestamp("2022-01-03T00:00Z"),
               gefs_source_manifest_json=json.dumps({"run_utc": run.isoformat(), "messages": messages}))
    return pd.DataFrame([row])


def evidence(url):
    return {"url": url, "http_status": 200, "last_modified": "Sun, 02 Jan 2022 03:43:33 GMT",
            "etag": '"' + "a" * 32 + '"', "cache_sha256": "b" * 64, "cache_path": "test.json"}


def test_masks_each_product_lead_without_dropping_rows():
    frame = sample()
    def fetch(url):
        result = evidence(url)
        if "gespr" in url and "f024" in url:
            result["last_modified"] = "Sun, 02 Jan 2022 04:00:01 GMT"
        return result
    result, report = audit_frame(frame, fetch)
    assert len(result) == 1
    assert result.gefs_spread_temperature_c_f024.isna().all()
    assert result.gefs_mean_temperature_c_f024.iloc[0] == 1
    assert result.gefs_spread_temperature_c_f006.iloc[0] == 1
    assert report["masked_values"] == 4
    assert result.gefs_availability_policy.iloc[0] == "archive_timestamp_proxy"


@pytest.mark.parametrize("change", [{"http_status": 403}, {"etag": '"abcd-2"'},
                                  {"last_modified": None}, {"last_modified": "2022-01-02 01:00"}])
def test_unknown_evidence_never_becomes_available(change):
    result, report = audit_frame(sample(), lambda url: {**evidence(url), **change})
    assert result[list(OPERATIONAL_GEFS_FEATURES)].isna().all().all()
    assert report["unknown_values"] == 18


def test_same_timestamp_availability_is_allowed():
    result, _ = audit_frame(sample(), lambda url: {**evidence(url),
                         "last_modified": "Sun, 02 Jan 2022 04:00:00 GMT"})
    assert result.gefs_availability_all_available.all()


@pytest.mark.parametrize("mutation", ["missing_manifest", "nan_value", "wrong_run", "wrong_url"])
def test_invalid_raw_extraction_is_not_reported_as_audited(mutation):
    frame = sample()
    if mutation == "missing_manifest":
        frame["gefs_source_manifest_json"] = None
    elif mutation == "nan_value":
        frame[OPERATIONAL_GEFS_FEATURES[0]] = np.nan
    elif mutation == "wrong_run":
        frame["gefs_run_utc"] = pd.Timestamp("2022-01-02T06:00Z")
    else:
        manifest = json.loads(frame.gefs_source_manifest_json.iloc[0])
        manifest["messages"][0]["source_url"] = "https://example.com/not-noaa"
        frame["gefs_source_manifest_json"] = json.dumps(manifest)
    with pytest.raises(ValueError):
        audit_frame(frame, evidence)


def test_head_cache_is_immutable_and_reused(tmp_path):
    class Response:
        status_code = 200
        headers: ClassVar[dict] = {"Last-Modified": "Sun, 02 Jan 2022 03:43:33 GMT", "ETag": '"' + "a" * 32 + '"'}
    calls = []
    def head(url, **kwargs):
        calls.append(url)
        return Response()
    url = gefs_data_url(datetime(2022, 1, 2, tzinfo=UTC), product="geavg", lead_hours=6)
    first = head_evidence(url, tmp_path, head=head)
    second = head_evidence(url, tmp_path, head=head)
    assert len(calls) == 1
    assert first["cache_sha256"] == second["cache_sha256"]
    assert first["http_status"] == 200


def test_head_failures_are_bounded_and_retryable_without_overwriting(tmp_path):
    class Response:
        status_code = 503
        headers: ClassVar[dict] = {}
    calls = []
    def head(url, **kwargs):
        calls.append(url)
        return Response()
    url = gefs_data_url(datetime(2022, 1, 2, tzinfo=UTC), product="geavg", lead_hours=6)
    first = head_evidence(url, tmp_path, head=head, sleeper=lambda _: None)
    assert len(calls) == 3
    original = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    Response.status_code = 200
    second = head_evidence(url, tmp_path, head=head, sleeper=lambda _: None)
    assert len(calls) == 4
    assert first["http_status"] == 503 and second["http_status"] == 200
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())


def test_object_timestamp_before_initialization_is_unknown():
    result, report = audit_frame(sample(), lambda url: {**evidence(url),
                           "last_modified": "Sat, 01 Jan 2022 23:59:59 GMT"})
    assert report["unknown_values"] == 18
    assert not result.gefs_availability_all_available.any()
