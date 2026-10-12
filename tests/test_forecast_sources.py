from datetime import UTC, datetime

import pytest
import requests

from haze_forecast.forecast_sources import (
    GFS_ARCHIVE_CATALOG_ROOT,
    GFS_AWS_ROOT,
    GFSProbeResult,
    build_operational_gfs_request,
    download_gfs_surface_messages,
    gfs_aws_data_url,
    gfs_aws_index_url,
    gfs_catalog_url,
    gfs_dods_url,
    parse_gfs_index,
    probe_gfs_aws,
    probe_gfs_opendap,
    range_for_gfs_index_entry,
    select_gfs_surface_fields,
    write_probe_manifest,
)


def test_gfs_urls_use_archive_date_and_cycle():
    run_time = datetime(2021, 3, 23, 18, tzinfo=UTC)

    assert gfs_catalog_url(run_time) == (
        f"{GFS_ARCHIVE_CATALOG_ROOT}/202103/20210323/catalog.xml"
    )
    assert gfs_dods_url(run_time, lead_hours=6) == (
        "https://www.ncei.noaa.gov/thredds/dodsC/gfs-g4-files/202103/20210323/"
        "gfs_3_20210323_1800_006.grb2"
    )
    assert gfs_aws_data_url(run_time, lead_hours=6) == (
        f"{GFS_AWS_ROOT}/gfs.20210323/18/atmos/gfs.t18z.pgrb2.0p25.f006"
    )
    assert gfs_aws_index_url(run_time, lead_hours=6).endswith("pgrb2.0p25.f006.idx")


def test_operational_request_never_uses_a_forecast_run_after_issue_time():
    request = build_operational_gfs_request(
        issued_at=datetime(2021, 3, 23, 14, tzinfo=UTC),
        latitude=30.67,
        longitude=104.06,
        max_lead_hours=24,
    )

    assert request.run_time == datetime(2021, 3, 23, 12, tzinfo=UTC)
    assert request.run_time <= request.issued_at
    assert request.lead_hours == tuple(range(3, 25, 3))
    assert request.valid_times[0] == datetime(2021, 3, 23, 15, tzinfo=UTC)
    assert request.valid_times[-1] == datetime(2021, 3, 24, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("issued_at", "max_lead_hours"),
    [
        (datetime(2021, 3, 23, 14), 24),  # noqa: DTZ001 - intentional invalid input
        (datetime(2021, 3, 23, 14, tzinfo=UTC), 0),
        (datetime(2021, 3, 23, 14, tzinfo=UTC), 25),
    ],
)
def test_operational_request_rejects_ambiguous_or_unsupported_inputs(issued_at, max_lead_hours):
    with pytest.raises(ValueError):
        build_operational_gfs_request(
            issued_at=issued_at,
            latitude=30.67,
            longitude=104.06,
            max_lead_hours=max_lead_hours,
        )


def test_probe_manifest_keeps_available_flag_and_error_evidence(tmp_path):
    result = GFSProbeResult(
        requested_at_utc="2021-03-23T18:00:00+00:00",
        url="https://example.test/schema.dds",
        status_code=None,
        response_bytes=0,
        error="ReadTimeout: timed out",
    )
    output = tmp_path / "probe.json"

    write_probe_manifest(output, result)

    assert output.read_text(encoding="utf-8") == """{
  "requested_at_utc": "2021-03-23T18:00:00+00:00",
  "url": "https://example.test/schema.dds",
  "status_code": null,
  "response_bytes": 0,
  "error": "ReadTimeout: timed out",
  "available": false
}"""


def test_probe_records_network_failure_without_claiming_data_is_available(monkeypatch):
    def fail_request(*_args, **_kwargs):
        raise requests.ReadTimeout("archive timed out")

    monkeypatch.setattr("haze_forecast.forecast_sources.requests.get", fail_request)

    result = probe_gfs_opendap(datetime(2021, 3, 23, 18, tzinfo=UTC), lead_hours=6)

    assert result.available is False
    assert result.status_code is None
    assert result.response_bytes == 0
    assert result.error == "ReadTimeout: archive timed out"


def test_aws_probe_uses_compact_index_and_accepts_a_nonempty_response(monkeypatch):
    class Response:
        status_code = 200
        content = b"1:0:d=2021032300:TMP:2 m above ground:6 hour fcst:\n"

    def successful_request(*_args, **_kwargs):
        return Response()

    monkeypatch.setattr("haze_forecast.forecast_sources.requests.get", successful_request)
    result = probe_gfs_aws(datetime(2021, 3, 23, 18, tzinfo=UTC), lead_hours=6)

    assert result.available is True
    assert result.url.endswith("gfs.t18z.pgrb2.0p25.f006.idx")


def test_gfs_index_selects_only_exact_surface_fields_and_computes_byte_range():
    entries = parse_gfs_index("""1:0:d=2021032300:TMP:2 mb:6 hour fcst:
2:100:d=2021032300:TMP:2 m above ground:6 hour fcst:
3:200:d=2021032300:RH:2 m above ground:6 hour fcst:
4:300:d=2021032300:UGRD:10 m above ground:6 hour fcst:
5:400:d=2021032300:VGRD:10 m above ground:6 hour fcst:
6:500:d=2021032300:HPBL:surface:6 hour fcst:
7:600:d=2021032300:TMP:80 m above ground:6 hour fcst:""")

    fields = select_gfs_surface_fields(entries)

    assert [entry.variable for entry in fields.values()] == ["TMP", "RH", "UGRD", "VGRD", "HPBL"]
    assert range_for_gfs_index_entry(entries, fields["u_wind_10m"]) == (300, 399)


def test_gfs_range_download_reads_only_the_requested_grib_message(monkeypatch):
    index = """1:0:d=2021032300:TMP:2 m above ground:6 hour fcst:
2:100:d=2021032300:RH:2 m above ground:6 hour fcst:
3:200:d=2021032300:UGRD:10 m above ground:6 hour fcst:
4:300:d=2021032300:VGRD:10 m above ground:6 hour fcst:
5:400:d=2021032300:HPBL:surface:6 hour fcst:
6:500:d=2021032300:HGT:surface:6 hour fcst:"""

    class Response:
        def __init__(self, status_code, content, headers=None):
            self.status_code = status_code
            self.content = content
            self.headers = headers or {}

        def raise_for_status(self):
            return None

    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith(".idx"):
            return Response(200, index.encode())
        return Response(206, b"GRIB", {"Content-Range": "bytes 200-299/600"})

    monkeypatch.setattr("haze_forecast.forecast_sources.requests.get", fake_get)
    messages = download_gfs_surface_messages(
        datetime(2021, 3, 23, 18, tzinfo=UTC),
        lead_hours=6,
        field_names=("u_wind_10m",),
    )

    assert messages["u_wind_10m"].content == b"GRIB"
    assert messages["u_wind_10m"].byte_range == (200, 299)
    assert calls[1][1]["headers"]["Range"] == "bytes=200-299"


def test_gfs_range_download_retries_a_timed_out_partial_response(monkeypatch):
    index = """1:0:d=2021032300:TMP:2 m above ground:6 hour fcst:
2:100:d=2021032300:RH:2 m above ground:6 hour fcst:
3:200:d=2021032300:UGRD:10 m above ground:6 hour fcst:
4:300:d=2021032300:VGRD:10 m above ground:6 hour fcst:
5:400:d=2021032300:HPBL:surface:6 hour fcst:
6:500:d=2021032300:HGT:surface:6 hour fcst:"""

    class IndexResponse:
        status_code = 200
        content = index.encode()

        def raise_for_status(self):
            return None

    class TimedOutRangeResponse:
        status_code = 206

        def __init__(self):
            self.headers = {"Content-Range": "bytes 200-299/600"}

        @property
        def content(self):
            raise requests.ConnectionError("range read timed out")

    class SuccessfulRangeResponse:
        status_code = 206

        def __init__(self):
            self.headers = {"Content-Range": "bytes 200-299/600"}
            self.content = b"GRIB"

    calls = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        if url.endswith(".idx"):
            return IndexResponse()
        return TimedOutRangeResponse() if len(calls) == 2 else SuccessfulRangeResponse()

    monkeypatch.setattr("haze_forecast.forecast_sources.requests.get", fake_get)
    messages = download_gfs_surface_messages(
        datetime(2021, 3, 23, 18, tzinfo=UTC),
        lead_hours=6,
        field_names=("u_wind_10m",),
        max_attempts=2,
        retry_delay_s=0,
    )

    assert messages["u_wind_10m"].content == b"GRIB"
    assert len(calls) == 3
