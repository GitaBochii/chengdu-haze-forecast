from datetime import UTC, datetime
from threading import Event, Lock, Thread

import pytest
from requests.exceptions import HTTPError, SSLError

from haze_forecast.forecast_sources import parse_gfs_index
from haze_forecast.gefs_reforecast import (
    GEFS_REFORECAST_ROOT,
    GEFS_SURFACE_VARIABLES,
    GEFSReforecastIndexCache,
    download_gefs_reforecast_message,
    gefs_reforecast_data_url,
    gefs_reforecast_index_url,
    range_for_reforecast_lead,
    select_reforecast_lead,
)


def test_surface_variable_mapping_requires_10_metre_wind_messages():
    assert tuple(GEFS_SURFACE_VARIABLES) == (
        "tmp_2m",
        "spfh_2m",
        "ugrd_hgt",
        "vgrd_hgt",
        "pres_sfc",
        "pbl_hgt",
    )
    assert GEFS_SURFACE_VARIABLES["ugrd_hgt"].level_hint == "10 m above ground"
    assert GEFS_SURFACE_VARIABLES["vgrd_hgt"].level_hint == "10 m above ground"
    assert GEFS_SURFACE_VARIABLES["tmp_2m"].level_hint is None


def test_gefs_reforecast_urls_encode_run_member_and_variable():
    run_time = datetime(2015, 1, 1, tzinfo=UTC)

    data_url = gefs_reforecast_data_url(run_time, member="p03", variable="pbl_hgt")

    assert data_url == (
        f"{GEFS_REFORECAST_ROOT}/2015/2015010100/p03/Days:1-10/"
        "pbl_hgt_2015010100_p03.grib2"
    )
    assert gefs_reforecast_index_url(run_time, member="p03", variable="pbl_hgt") == (
        f"{data_url}.idx"
    )


@pytest.mark.parametrize(
    ("run_time", "member", "variable"),
    [
        (datetime(1999, 12, 31, tzinfo=UTC), "c00", "tmp_2m"),
        (datetime(2020, 1, 1, tzinfo=UTC), "c00", "tmp_2m"),
        (datetime(2015, 1, 1, 6, tzinfo=UTC), "c00", "tmp_2m"),
        (datetime(2015, 1, 1, tzinfo=UTC), "p05", "tmp_2m"),
        (datetime(2015, 1, 1, tzinfo=UTC), "c00", "unknown"),
    ],
)
def test_gefs_reforecast_url_rejects_unavailable_layouts(run_time, member, variable):
    with pytest.raises(ValueError):
        gefs_reforecast_data_url(run_time, member=member, variable=variable)


def test_reforecast_lead_selection_and_range_use_only_one_grib_message():
    entries = parse_gfs_index(
        "1:0:d=2015010100:TMP:2 m above ground:3 hour fcst:\n"
        "2:100:d=2015010100:TMP:2 m above ground:6 hour fcst:\n"
        "3:250:d=2015010100:TMP:2 m above ground:9 hour fcst:\n"
        "4:400:d=2015010100:TMP:2 m above ground:12 hour fcst:"
    )

    entry = select_reforecast_lead(entries, lead_hours=6)

    assert entry.record_number == 2
    assert range_for_reforecast_lead(entries, entry) == (100, 249)


def test_reforecast_lead_selection_can_disambiguate_wind_height():
    entries = parse_gfs_index(
        "1:0:d=2015010100:UGRD:10 m above ground:24 hour fcst:\n"
        "2:100:d=2015010100:UGRD:100 m above ground:24 hour fcst:"
    )

    entry = select_reforecast_lead(entries, lead_hours=24, level_hint="10 m above ground")

    assert entry.record_number == 1


def test_reforecast_lead_selection_rejects_similar_but_wrong_wind_height():
    entries = parse_gfs_index("1:0:d=2015010100:UGRD:110 m above ground:24 hour fcst:")

    with pytest.raises(ValueError, match="exactly one"):
        select_reforecast_lead(entries, lead_hours=24, level_hint="10 m above ground")


def test_reforecast_lead_selection_rejects_missing_or_ambiguous_lead():
    entries = parse_gfs_index(
        "1:0:d=2015010100:TMP:2 m above ground:3 hour fcst:\n"
        "2:100:d=2015010100:TMP:2 m above ground:6 hour fcst:\n"
        "3:250:d=2015010100:TMP:2 m above ground:6 hour fcst:"
    )

    with pytest.raises(ValueError, match="exactly one"):
        select_reforecast_lead(entries, lead_hours=6)
    with pytest.raises(ValueError, match="exactly one"):
        select_reforecast_lead(entries, lead_hours=12)


def test_reforecast_downloader_reads_index_then_exact_single_message_range(monkeypatch):
    index = (
        "1:0:d=2015010100:TMP:2 m above ground:3 hour fcst:\n"
        "2:100:d=2015010100:TMP:2 m above ground:6 hour fcst:\n"
        "3:250:d=2015010100:TMP:2 m above ground:9 hour fcst:\n"
        "4:400:d=2015010100:TMP:2 m above ground:12 hour fcst:"
    )

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
            return Response(200, index.encode("utf-8"))
        return Response(206, b"GRIB", {"Content-Range": "bytes 100-249/300"})

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)
    message = download_gefs_reforecast_message(
        datetime(2015, 1, 1, tzinfo=UTC), member="c00", variable="tmp_2m", lead_hours=6
    )

    assert message.content == b"GRIB"
    assert message.byte_range == (100, 249)
    assert message.sha256 == "1ed0532076444a48a796ea8edeb377ad0d4180936bc47d97fd199a5216b16f11"
    assert calls[1][1]["headers"]["Range"] == "bytes=100-249"


def test_reforecast_downloader_retries_transient_ssl_range_failures_without_sleeping(monkeypatch):
    index = (
        "1:0:d=2015010100:VGRD:10 m above ground:3 hour fcst:\n"
        "2:100:d=2015010100:VGRD:10 m above ground:6 hour fcst:\n"
        "3:250:d=2015010100:VGRD:10 m above ground:9 hour fcst:"
    )

    class Response:
        def __init__(self, status_code, content, headers=None):
            self.status_code = status_code
            self.content = content
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise HTTPError(f"HTTP {self.status_code}")

    range_attempts = 0
    delays = []

    def fake_get(url, **_kwargs):
        nonlocal range_attempts
        if url.endswith(".idx"):
            return Response(200, index.encode("utf-8"))
        range_attempts += 1
        if range_attempts < 3:
            raise SSLError("unexpected eof while reading")
        return Response(206, b"GRIB", {"Content-Range": "bytes 100-249/300"})

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)
    monkeypatch.setattr("haze_forecast.gefs_reforecast.time.sleep", delays.append)

    message = download_gefs_reforecast_message(
        datetime(2015, 1, 1, tzinfo=UTC), member="p02", variable="vgrd_hgt", lead_hours=6,
        level_hint="10 m above ground",
    )

    assert message.content == b"GRIB"
    assert range_attempts == 3
    assert delays == [0.5, 1.0]


def test_reforecast_downloader_does_not_retry_non_transient_http_4xx(monkeypatch):
    index = (
        "1:0:d=2015010100:VGRD:10 m above ground:3 hour fcst:\n"
        "2:100:d=2015010100:VGRD:10 m above ground:6 hour fcst:\n"
        "3:250:d=2015010100:VGRD:10 m above ground:9 hour fcst:"
    )

    class Response:
        def __init__(self, status_code, content, headers=None):
            self.status_code = status_code
            self.content = content
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise HTTPError(f"HTTP {self.status_code}")

    range_attempts = 0
    delays = []

    def fake_get(url, **_kwargs):
        nonlocal range_attempts
        if url.endswith(".idx"):
            return Response(200, index.encode("utf-8"))
        range_attempts += 1
        return Response(404, b"missing")

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)
    monkeypatch.setattr("haze_forecast.gefs_reforecast.time.sleep", delays.append)

    with pytest.raises(HTTPError, match="404"):
        download_gefs_reforecast_message(
            datetime(2015, 1, 1, tzinfo=UTC), member="p02", variable="vgrd_hgt", lead_hours=6,
            level_hint="10 m above ground",
        )

    assert range_attempts == 1
    assert delays == []


def test_reforecast_downloader_retries_temporary_http_5xx(monkeypatch):
    index = (
        "1:0:d=2015010100:TMP:2 m above ground:3 hour fcst:\n"
        "2:100:d=2015010100:TMP:2 m above ground:6 hour fcst:\n"
        "3:250:d=2015010100:TMP:2 m above ground:9 hour fcst:"
    )

    class Response:
        def __init__(self, status_code, content, headers=None):
            self.status_code = status_code
            self.content = content
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise HTTPError(f"HTTP {self.status_code}")

    range_attempts = 0
    delays = []

    def fake_get(url, **_kwargs):
        nonlocal range_attempts
        if url.endswith(".idx"):
            return Response(200, index.encode("utf-8"))
        range_attempts += 1
        if range_attempts == 1:
            return Response(503, b"temporarily unavailable")
        return Response(206, b"GRIB", {"Content-Range": "bytes 100-249/300"})

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)
    monkeypatch.setattr("haze_forecast.gefs_reforecast.time.sleep", delays.append)

    message = download_gefs_reforecast_message(
        datetime(2015, 1, 1, tzinfo=UTC), member="p02", variable="tmp_2m", lead_hours=6
    )

    assert message.content == b"GRIB"
    assert range_attempts == 2
    assert delays == [0.5]


def test_reforecast_downloader_does_not_retry_protocol_validation_failure(monkeypatch):
    index = (
        "1:0:d=2015010100:TMP:2 m above ground:3 hour fcst:\n"
        "2:100:d=2015010100:TMP:2 m above ground:6 hour fcst:\n"
        "3:250:d=2015010100:TMP:2 m above ground:9 hour fcst:"
    )

    class Response:
        def __init__(self, status_code, content, headers=None):
            self.status_code = status_code
            self.content = content
            self.headers = headers or {}

        def raise_for_status(self):
            return None

    range_attempts = 0
    delays = []

    def fake_get(url, **_kwargs):
        nonlocal range_attempts
        if url.endswith(".idx"):
            return Response(200, index.encode("utf-8"))
        range_attempts += 1
        return Response(206, b"GRIB", {"Content-Range": "bytes 0-99/300"})

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)
    monkeypatch.setattr("haze_forecast.gefs_reforecast.time.sleep", delays.append)

    with pytest.raises(RuntimeError, match="byte range"):
        download_gefs_reforecast_message(
            datetime(2015, 1, 1, tzinfo=UTC), member="p02", variable="tmp_2m", lead_hours=6
        )

    assert range_attempts == 1
    assert delays == []


def test_reforecast_downloader_reuses_supplied_index_cache_for_multiple_leads(monkeypatch):
    index = (
        "1:0:d=2015010100:TMP:2 m above ground:3 hour fcst:\n"
        "2:100:d=2015010100:TMP:2 m above ground:6 hour fcst:\n"
        "3:250:d=2015010100:TMP:2 m above ground:9 hour fcst:\n"
        "4:400:d=2015010100:TMP:2 m above ground:12 hour fcst:"
    )

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
            return Response(200, index.encode("utf-8"))
        byte_range = kwargs["headers"]["Range"].removeprefix("bytes=")
        return Response(206, b"GRIB", {"Content-Range": f"bytes {byte_range}/300"})

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)
    cache = GEFSReforecastIndexCache()
    request = {
        "run_time": datetime(2015, 1, 1, tzinfo=UTC),
        "member": "c00",
        "variable": "tmp_2m",
        "index_cache": cache,
    }

    six_hour = download_gefs_reforecast_message(lead_hours=6, **request)
    nine_hour = download_gefs_reforecast_message(lead_hours=9, **request)

    assert [message.byte_range for message in (six_hour, nine_hour)] == [(100, 249), (250, 399)]
    assert [url for url, _kwargs in calls if url.endswith(".idx")] == [six_hour.index_url]


def test_reforecast_index_cache_fetches_different_keys_concurrently(monkeypatch):
    cache = GEFSReforecastIndexCache()
    both_requests_started = Event()
    release_requests = Event()
    calls_lock = Lock()
    call_count = 0

    class Response:
        content = b"1:0:d=2015010100:TMP:2 m above ground:3 hour fcst:"

        def raise_for_status(self):
            return None

    def fake_get(_url, **_kwargs):
        nonlocal call_count
        with calls_lock:
            call_count += 1
            if call_count == 2:
                both_requests_started.set()
        assert release_requests.wait(timeout=2)
        return Response()

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)
    request = {
        "run_time": datetime(2015, 1, 1, tzinfo=UTC),
        "member": "c00",
        "timeout_s": 1,
    }
    threads = [
        Thread(target=cache.get_entries, kwargs={**request, "variable": variable})
        for variable in ("tmp_2m", "pbl_hgt")
    ]
    for thread in threads:
        thread.start()

    requests_were_concurrent = both_requests_started.wait(timeout=0.3)
    release_requests.set()
    for thread in threads:
        thread.join(timeout=2)

    assert requests_were_concurrent
    assert call_count == 2
    assert not any(thread.is_alive() for thread in threads)


def test_reforecast_index_cache_shares_same_in_flight_error(monkeypatch):
    cache = GEFSReforecastIndexCache()
    first_request_started = Event()
    release_request = Event()
    calls_lock = Lock()
    call_count = 0
    expected_error = RuntimeError("index unavailable")

    def fake_get(_url, **_kwargs):
        nonlocal call_count
        with calls_lock:
            call_count += 1
            first_request_started.set()
        assert release_request.wait(timeout=2)
        raise expected_error

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)
    request = {
        "run_time": datetime(2015, 1, 1, tzinfo=UTC),
        "member": "c00",
        "variable": "tmp_2m",
        "timeout_s": 1,
    }
    errors = []

    def read_entries():
        try:
            cache.get_entries(**request)
        except RuntimeError as error:
            errors.append(error)

    first = Thread(target=read_entries)
    second = Thread(target=read_entries)
    first.start()
    assert first_request_started.wait(timeout=1)
    second.start()
    release_request.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert call_count == 1
    assert errors == [expected_error, expected_error]
    assert not first.is_alive()
    assert not second.is_alive()
