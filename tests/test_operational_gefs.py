from datetime import UTC, datetime

import pytest

from haze_forecast.operational_gefs import (
    gefs_data_url,
    select_operational_gefs_run,
    validate_operational_gefs_timing,
)


def test_gefs_url_uses_the_operational_archive_layout():
    run = datetime(2024, 1, 1, 0, tzinfo=UTC)

    assert gefs_data_url(run, product="geavg", lead_hours=6).endswith(
        "gefs.20240101/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f006"
    )
    assert gefs_data_url(run, product="gec00", lead_hours=6).endswith(
        "gefs.20240101/00/atmos/pgrb2bp5/gec00.t00z.pgrb2b.0p50.f006"
    )


def test_gefs_selects_only_an_already_issued_run_and_future_valid_times():
    issue = datetime(2024, 1, 1, 14, tzinfo=UTC)

    run, valid_times = select_operational_gefs_run(issue)

    assert run == datetime(2024, 1, 1, 12, tzinfo=UTC)
    assert valid_times == {
        6: datetime(2024, 1, 1, 18, tzinfo=UTC),
        24: datetime(2024, 1, 2, 12, tzinfo=UTC),
    }


def test_timing_rejects_a_forecast_that_was_not_issued():
    with pytest.raises(ValueError, match="issued after"):
        validate_operational_gefs_timing(
            datetime(2024, 1, 1, 6, tzinfo=UTC),
            datetime(2024, 1, 1, 12, tzinfo=UTC),
            datetime(2024, 1, 1, 18, tzinfo=UTC),
        )


class _Response:
    def __init__(self, *, status_code: int, content: bytes, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None


def test_downloader_rejects_a_full_object_response(monkeypatch):
    from haze_forecast.operational_gefs import download_operational_gefs_surface_messages

    index = b"\n".join(
        [
            b"1:0:d=2024010100:TMP:2 m above ground:6 hour fcst",
            b"2:10:d=2024010100:RH:2 m above ground:6 hour fcst",
            b"3:20:d=2024010100:UGRD:10 m above ground:6 hour fcst",
            b"4:30:d=2024010100:VGRD:10 m above ground:6 hour fcst",
            b"5:40:d=2024010100:HPBL:surface:6 hour fcst",
            b"6:50:d=2024010100:PRMSL:mean sea level:6 hour fcst",
        ]
    )

    def fake_get(url: str, **kwargs: object) -> _Response:
        return _Response(status_code=200, content=index if url.endswith(".idx") else b"entire-file")

    monkeypatch.setattr("haze_forecast.operational_gefs.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="HTTP 206"):
        download_operational_gefs_surface_messages(
            datetime(2024, 1, 1, 0, tzinfo=UTC), product="geavg", lead_hours=6
        )


def test_downloader_rejects_non_positive_parallelism():
    from haze_forecast.operational_gefs import download_operational_gefs_surface_messages

    with pytest.raises(ValueError, match="max_workers"):
        download_operational_gefs_surface_messages(
            datetime(2024, 1, 1, 0, tzinfo=UTC),
            product="geavg",
            lead_hours=6,
            max_workers=0,
        )
