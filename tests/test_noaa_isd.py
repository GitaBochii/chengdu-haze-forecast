from __future__ import annotations

import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from haze_forecast import noaa_isd
from haze_forecast.noaa_isd import (
    aggregate_station_hourly,
    annual_csv_url,
    audit_station_hourly,
    download_annual_csv,
    haversine_km,
    nearby_stations,
    prepare_noaa_hourly,
)


def _raw_row(**overrides: str) -> dict[str, str]:
    row = {
        "DATE": "2024-01-01T00:00:00Z",
        "TMP": "0050,1",
        "DEW": "0010,1",
        "SLP": "10134,1",
        "MAA": "2,0015,1",
        "VIS": "016000,1",
        "AA1": "01,0025,1,1",
        "GF1": "05,1,0000,1",
        "WND": "090,1,N,0010,1",
    }
    row.update(overrides)
    return row


def test_haversine_km_keeps_the_60_km_boundary() -> None:
    boundary_latitude = 30.5723 + math.degrees(60.0 / 6371.0088)

    distance = haversine_km(
        pd.Series([104.0665, 104.0665]),
        pd.Series([30.5723, boundary_latitude]),
    )

    assert distance.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert distance.iloc[1] == pytest.approx(60.0, abs=1e-9)


def test_haversine_km_returns_nan_for_invalid_coordinates() -> None:
    distance = haversine_km(
        pd.Series([104.0665, 181.0, 104.0665]),
        pd.Series([30.5723, 30.5723, 90.1]),
    )

    assert distance.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert pd.isna(distance.iloc[1])
    assert pd.isna(distance.iloc[2])


def test_nearby_stations_excludes_a_station_just_outside_the_strict_60_km_limit() -> None:
    inside_latitude = 30.5723 + math.degrees(59.999 / 6371.0088)
    outside_latitude = 30.5723 + math.degrees(60.0000000005 / 6371.0088)
    stations = pd.DataFrame(
        {
            "station": ["inside", "outside"],
            "longitude": [104.0665, 104.0665],
            "latitude": [inside_latitude, outside_latitude],
        }
    )

    nearby = nearby_stations(stations)

    assert nearby["station"].tolist() == ["inside"]
    assert nearby.loc[0, "distance_km"] < 60.0


def test_nearby_stations_excludes_invalid_coordinates() -> None:
    stations = pd.DataFrame(
        {
            "station": ["center", "bad_longitude", "bad_latitude"],
            "longitude": [104.0665, -180.1, 104.0665],
            "latitude": [30.5723, 30.5723, 90.1],
        }
    )

    nearby = nearby_stations(stations)

    assert nearby["station"].tolist() == ["center"]


def test_nearby_stations_keeps_a_station_exactly_on_the_60_km_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stations = pd.DataFrame({"longitude": [104.0], "latitude": [30.0]})

    def exact_boundary_distance(*_args: pd.Series) -> pd.Series:
        return pd.Series([60.0])

    monkeypatch.setattr(noaa_isd, "haversine_km", exact_boundary_distance)

    nearby = nearby_stations(stations)

    assert nearby["distance_km"].tolist() == [60.0]


def test_prepare_noaa_hourly_parses_values_and_converts_utc_to_china_time() -> None:
    prepared = prepare_noaa_hourly(pd.DataFrame([_raw_row()]))

    assert prepared.columns.tolist() == [
        "timestamp",
        "noaa_temperature_c",
        "noaa_dewpoint_c",
        "noaa_sea_level_pressure_hpa",
        "noaa_pressure_change_hpa",
        "noaa_visibility_m",
        "noaa_precipitation_mm",
        "noaa_cloud_cover_oktas",
        "noaa_wind_direction_deg",
        "noaa_wind_speed_ms",
    ]
    assert prepared.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 08:00:00+08:00")
    assert prepared.loc[0, "noaa_temperature_c"] == 5.0
    assert prepared.loc[0, "noaa_dewpoint_c"] == 1.0
    assert prepared.loc[0, "noaa_sea_level_pressure_hpa"] == 1013.4
    assert prepared.loc[0, "noaa_pressure_change_hpa"] == 1.5
    assert prepared.loc[0, "noaa_visibility_m"] == 16000.0
    assert prepared.loc[0, "noaa_precipitation_mm"] == 2.5
    assert prepared.loc[0, "noaa_cloud_cover_oktas"] == 5.0
    assert prepared.loc[0, "noaa_wind_direction_deg"] == 90.0
    assert prepared.loc[0, "noaa_wind_speed_ms"] == 1.0


def test_prepare_noaa_hourly_keeps_records_when_optional_fields_are_absent() -> None:
    raw_row = _raw_row()
    del raw_row["MAA"]
    del raw_row["GF1"]

    prepared = prepare_noaa_hourly(pd.DataFrame([raw_row]))

    assert prepared["timestamp"].tolist() == [pd.Timestamp("2024-01-01 08:00:00+08:00")]
    assert prepared["noaa_temperature_c"].tolist() == [5.0]
    assert pd.isna(prepared.loc[0, "noaa_pressure_change_hpa"])
    assert pd.isna(prepared.loc[0, "noaa_cloud_cover_oktas"])


def test_prepare_noaa_hourly_rejects_rows_without_date_column() -> None:
    raw_row = _raw_row()
    del raw_row["DATE"]

    with pytest.raises(ValueError, match="DATE"):
        prepare_noaa_hourly(pd.DataFrame([raw_row]))


def test_prepare_noaa_hourly_replaces_missing_sentinels_and_invalid_values() -> None:
    prepared = prepare_noaa_hourly(
        pd.DataFrame(
            [
                _raw_row(
                    TMP="9999,9",
                    DEW="9999,9",
                    SLP="99999,9",
                    MAA="9,9999,9",
                    VIS="160001,1",
                    AA1="01,9999,9,9",
                    GF1="99,9,9999,9",
                    WND="999,9,N,1501,9",
                )
            ]
        )
    )

    assert pd.isna(prepared.loc[0, "noaa_temperature_c"])
    assert pd.isna(prepared.loc[0, "noaa_dewpoint_c"])
    assert pd.isna(prepared.loc[0, "noaa_sea_level_pressure_hpa"])
    assert pd.isna(prepared.loc[0, "noaa_pressure_change_hpa"])
    assert pd.isna(prepared.loc[0, "noaa_visibility_m"])
    assert pd.isna(prepared.loc[0, "noaa_precipitation_mm"])
    assert pd.isna(prepared.loc[0, "noaa_cloud_cover_oktas"])
    assert pd.isna(prepared.loc[0, "noaa_wind_direction_deg"])
    assert pd.isna(prepared.loc[0, "noaa_wind_speed_ms"])


def test_prepare_noaa_hourly_discards_values_with_a_bad_quality_code() -> None:
    prepared = prepare_noaa_hourly(pd.DataFrame([_raw_row(TMP="0050,3")]))

    assert pd.isna(prepared.loc[0, "noaa_temperature_c"])


def test_prepare_noaa_hourly_discards_bad_quality_codes_for_every_other_field() -> None:
    prepared = prepare_noaa_hourly(
        pd.DataFrame(
            [
                _raw_row(
                    DEW="0010,3",
                    SLP="10134,3",
                    MAA="2,0015,3",
                    VIS="016000,3",
                    AA1="01,0025,1,3",
                    GF1="05,3,0000,1",
                    WND="090,3,N,0010,3",
                )
            ]
        )
    )

    assert pd.isna(prepared.loc[0, "noaa_dewpoint_c"])
    assert pd.isna(prepared.loc[0, "noaa_sea_level_pressure_hpa"])
    assert pd.isna(prepared.loc[0, "noaa_pressure_change_hpa"])
    assert pd.isna(prepared.loc[0, "noaa_visibility_m"])
    assert pd.isna(prepared.loc[0, "noaa_precipitation_mm"])
    assert pd.isna(prepared.loc[0, "noaa_cloud_cover_oktas"])
    assert pd.isna(prepared.loc[0, "noaa_wind_direction_deg"])
    assert pd.isna(prepared.loc[0, "noaa_wind_speed_ms"])


def test_prepare_noaa_hourly_uses_the_next_available_hour_for_a_non_hourly_observation() -> None:
    prepared = prepare_noaa_hourly(
        pd.DataFrame(
            [
                _raw_row(DATE="2024-01-01T00:50:00Z", TMP="0050,1"),
            ]
        )
    )

    assert prepared["timestamp"].tolist() == [pd.Timestamp("2024-01-01 09:00:00+08:00")]


def test_prepare_noaa_hourly_keeps_an_observation_exactly_on_the_hour() -> None:
    prepared = prepare_noaa_hourly(
        pd.DataFrame(
            [
                _raw_row(DATE="2024-01-01T00:00:00Z", TMP="0050,1"),
            ]
        )
    )

    assert prepared["timestamp"].tolist() == [pd.Timestamp("2024-01-01 08:00:00+08:00")]
    assert prepared["noaa_temperature_c"].tolist() == [5.0]


def test_prepare_noaa_hourly_prefers_more_valid_weather_fields_within_available_hour() -> None:
    prepared = prepare_noaa_hourly(
        pd.DataFrame(
            [
                _raw_row(DATE="2024-01-01T00:10:00Z", TMP="0050,1"),
                _raw_row(
                    DATE="2024-01-01T00:50:00Z",
                    TMP="0050,3",
                    DEW="0010,3",
                    SLP="10134,3",
                    MAA="2,0015,3",
                    VIS="016000,3",
                    AA1="01,0025,1,3",
                    GF1="05,3,0000,1",
                    WND="090,3,N,0010,3",
                ),
            ]
        )
    )

    assert prepared["timestamp"].tolist() == [pd.Timestamp("2024-01-01 09:00:00+08:00")]
    assert prepared.loc[0, "noaa_temperature_c"] == 5.0
    assert prepared.loc[0, "noaa_visibility_m"] == 16000.0


def test_prepare_noaa_hourly_uses_later_observation_when_valid_field_counts_tie() -> None:
    prepared = prepare_noaa_hourly(
        pd.DataFrame(
            [
                _raw_row(DATE="2024-01-01T00:10:00Z", TMP="0050,1"),
                _raw_row(DATE="2024-01-01T00:50:00Z", TMP="0070,1"),
            ]
        )
    )

    assert prepared["timestamp"].tolist() == [pd.Timestamp("2024-01-01 09:00:00+08:00")]
    assert prepared["noaa_temperature_c"].tolist() == [7.0]


def _prepared_hourly(year: int, count: int, temperature_c: float = 5.0) -> pd.DataFrame:
    timestamps = pd.date_range(
        f"{year}-01-01", periods=count, freq="h", tz="Asia/Shanghai"
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "noaa_temperature_c": temperature_c,
            "noaa_visibility_m": 16000.0,
        }
    )


def test_annual_csv_url_uses_ncei_global_hourly_layout() -> None:
    assert annual_csv_url("562940", "99999", 2021) == (
        "https://www.ncei.noaa.gov/data/global-hourly/access/2021/56294099999.csv"
    )


def test_download_annual_csv_caches_bytes_and_returns_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "2021" / "56294099999.csv"
    payload = b"DATE,TMP\n2021-01-01T00:00:00Z,0050,1\n"
    calls: list[tuple[str, int]] = []

    class Response:
        content = payload

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, *, timeout: int) -> Response:
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(noaa_isd.requests, "get", fake_get)

    digest = download_annual_csv("https://example.test/source.csv", destination)

    assert destination.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert calls == [("https://example.test/source.csv", 60)]
    assert download_annual_csv("https://example.test/source.csv", destination) == digest
    assert calls == [("https://example.test/source.csv", 60)]


def test_audit_station_hourly_rejects_station_below_one_year_coverage_threshold() -> None:
    result, report = audit_station_hourly(
        {"usaf": "562940", "wban": "99999", "latitude": 30.6, "longitude": 104.0},
        {2020: _prepared_hourly(2020, 6130)},
        years=[2020],
        minimum_coverage=0.70,
        annual_sources={2020: {"url": "https://example.test/2020.csv", "sha256": "abc"}},
    )

    assert result is None
    assert report["eligible"] is False
    assert report["years"]["2020"]["expected_hours"] == 8784
    assert report["years"]["2020"]["coverage"] < 0.70
    assert "coverage" in report["failure_reasons"][0]
    assert report["years"]["2020"]["source_url"] == "https://example.test/2020.csv"
    assert report["years"]["2020"]["sha256"] == "abc"


def test_audit_station_hourly_uses_default_70_percent_coverage_threshold() -> None:
    result, report = audit_station_hourly(
        {"usaf": "562940", "wban": "99999"},
        {2020: _prepared_hourly(2020, 6130)},
        years=[2020],
    )

    assert result is None
    assert report["eligible"] is False
    assert report["minimum_coverage"] == 0.70
    assert report["years"]["2020"]["coverage"] < 0.70


def test_audit_station_hourly_accepts_complete_local_calendar_year() -> None:
    hourly = _prepared_hourly(2021, 8760)

    result, report = audit_station_hourly(
        {"usaf": "562940", "wban": "99999"},
        {2021: hourly},
        years=[2021],
        minimum_coverage=0.70,
    )

    assert report["eligible"] is True
    assert report["years"]["2021"]["observed_hours"] == 8760
    assert report["years"]["2021"]["coverage"] == pytest.approx(1.0)
    assert report["years"]["2021"]["variable_non_null_rate"]["noaa_temperature_c"] == 1.0
    assert result is not None
    assert len(result) == 8760


def test_aggregate_station_hourly_uses_city_level_median_per_timestamp() -> None:
    first = _prepared_hourly(2021, 2, temperature_c=2.0)
    second = _prepared_hourly(2021, 2, temperature_c=8.0)
    third = _prepared_hourly(2021, 2, temperature_c=20.0)

    aggregated = aggregate_station_hourly([first, second, third])

    assert aggregated["timestamp"].tolist() == first["timestamp"].tolist()
    assert aggregated["noaa_temperature_c"].tolist() == [8.0, 8.0]


def test_audit_noaa_isd_records_every_requested_year_for_an_invalid_station(
    tmp_path: Path,
) -> None:
    _, report = noaa_isd.audit_noaa_isd(
        pd.DataFrame(
            {
                "usaf": ["not-a-station"],
                "wban": ["99999"],
                "longitude": [104.0665],
                "latitude": [30.5723],
            }
        ),
        tmp_path / "raw",
        years=[2019, 2020],
    )

    station_report = report["stations"][0]

    assert station_report["eligible"] is False
    assert station_report["variable_non_null_rate"] == {}
    assert set(station_report["years"]) == {"2019", "2020"}
    for year, annual_report in station_report["years"].items():
        assert annual_report["year"] == int(year)
        assert annual_report["source_url"] is None
        assert annual_report["sha256"] is None
        assert annual_report["coverage"] == 0.0
        assert annual_report["variable_non_null_rate"] == {}
        assert annual_report["failure_reason"]
    json.dumps(report)


def test_audit_noaa_isd_combines_previous_utc_file_for_first_local_year_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary_hours = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01 00:00", periods=8, freq="h", tz="Asia/Shanghai"
            ),
            "noaa_temperature_c": 1.0,
        }
    )
    target_hours = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01 08:00", periods=8776, freq="h", tz="Asia/Shanghai"
            ),
            "noaa_temperature_c": 5.0,
        }
    )
    prepared_by_utc_year = {2019: boundary_hours, 2020: target_hours}
    downloaded_years: list[int] = []

    def fake_download(_url: str, destination: Path, timeout_s: int = 60) -> str:
        del timeout_s
        year = int(destination.parent.name)
        downloaded_years.append(year)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(year), encoding="utf-8")
        return f"hash-{year}"

    def fake_read_csv(path: Path, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"utc_year": [int(path.read_text(encoding="utf-8"))]})

    def fake_prepare(raw: pd.DataFrame) -> pd.DataFrame:
        return prepared_by_utc_year[int(raw.loc[0, "utc_year"])].copy()

    monkeypatch.setattr(noaa_isd, "download_annual_csv", fake_download)
    monkeypatch.setattr(noaa_isd.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(noaa_isd, "prepare_noaa_hourly", fake_prepare)

    city_hourly, report = noaa_isd.audit_noaa_isd(
        pd.DataFrame(
            {
                "usaf": ["562940"],
                "wban": ["99999"],
                "longitude": [104.0665],
                "latitude": [30.5723],
            }
        ),
        tmp_path / "raw",
        years=[2020],
        minimum_coverage=1.0,
    )

    assert downloaded_years == [2019, 2020]
    assert report["stations"][0]["years"]["2020"]["coverage"] == 1.0
    assert report["stations"][0]["boundary_sources"]["2020"] == {
        "source_utc_year": 2019,
        "source_url": annual_csv_url("562940", "99999", 2019),
        "sha256": "hash-2019",
        "failure_reason": None,
    }
    assert city_hourly is not None
    assert city_hourly["timestamp"].iloc[:8].tolist() == boundary_hours["timestamp"].tolist()


def test_audit_noaa_isd_reports_failed_boundary_source_without_fabricating_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_hours = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01 08:00", periods=8776, freq="h", tz="Asia/Shanghai"
            ),
            "noaa_temperature_c": 5.0,
        }
    )

    def fake_download(_url: str, destination: Path, timeout_s: int = 60) -> str:
        del timeout_s
        year = int(destination.parent.name)
        if year == 2019:
            raise noaa_isd.requests.RequestException("boundary unavailable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(year), encoding="utf-8")
        return "hash-2020"

    def fake_read_csv(_path: Path, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"utc_year": [2020]})

    monkeypatch.setattr(noaa_isd, "download_annual_csv", fake_download)
    monkeypatch.setattr(noaa_isd.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(noaa_isd, "prepare_noaa_hourly", lambda _raw: target_hours.copy())

    city_hourly, report = noaa_isd.audit_noaa_isd(
        pd.DataFrame(
            {
                "usaf": ["562940"],
                "wban": ["99999"],
                "longitude": [104.0665],
                "latitude": [30.5723],
            }
        ),
        tmp_path / "raw",
        years=[2020],
    )

    assert city_hourly is None
    station_report = report["stations"][0]
    annual_report = station_report["years"]["2020"]
    assert station_report["eligible"] is False
    assert annual_report["coverage"] == pytest.approx(8776 / 8784)
    assert any("boundary source 2019 failed" in reason for reason in station_report["failure_reasons"])
    boundary_report = station_report["boundary_sources"]["2020"]
    assert boundary_report["source_utc_year"] == 2019
    assert boundary_report["sha256"] is None
    assert "boundary unavailable" in boundary_report["failure_reason"]


def test_audit_noaa_isd_uses_a_boundary_source_for_each_non_contiguous_target_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared_by_utc_year = {
        2017: pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2018-01-01 00:00", periods=8, freq="h", tz="Asia/Shanghai"
                ),
                "noaa_temperature_c": 1.0,
            }
        ),
        2018: pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2018-01-01 08:00", periods=8760, freq="h", tz="Asia/Shanghai"
                ),
                "noaa_temperature_c": 2.0,
            }
        ),
        2019: pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2019-01-01 08:00", periods=8760, freq="h", tz="Asia/Shanghai"
                ),
                "noaa_temperature_c": 3.0,
            }
        ),
        2020: pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01 08:00", periods=8784, freq="h", tz="Asia/Shanghai"
                ),
                "noaa_temperature_c": 4.0,
            }
        ),
    }
    downloaded_years: list[int] = []

    def fake_download(_url: str, destination: Path, timeout_s: int = 60) -> str:
        del timeout_s
        year = int(destination.parent.name)
        downloaded_years.append(year)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(year), encoding="utf-8")
        return f"hash-{year}"

    def fake_read_csv(path: Path, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"utc_year": [int(path.read_text(encoding="utf-8"))]})

    monkeypatch.setattr(noaa_isd, "download_annual_csv", fake_download)
    monkeypatch.setattr(noaa_isd.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(
        noaa_isd,
        "prepare_noaa_hourly",
        lambda raw: prepared_by_utc_year[int(raw.loc[0, "utc_year"])].copy(),
    )

    city_hourly, report = noaa_isd.audit_noaa_isd(
        pd.DataFrame(
            {
                "usaf": ["562940"],
                "wban": ["99999"],
                "longitude": [104.0665],
                "latitude": [30.5723],
            }
        ),
        tmp_path / "raw",
        years=[2018, 2020],
        minimum_coverage=1.0,
    )

    assert downloaded_years == [2017, 2018, 2019, 2020]
    station_report = report["stations"][0]
    assert station_report["eligible"] is True
    assert station_report["years"]["2018"]["coverage"] == 1.0
    assert station_report["years"]["2020"]["coverage"] == 1.0
    assert station_report["boundary_sources"]["2018"]["source_utc_year"] == 2017
    assert station_report["boundary_sources"]["2020"]["source_utc_year"] == 2019
    assert city_hourly is not None
    assert city_hourly.loc[city_hourly["timestamp"].dt.year.eq(2020), "timestamp"].iloc[:8].tolist() == (
        prepared_by_utc_year[2019]["timestamp"].iloc[-8:].tolist()
    )


def test_noaa_audit_cli_does_not_write_parquet_without_an_eligible_station(tmp_path: Path) -> None:
    station_list = tmp_path / "stations.csv"
    station_list.write_text(
        "usaf,wban,longitude,latitude\n562940,99999,0.0,0.0\n", encoding="utf-8"
    )
    raw_dir = tmp_path / "raw"
    output = tmp_path / "report.json"
    stale_parquet = output.parent / "noaa_isd_hourly.parquet"
    stale_parquet.write_bytes(b"previous successful output")
    command = [
        sys.executable,
        "scripts/audit_noaa_isd_meteorology.py",
        "--station-list",
        str(station_list),
        "--raw-dir",
        str(raw_dir),
        "--output",
        str(output),
    ]

    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"PYTHONPATH": str(Path.cwd() / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert not stale_parquet.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["city_hourly_sha256"] is None
    assert report["eligible"] is False


def test_noaa_audit_cli_hashes_the_exact_published_hourly_parquet(tmp_path, monkeypatch):
    hourly = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="Asia/Shanghai"),
            "noaa_temperature_c": [1.0, 2.0],
        }
    )
    report = {"eligible_station_count": 1, "city_hourly_rows": 2, "minimum_coverage": 0.70}
    monkeypatch.setattr(noaa_isd, "audit_noaa_isd", lambda *args, **kwargs: (hourly, report))
    station_list = tmp_path / "stations.csv"
    station_list.write_text("usaf,wban,longitude,latitude\n562940,99999,104.0,30.6\n", encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_noaa_isd_meteorology.py",
            "--station-list",
            str(station_list),
            "--output",
            str(output),
        ],
    )

    runpy.run_path("scripts/audit_noaa_isd_meteorology.py", run_name="__main__")
    hourly_path = output.parent / "noaa_isd_hourly.parquet"
    published_report = json.loads(output.read_text(encoding="utf-8"))
    assert published_report["city_hourly_sha256"] == hashlib.sha256(
        hourly_path.read_bytes()
    ).hexdigest()
    assert published_report["eligible"] is True


@pytest.mark.parametrize("minimum_coverage", ["0.69", "nan"])
def test_noaa_audit_cli_rejects_minimum_coverage_below_70_percent(monkeypatch, minimum_coverage):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_noaa_isd_meteorology.py",
            "--station-list",
            "stations.csv",
            "--minimum-coverage",
            minimum_coverage,
        ],
    )
    cli = runpy.run_path("scripts/audit_noaa_isd_meteorology.py", run_name="noaa_audit_cli_test")

    with pytest.raises(SystemExit) as error:
        cli["parse_args"]()

    assert error.value.code == 2


def test_noaa_audit_cli_rejects_a_single_2020_year(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_noaa_isd_meteorology.py",
            "--station-list",
            "stations.csv",
            "--years",
            "2020",
        ],
    )
    cli = runpy.run_path("scripts/audit_noaa_isd_meteorology.py", run_name="noaa_audit_cli_test")

    with pytest.raises(SystemExit) as error:
        cli["parse_args"]()

    assert error.value.code == 2
