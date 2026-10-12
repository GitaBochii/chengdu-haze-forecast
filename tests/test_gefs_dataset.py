import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from requests.exceptions import SSLError

from haze_forecast.gefs_dataset import (
    _relative_humidity_percent,
    enrich_event_rows_with_gefs_pbl,
    enrich_event_rows_with_gefs_surface,
)
from haze_forecast.gefs_reforecast import GEFSReforecastMessage
from haze_forecast.grib import GRIBPointValue


def test_relative_humidity_percent_matches_reference_thermodynamic_value():
    relative_humidity = _relative_humidity_percent(
        temperature_k=293.15,
        specific_humidity_kg_kg=0.01,
        surface_pressure_pa=101325.0,
    )

    assert relative_humidity == pytest.approx(69.28608902019099)


def test_surface_enrichment_extracts_all_members_with_time_safe_provenance(tmp_path):
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2015-01-01 08:00:00+00:00", "2015-01-01 09:00:00+00:00"]
            )
        }
    )
    member_offsets = {"c00": 0.0, "p01": 1.0, "p02": 2.0, "p03": 3.0, "p04": 4.0}
    base_values = {
        "tmp_2m": 280.0,
        "spfh_2m": 0.004,
        "ugrd_hgt": 2.0,
        "vgrd_hgt": 4.0,
        "pres_sfc": 100000.0,
        "pbl_hgt": 100.0,
    }
    calls = []

    def downloader(run_time, *, member, variable, lead_hours, level_hint=None, **_kwargs):
        calls.append((run_time, member, variable, lead_hours, level_hint))
        content = f"{variable}/{member}".encode("ascii")
        return GEFSReforecastMessage(
            run_time=run_time,
            member=member,
            variable=variable,
            lead_hours=lead_hours,
            source_url=f"https://example.test/{variable}/{member}",
            index_url=f"https://example.test/{variable}/{member}.idx",
            byte_range=(10, 20),
            content=content,
            sha256=f"sha-{variable}-{member}",
        )

    def point_query(path: Path, *_args, **_kwargs):
        variable, member = path.read_bytes().decode("ascii").split("/")
        offset = member_offsets[member]
        if variable == "spfh_2m":
            offset /= 1000.0
        return GRIBPointValue(
            longitude=104.0,
            latitude=30.5,
            value=base_values[variable] + offset,
        )

    enriched = enrich_event_rows_with_gefs_surface(
        rows,
        longitude=104.0665,
        latitude=30.5723,
        wgrib2_executable=tmp_path / "wgrib2.exe",
        downloader=downloader,
        point_query=point_query,
    )

    assert len(calls) == 30
    assert {call[4] for call in calls if call[2] in {"ugrd_hgt", "vgrd_hgt"}} == {
        "10 m above ground"
    }
    assert {call[4] for call in calls if call[2] not in {"ugrd_hgt", "vgrd_hgt"}} == {None}
    assert enriched["gefs_tmp_2m_median_k"].tolist() == [282.0, 282.0]
    assert enriched["gefs_spfh_2m_p10_kg_kg"].tolist() == pytest.approx([0.0044, 0.0044])
    assert enriched["gefs_pres_sfc_p90_pa"].tolist() == [100003.6, 100003.6]
    assert enriched["gefs_pblh_median_m"].tolist() == [102.0, 102.0]
    assert enriched["gefs_relative_humidity_median_pct"].tolist() == pytest.approx(
        [84.62064198331677, 84.62064198331677]
    )
    assert enriched["gefs_relative_humidity_std_pct"].tolist() == pytest.approx(
        [12.000209949135447, 12.000209949135447]
    )
    assert enriched["gefs_relative_humidity_p10_pct"].tolist() == pytest.approx(
        [69.03847067746128, 69.03847067746128]
    )
    assert enriched["gefs_relative_humidity_p90_pct"].tolist() == pytest.approx(
        [96.02963548342636, 96.02963548342636]
    )
    assert enriched["gefs_forecast_valid_utc"].ge(
        pd.to_datetime(enriched["timestamp"], utc=True) + timedelta(hours=6)
    ).all()
    assert (
        enriched["gefs_run_utc"]
        <= pd.to_datetime(enriched["timestamp"], utc=True)
    ).all()
    assert (
        pd.to_datetime(enriched["timestamp"], utc=True) < enriched["gefs_forecast_valid_utc"]
    ).all()

    manifest = json.loads(enriched.iloc[0]["gefs_source_manifest_json"])
    assert manifest["run_utc"] == "2015-01-01T00:00:00+00:00"
    assert manifest["lead_hours"] == 15
    assert len(manifest["messages"]) == 30
    assert manifest["messages"][0] == {
        "variable": "tmp_2m",
        "member": "c00",
        "source_url": "https://example.test/tmp_2m/c00",
        "index_url": "https://example.test/tmp_2m/c00.idx",
        "byte_range": [10, 20],
        "sha256": "sha-tmp_2m-c00",
        "run_utc": "2015-01-01T00:00:00+00:00",
        "lead_hours": 15,
    }


def test_surface_enrichment_rejects_downloader_identity_mismatch(tmp_path):
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2015-01-01 08:00:00+00:00"])})

    def downloader(run_time, *, member, variable, lead_hours, **_kwargs):
        return GEFSReforecastMessage(
            run_time=datetime(2015, 1, 2, tzinfo=UTC),
            member=member,
            variable=variable,
            lead_hours=lead_hours,
            source_url="https://example.test/data",
            index_url="https://example.test/index",
            byte_range=(0, 3),
            content=b"GRIB",
            sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="identity"):
        enrich_event_rows_with_gefs_surface(
            rows,
            longitude=104.0665,
            latitude=30.5723,
            wgrib2_executable=tmp_path / "wgrib2.exe",
            downloader=downloader,
            point_query=lambda *_args, **_kwargs: GRIBPointValue(104.0, 30.5, 1.0),
        )


def test_surface_enrichment_failure_note_names_variable_and_member(tmp_path):
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2015-01-01 08:00:00+00:00"])})

    def downloader(*_args, **_kwargs):
        raise SSLError("unexpected eof while reading")

    with pytest.raises(SSLError) as raised:
        enrich_event_rows_with_gefs_surface(
            rows,
            longitude=104.0665,
            latitude=30.5723,
            wgrib2_executable=tmp_path / "wgrib2.exe",
            variables=("vgrd_hgt",),
            downloader=downloader,
            point_query=lambda *_args, **_kwargs: GRIBPointValue(104.0, 30.5, 1.0),
        )

    assert len(raised.value.__notes__) == 1
    assert raised.value.__notes__[0].startswith(
        "GEFSv12 surface extraction failed for variable=vgrd_hgt, member="
    )
    assert raised.value.__notes__[0].removeprefix(
        "GEFSv12 surface extraction failed for variable=vgrd_hgt, member="
    ) in {"c00", "p01", "p02", "p03", "p04"}


def test_surface_enrichment_rejects_horizon_shorter_than_mandatory_six_hours(tmp_path):
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2015-01-01 08:00:00+00:00"])})

    def downloader(run_time, *, member, variable, lead_hours, **_kwargs):
        return GEFSReforecastMessage(
            run_time=run_time,
            member=member,
            variable=variable,
            lead_hours=lead_hours,
            source_url="https://example.test/data",
            index_url="https://example.test/index",
            byte_range=(0, 3),
            content=b"GRIB",
            sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="at least 6"):
        enrich_event_rows_with_gefs_pbl(
            rows,
            longitude=104.0665,
            latitude=30.5723,
            wgrib2_executable=tmp_path / "wgrib2.exe",
            horizon_hours=3,
            downloader=downloader,
            point_query=lambda *_args, **_kwargs: GRIBPointValue(104.0, 30.5, 1.0),
        )


def test_gefs_pbl_enrichment_caches_members_for_rows_with_same_run_and_lead(tmp_path):
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2015-01-01 08:00:00+00:00", "2015-01-01 09:00:00+00:00"]
            ),
            "remaining_hours": [12.0, 11.0],
        }
    )
    values = {"c00": 100.0, "p01": 110.0, "p02": 120.0, "p03": 130.0, "p04": 140.0}
    calls = []

    def downloader(run_time, *, member, variable, lead_hours, **_kwargs):
        calls.append((run_time, member, variable, lead_hours))
        return GEFSReforecastMessage(
            run_time=run_time,
            member=member,
            variable=variable,
            lead_hours=lead_hours,
            source_url="https://example.test/data",
            index_url="https://example.test/index",
            byte_range=(0, 3),
            content=member.encode("ascii"),
            sha256="0" * 64,
        )

    def point_query(path: Path, *_args, **_kwargs):
        assert path.read_bytes() == path.stem.split("_")[-1].encode("ascii")
        return GRIBPointValue(longitude=104.0, latitude=30.5, value=values[path.stem.split("_")[-1]])

    enriched = enrich_event_rows_with_gefs_pbl(
        rows,
        longitude=104.0665,
        latitude=30.5723,
        wgrib2_executable=tmp_path / "wgrib2.exe",
        downloader=downloader,
        point_query=point_query,
    )

    assert len(calls) == 5
    assert {call[3] for call in calls} == {15}
    assert enriched["gefs_run_utc"].tolist() == [
        pd.Timestamp("2015-01-01 00:00:00+00:00"),
        pd.Timestamp("2015-01-01 00:00:00+00:00"),
    ]
    assert enriched["gefs_pblh_median_m"].tolist() == [120.0, 120.0]
    assert enriched["gefs_pblh_p10_m"].tolist() == [104.0, 104.0]
    assert enriched["gefs_pblh_p90_m"].tolist() == [136.0, 136.0]
    assert enriched["gefs_forecast_valid_utc"].ge(
        pd.to_datetime(enriched["timestamp"], utc=True) + timedelta(hours=6)
    ).all()


def test_surface_enrichment_parallelises_members_and_keeps_manifest_member_order(tmp_path):
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2015-01-01 08:00:00+00:00"])})
    values = {"c00": 100.0, "p01": 110.0, "p02": 120.0, "p03": 130.0, "p04": 140.0}
    active = 0
    peak_active = 0
    lock = threading.Lock()
    both_default_workers_started = threading.Event()

    def downloader(run_time, *, member, variable, lead_hours, **_kwargs):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            if active == 2:
                both_default_workers_started.set()
        assert both_default_workers_started.wait(timeout=1)
        with lock:
            active -= 1
        return GEFSReforecastMessage(
            run_time=run_time,
            member=member,
            variable=variable,
            lead_hours=lead_hours,
            source_url=f"https://example.test/{member}",
            index_url=f"https://example.test/{member}.idx",
            byte_range=(0, 3),
            content=member.encode("ascii"),
            sha256=f"sha-{member}",
        )

    def point_query(path: Path, *_args, **_kwargs):
        member = path.read_bytes().decode("ascii")
        return GRIBPointValue(longitude=104.0, latitude=30.5, value=values[member])

    enriched = enrich_event_rows_with_gefs_surface(
        rows,
        longitude=104.0665,
        latitude=30.5723,
        wgrib2_executable=tmp_path / "wgrib2.exe",
        variables=("pbl_hgt",),
        downloader=downloader,
        point_query=point_query,
    )

    assert peak_active == 2
    assert enriched["gefs_pblh_median_m"].tolist() == [120.0]
    manifest = json.loads(enriched.iloc[0]["gefs_source_manifest_json"])
    assert [message["member"] for message in manifest["messages"]] == [
        "c00",
        "p01",
        "p02",
        "p03",
        "p04",
    ]


def test_surface_enrichment_reuses_each_member_index_for_multiple_leads(tmp_path, monkeypatch):
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2015-01-01 08:00:00+00:00", "2015-01-01 11:00:00+00:00"]
            )
        }
    )
    index = (
        "1:0:d=2015010100:HGT:surface:12 hour fcst:\n"
        "2:100:d=2015010100:HGT:surface:15 hour fcst:\n"
        "3:200:d=2015010100:HGT:surface:18 hour fcst:\n"
        "4:300:d=2015010100:HGT:surface:21 hour fcst:"
    )
    calls = []

    class Response:
        def __init__(self, status_code, content, headers=None):
            self.status_code = status_code
            self.content = content
            self.headers = headers or {}

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith(".idx"):
            return Response(200, index.encode("utf-8"))
        member = next(member for member in ("c00", "p01", "p02", "p03", "p04") if f"/{member}/" in url)
        byte_range = kwargs["headers"]["Range"].removeprefix("bytes=")
        return Response(206, member.encode("ascii"), {"Content-Range": f"bytes {byte_range}/999"})

    monkeypatch.setattr("haze_forecast.gefs_reforecast.requests.get", fake_get)

    enriched = enrich_event_rows_with_gefs_surface(
        rows,
        longitude=104.0665,
        latitude=30.5723,
        wgrib2_executable=tmp_path / "wgrib2.exe",
        variables=("pbl_hgt",),
        point_query=lambda path, *_args, **_kwargs: GRIBPointValue(
            104.0, 30.5, float(path.read_bytes()[-1])
        ),
    )

    assert len([url for url, _kwargs in calls if url.endswith(".idx")]) == 5
    assert enriched["gefs_lead_hours"].tolist() == [15, 18]
    manifests = [json.loads(value) for value in enriched["gefs_source_manifest_json"]]
    assert all([message["member"] for message in manifest["messages"]] == ["c00", "p01", "p02", "p03", "p04"] for manifest in manifests)
