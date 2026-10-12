from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from haze_forecast.forecast_audit import audit_forecast_file, write_audit_report


def _fully_described_record(**overrides: object) -> dict[str, object]:
    prediction_time = datetime(2021, 3, 23, 18, tzinfo=UTC)
    record: dict[str, object] = {
        "prediction_time": prediction_time,
        "run_time": datetime(2021, 3, 23, 12, tzinfo=UTC),
        "published_time": datetime(2021, 3, 23, 17, tzinfo=UTC),
        "lead_hours": 6,
        "latitude": 30.5723,
        "longitude": 104.0665,
        "variables": ("temperature_2m", "u_wind_10m"),
        "download_url": "https://example.test/gfs.t12z.f006.grib2",
        "sha256": "a" * 64,
        "event_hours": (prediction_time,),
    }
    record.update(overrides)
    return record


def test_audit_forecast_file_rejects_a_file_published_after_prediction_time() -> None:
    issue_time = datetime(2021, 3, 23, 18, tzinfo=UTC)

    result = audit_forecast_file(
        **_fully_described_record(published_time=issue_time + timedelta(hours=1))
    )

    assert result["eligible"] is False
    assert "published" in result["reason"]


def test_audit_forecast_file_accepts_complete_no_leak_hour_aligned_record() -> None:
    result = audit_forecast_file(**_fully_described_record())

    assert result["eligible"] is True
    assert result["valid_time"] == "2021-03-23T18:00:00+00:00"
    assert result["reason"] is None
    assert result["variables"] == ["temperature_2m", "u_wind_10m"]


def test_audit_forecast_file_rejects_missing_lead_and_unaligned_event_hour() -> None:
    missing_lead = audit_forecast_file(**_fully_described_record(lead_hours=None))
    unaligned = audit_forecast_file(
        **_fully_described_record(
            run_time=datetime(2021, 3, 23, 12, 30, tzinfo=UTC),
            published_time=datetime(2021, 3, 23, 17, tzinfo=UTC),
        )
    )

    assert missing_lead["eligible"] is False
    assert "lead" in missing_lead["reason"]
    assert unaligned["eligible"] is False
    assert "event hour" in unaligned["reason"]


def test_audit_forecast_file_requires_the_exact_raw_file_hash() -> None:
    result = audit_forecast_file(**_fully_described_record(sha256=None))

    assert result["eligible"] is False
    assert "sha256" in result["reason"]


def test_audit_forecast_file_rejects_a_source_whose_local_probe_failed() -> None:
    result = audit_forecast_file(**_fully_described_record(source_available=False))

    assert result["eligible"] is False
    assert "source probe" in result["reason"]


def test_write_audit_report_publishes_json_only(tmp_path: Path) -> None:
    result = audit_forecast_file(**_fully_described_record(sha256=None))
    output = tmp_path / "historical_forecast_audit" / "report.json"

    write_audit_report(output, [result])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["eligible"] is False
    assert report["eligible_record_count"] == 0
    assert report["records"][0]["sha256"] is None
    assert list(output.parent.iterdir()) == [output]


def test_historical_forecast_audit_cli_publishes_only_ineligible_json_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "historical_forecast_audit"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_historical_forecasts.py",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--output",
            str(output_dir),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"PYTHONPATH": str(Path.cwd() / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    report_path = output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["eligible"] is False
    assert report["record_count"] == 2
    assert sorted(path.name for path in output_dir.iterdir()) == ["report.json"]
