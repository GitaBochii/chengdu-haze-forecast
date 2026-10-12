import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from haze_forecast.multicity_audit import audit_city_frame, validate_city_partition


def _city_frame(start: str, event_count: int = 1) -> pd.DataFrame:
    rows_per_event = 80
    timestamps = pd.date_range(start, periods=rows_per_event * event_count, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "aqi": 80.0,
            "pm2_5": 20.0,
            "pm10": 30.0,
            "so2": 8.0,
            "no2": 20.0,
            "co": 0.7,
            "o3": 60.0,
            "temperature_c": 15.0,
            "relative_humidity_pct": 65.0,
            "wind_u_ms": 1.0,
            "wind_v_ms": -1.0,
            "boundary_layer_height_m": 250.0,
        }
    )
    for offset in range(event_count):
        event_start = offset * rows_per_event + 24
        frame.loc[event_start : event_start + 23, ["aqi", "pm2_5", "pm10"]] = [
            150.0,
            150.0,
            180.0,
        ]
    return frame


def test_city_audit_rejects_labels_that_overlap_train_and_chengdu_validation_time():
    city_frame = _city_frame("2020-01-01")

    with pytest.raises(ValueError, match="time boundary"):
        validate_city_partition(city_frame, pd.Timestamp("2020-01-02"))


def test_city_audit_reports_independent_labels_coverage_and_event_count():
    report = audit_city_frame(
        "mianyang",
        _city_frame("2019-01-01", event_count=2),
        pd.Timestamp("2020-01-01"),
    )

    assert report["eligible"] is True
    assert report["completed_events"] == 2
    assert report["labeled_hours"] > 0
    assert report["time_coverage"]["rate"] == pytest.approx(1.0)
    assert report["variable_non_null_rate"]["pm2_5"] == pytest.approx(1.0)
    assert report["labeling"]["method"] == "haze_forecast.labels.label_city_data"
    assert report["time_boundary"]["eligible_for_chengdu_train"] is True


def test_city_audit_reports_missing_required_variables_without_labeling():
    city_frame = _city_frame("2019-01-01").drop(columns="wind_v_ms")

    report = audit_city_frame("mianyang", city_frame, pd.Timestamp("2020-01-01"))

    assert report["eligible"] is False
    assert report["completed_events"] == 0
    assert "missing required columns: ['wind_v_ms']" in report["failure_reasons"]
    assert report["labeling"]["performed"] is False


def test_multicity_audit_cli_writes_rejection_evidence_when_only_chengdu_is_local(tmp_path: Path):
    city_root = tmp_path / "cities"
    (city_root / "chengdu").mkdir(parents=True)
    output = tmp_path / "audit" / "report.json"
    metadata = tmp_path / "model_metadata.json"
    metadata.write_text(
        json.dumps({"time_ranges": {"test": ["2019-12-23T04:00:00+08:00"]}}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_multicity_samples.py",
            "--city-data-dir",
            str(city_root),
            "--chengdu-metadata",
            str(metadata),
            "--output",
            str(output),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"PYTHONPATH": str(Path.cwd() / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["eligible_city_count"] == 0
    assert report["eligible"] is False
    assert report["cities"][0]["city"] == "chengdu"
    assert "primary city" in report["cities"][0]["failure_reasons"][0]
