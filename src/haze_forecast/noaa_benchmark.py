"""Audited NOAA ISD feature comparison using the existing AFT event protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from haze_forecast.features import model_feature_names, training_rows
from haze_forecast.spatial_benchmark import benchmark_spatial_features
from haze_forecast.storage import write_json_atomic

_LOCAL_TIMEZONE = "Asia/Shanghai"
_MIN_NOAA_COVERAGE = 0.70
_REQUIRED_AUDIT_YEARS = list(range(2015, 2022))
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def audit_is_eligible(report: Mapping[str, Any]) -> bool:
    """Return whether an audit report certifies that a city NOAA table is usable."""
    if not isinstance(report, Mapping):
        raise TypeError("NOAA audit report must be an object with an eligible status")
    if report.get("eligible") is not True:
        return False
    try:
        eligible_station_count = int(report.get("eligible_station_count", 0))
    except (TypeError, ValueError):
        return False
    eligible = eligible_station_count > 0
    try:
        minimum_coverage = float(report["minimum_coverage"])
    except (KeyError, TypeError, ValueError):
        return False
    if not eligible or not math.isfinite(minimum_coverage) or minimum_coverage < _MIN_NOAA_COVERAGE:
        return False
    years = report.get("years")
    stations = report.get("stations")
    if years != _REQUIRED_AUDIT_YEARS or not isinstance(stations, list):
        return False
    eligible_stations = [station for station in stations if isinstance(station, Mapping) and station.get("eligible")]
    if not eligible_stations:
        return False
    for station in eligible_stations:
        annual_reports = station.get("years")
        if not isinstance(annual_reports, Mapping):
            return False
        for year in years:
            annual_report = annual_reports.get(str(year))
            try:
                coverage = float(annual_report["coverage"])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(coverage) or coverage < _MIN_NOAA_COVERAGE:
                return False
    return True


def audit_report_sha256(path: Path) -> str:
    """Hash the exact audit report used to authorize a benchmark run."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return audit_report_sha256(path)


def _validate_benchmark_controls(audit_sha256: object, n_splits: object) -> str:
    if n_splits != 3 or isinstance(n_splits, bool):
        raise ValueError("NOAA benchmark requires exactly 3 complete-event folds")
    if not isinstance(audit_sha256, str) or _SHA256_PATTERN.fullmatch(audit_sha256) is None:
        raise ValueError("NOAA benchmark requires a valid 64-character audit report sha256")
    return audit_sha256.lower()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _validate_audit_provenance(
    audit_report: Mapping[str, Any], audit_report_path: Path | None, audit_sha256: str
) -> None:
    if audit_report_path is None:
        raise ValueError("NOAA benchmark requires an audit report path; direct API provenance bypass is blocked")
    try:
        file_bytes = audit_report_path.read_bytes()
        file_report = json.loads(file_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"NOAA benchmark cannot verify audit report path: {error}") from error
    actual_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if actual_sha256 != audit_sha256:
        raise ValueError("NOAA benchmark audit report sha256 does not match the audit file")
    if not isinstance(file_report, dict) or file_report != dict(audit_report):
        raise ValueError("NOAA benchmark audit report content does not match the verified audit file")

    years = audit_report.get("years")
    stations = audit_report.get("stations")
    if not isinstance(years, list) or not years or not isinstance(stations, list):
        raise ValueError("NOAA benchmark audit report has no verifiable eligible station sources")
    eligible_stations = [station for station in stations if isinstance(station, Mapping) and station.get("eligible")]
    if not eligible_stations:
        raise ValueError("NOAA benchmark audit report has no verifiable eligible station sources")
    for station in eligible_stations:
        annual_sources = station.get("years")
        boundary_sources = station.get("boundary_sources")
        if not isinstance(annual_sources, Mapping) or not isinstance(boundary_sources, Mapping):
            raise ValueError(  # noqa: TRY004
                "NOAA benchmark audit report is missing eligible station source hashes"
            )
        for year in years:
            key = str(year)
            annual_source = annual_sources.get(key)
            boundary_source = boundary_sources.get(key)
            if not isinstance(annual_source, Mapping) or not _is_sha256(annual_source.get("sha256")):
                raise ValueError("NOAA benchmark audit report has invalid eligible station source sha256")
            if not isinstance(boundary_source, Mapping) or not _is_sha256(boundary_source.get("sha256")):
                raise ValueError("NOAA benchmark audit report has invalid eligible station source sha256")


def _validate_hourly_provenance(audit_report: Mapping[str, Any], hourly_path: Path | None) -> None:
    if hourly_path is None:
        raise ValueError("NOAA benchmark requires an hourly parquet path; direct API bypass is blocked")
    expected_sha256 = audit_report.get("city_hourly_sha256")
    if not _is_sha256(expected_sha256):
        raise ValueError("NOAA benchmark audit report has no valid hourly sha256")
    try:
        actual_sha256 = _file_sha256(hourly_path)
    except OSError as error:
        raise ValueError(f"NOAA benchmark cannot verify hourly parquet path: {error}") from error
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError("NOAA benchmark hourly sha256 does not match the audited hourly parquet")


def _select_noaa_features(
    features: pd.DataFrame, noaa_columns: list[str]
) -> tuple[dict[str, float], list[str], dict[str, dict[str, float | str]], float]:
    """Keep NOAA columns with sufficient individual coverage for this training set."""
    non_null_rate = {name: float(features[name].notna().mean()) for name in noaa_columns}
    selected = [name for name in noaa_columns if non_null_rate[name] >= _MIN_NOAA_COVERAGE]
    excluded = {
        name: {
            "reason": "non_null_rate_below_0.70",
            "non_null_rate": non_null_rate[name],
        }
        for name in noaa_columns
        if name not in selected
    }
    if not selected:
        raise ValueError("NOAA feature coverage is below 0.70 before benchmark: no eligible variables")

    complete_rate = float(features[selected].notna().all(axis=1).mean())
    if complete_rate < _MIN_NOAA_COVERAGE:
        raise ValueError(
            "NOAA feature coverage is below 0.70 before benchmark: "
            f"selected_variables={selected}, complete_rate={complete_rate:.3f}"
        )
    return non_null_rate, selected, excluded, complete_rate


def _normalise_hourly_timestamps(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or "timestamp" not in frame.columns:
        raise ValueError(f"{label} must be a DataFrame with timestamp")
    normalised = frame.copy()
    try:
        timestamp = pd.to_datetime(normalised["timestamp"], errors="raise")
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} timestamps must be valid") from error
    if timestamp.dt.tz is None:
        timestamp = timestamp.dt.tz_localize(_LOCAL_TIMEZONE)
    else:
        timestamp = timestamp.dt.tz_convert(_LOCAL_TIMEZONE)
    normalised["timestamp"] = timestamp.dt.floor("h")
    if normalised["timestamp"].isna().any() or normalised["timestamp"].duplicated().any():
        raise ValueError(f"{label} must have one row for each valid local hour")
    return normalised.sort_values("timestamp").reset_index(drop=True)


def merge_audited_noaa_hourly(
    city_labeled: pd.DataFrame,
    noaa_hourly: pd.DataFrame | None,
    audit_report: Mapping[str, Any],
) -> pd.DataFrame:
    """Left join one audited NOAA observation row to each labelled local hour."""
    eligible = audit_is_eligible(audit_report)
    if not eligible:
        raise ValueError("NOAA audit is not eligible (eligible=false); benchmark training is blocked")
    if noaa_hourly is None or noaa_hourly.empty:
        raise ValueError(
            "NOAA audit is not eligible for training (eligible=true; no audited NOAA hourly table)"
        )
    city = _normalise_hourly_timestamps(city_labeled, "city labeled data")
    hourly = _normalise_hourly_timestamps(noaa_hourly, "NOAA hourly data")
    noaa_columns = [name for name in hourly.columns if name.startswith("noaa_")]
    if not noaa_columns:
        raise ValueError(
            "NOAA audit is not eligible for training (eligible=true; no audited NOAA variables)"
        )
    collisions = set(city.columns).intersection(hourly.columns) - {"timestamp"}
    if collisions:
        raise ValueError(f"NOAA hourly columns collide with city data: {sorted(collisions)}")
    return city.merge(hourly, on="timestamp", how="left", validate="one_to_one")


def benchmark_noaa_isd_features(
    city_labeled: pd.DataFrame,
    noaa_hourly: pd.DataFrame | None,
    audit_report: Mapping[str, Any],
    *,
    audit_report_sha256: str | None = None,
    audit_report_path: Path | None = None,
    hourly_path: Path | None = None,
    n_splits: int = 3,
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare existing inputs against the same inputs plus audited NOAA ISD values.

    ``noaa_hourly`` is retained only for legacy callers and is never used.  The
    hash-verified ``hourly_path`` is the sole source permitted for training.
    """
    del noaa_hourly
    if not audit_is_eligible(audit_report):
        raise ValueError("NOAA audit is not eligible (eligible=false); benchmark training is blocked")
    validated_audit_sha256 = _validate_benchmark_controls(audit_report_sha256, n_splits)
    _validate_audit_provenance(audit_report, audit_report_path, validated_audit_sha256)
    _validate_hourly_provenance(audit_report, hourly_path)
    try:
        audited_hourly = pd.read_parquet(hourly_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"NOAA benchmark cannot read verified hourly parquet: {error}") from error
    joined = merge_audited_noaa_hourly(city_labeled, audited_hourly, audit_report)
    features = training_rows(joined)
    noaa_columns = [name for name in features.columns if name.startswith("noaa_")]
    noaa_non_null_rate, selected_noaa_columns, excluded_noaa_features, noaa_complete_rate = (
        _select_noaa_features(features, noaa_columns)
    )
    enhanced = [
        name
        for name in model_feature_names(features)
        if not name.startswith("noaa_") or name in selected_noaa_columns
    ]
    baseline = [name for name in enhanced if not name.startswith("noaa_")]
    noaa_features = [name for name in enhanced if name.startswith("noaa_")]
    if not noaa_features:
        raise ValueError("NOAA audit is not eligible for training (eligible=true; no NOAA model features)")

    spatial_report, predictions = benchmark_spatial_features(
        features,
        baseline_features=baseline,
        enhanced_features=enhanced,
        n_splits=n_splits,
        seed=seed,
    )
    models = dict(spatial_report["models"])
    noaa_model = models.pop("directional")
    report: dict[str, object] = {
        "protocol": spatial_report["protocol"],
        "training_rows": len(features),
        "input_audit_report_sha256": validated_audit_sha256,
        "noaa_features": noaa_features,
        "noaa_feature_non_null_rate": noaa_non_null_rate,
        "noaa_feature_complete_rate": noaa_complete_rate,
        "excluded_noaa_features": excluded_noaa_features,
        "folds": spatial_report["folds"],
        "variants": ["baseline", "noaa_isd"],
        "models": {"baseline": models["baseline"], "noaa_isd": noaa_model},
        "decision": spatial_report["decision"],
    }
    renamed_predictions = predictions.copy()
    renamed_predictions["variant"] = renamed_predictions["variant"].replace(
        {"directional": "noaa_isd"}
    )
    return report, renamed_predictions


def write_noaa_benchmark_artifacts(
    report: dict[str, object], predictions: pd.DataFrame, output_dir: Path
) -> None:
    """Write the paired report and fold predictions for NOAA feature review."""
    required = {"fold", "variant", "event_id", "actual_remaining_hours", "predicted_remaining_hours"}
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("NOAA benchmark predictions must be a pandas DataFrame")
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"NOAA benchmark predictions missing columns: {sorted(missing)}")
    variants = set(predictions["variant"])
    if not variants.issubset({"baseline", "noaa_isd"}):
        raise ValueError("NOAA benchmark predictions have invalid variants")
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_dir = output_dir / "generations"
    generations_dir.mkdir(exist_ok=True)
    generation_id = uuid.uuid4().hex
    staging_dir = generations_dir / f".{generation_id}.part"
    generation_dir = generations_dir / generation_id
    current_path = output_dir / "current.json"
    current_staging_path = output_dir / "current.json.part"
    try:
        staging_dir.mkdir()
        write_json_atomic(staging_dir / "report.json", report)
        predictions.to_csv(staging_dir / "fold_predictions.csv", index=False, encoding="utf-8-sig")
        staging_dir.replace(generation_dir)
        write_json_atomic(
            current_staging_path,
            {
                "generation": generation_id,
                "report": f"generations/{generation_id}/report.json",
                "fold_predictions": f"generations/{generation_id}/fold_predictions.csv",
            },
        )
        current_staging_path.replace(current_path)
    finally:
        current_staging_path.unlink(missing_ok=True)
        if staging_dir.exists():
            for child in staging_dir.iterdir():
                child.unlink(missing_ok=True)
            staging_dir.rmdir()


def resolve_noaa_benchmark_artifacts(output_dir: Path) -> tuple[Path, Path]:
    """Return only the complete generation selected by the atomic current pointer."""
    current_path = output_dir / "current.json"
    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"NOAA benchmark has no valid current artifact pointer: {error}") from error
    if not isinstance(current, dict):
        raise ValueError("NOAA benchmark current artifact pointer must be an object")  # noqa: TRY004
    generation = current.get("generation")
    if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
        raise ValueError("NOAA benchmark current artifact pointer has an invalid generation")
    generation_dir = output_dir / "generations" / generation
    report_path = generation_dir / "report.json"
    prediction_path = generation_dir / "fold_predictions.csv"
    if not report_path.is_file() or not prediction_path.is_file():
        raise ValueError("NOAA benchmark current artifact generation is incomplete")
    return report_path, prediction_path
