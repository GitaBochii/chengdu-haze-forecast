"""Audit local city samples before they are considered for Chengdu-only validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from haze_forecast.constants import REQUIRED_COLUMNS
from haze_forecast.labels import label_city_data, label_summary

CITY_REQUIRED_COLUMNS = tuple(sorted(REQUIRED_COLUMNS - {"station_id"}))
DEFAULT_MINIMUM_COVERAGE = 0.70


def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" not in frame:
        raise ValueError("missing timestamp column")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("invalid timestamps")
    return timestamps


def _compatible_boundary(timestamps: pd.Series, boundary: pd.Timestamp) -> pd.Timestamp:
    timestamp_timezone = timestamps.dt.tz
    if timestamp_timezone is None and boundary.tzinfo is None:
        return boundary
    if timestamp_timezone is None or boundary.tzinfo is None:
        raise ValueError("time boundary timezone must match city timestamps")
    return boundary


def validate_city_partition(
    city_frame: pd.DataFrame, chengdu_validation_start: pd.Timestamp | str
) -> None:
    """Require all candidate-city observations to precede Chengdu's held-out period."""
    timestamps = _timestamp_series(city_frame)
    if timestamps.duplicated().any():
        raise ValueError("duplicate city timestamps")
    boundary = _compatible_boundary(timestamps, pd.Timestamp(chengdu_validation_start))
    if timestamps.ge(boundary).any():
        raise ValueError("time boundary overlap with Chengdu validation")


def _time_coverage(timestamps: pd.Series) -> dict[str, Any]:
    unique = timestamps.drop_duplicates().sort_values()
    if unique.empty:
        return {
            "start": None,
            "end": None,
            "observed_rows": 0,
            "distinct_timestamps": 0,
            "expected_hourly_timestamps": 0,
            "rate": 0.0,
        }
    expected = len(pd.date_range(unique.iloc[0], unique.iloc[-1], freq="h"))
    return {
        "start": unique.iloc[0].isoformat(),
        "end": unique.iloc[-1].isoformat(),
        "observed_rows": len(timestamps),
        "distinct_timestamps": len(unique),
        "expected_hourly_timestamps": expected,
        "rate": float(len(unique) / expected),
    }


def _variable_non_null_rates(frame: pd.DataFrame) -> dict[str, float | None]:
    rates: dict[str, float | None] = {}
    for column in CITY_REQUIRED_COLUMNS:
        if column == "timestamp":
            continue
        rates[column] = float(frame[column].notna().mean()) if column in frame else None
    return rates


def audit_city_frame(
    city: str,
    city_frame: pd.DataFrame,
    chengdu_validation_start: pd.Timestamp | str,
    *,
    minimum_coverage: float = DEFAULT_MINIMUM_COVERAGE,
    minimum_complete_events: int = 1,
) -> dict[str, Any]:
    """Audit schema, hourly coverage, labels, and the Chengdu validation boundary."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum coverage must be in (0, 1]")
    if minimum_complete_events < 1:
        raise ValueError("minimum complete events must be at least one")

    report: dict[str, Any] = {
        "city": city,
        "eligible": False,
        "minimum_coverage": minimum_coverage,
        "minimum_complete_events": minimum_complete_events,
        "failure_reasons": [],
        "time_coverage": _time_coverage(pd.Series(dtype="datetime64[ns]")),
        "variable_non_null_rate": _variable_non_null_rates(city_frame),
        "completed_events": 0,
        "labeled_hours": 0,
        "particle_primary_hours": 0,
        "labeling": {
            "performed": False,
            "method": "haze_forecast.labels.label_city_data",
            "require_aqi_recovery": True,
        },
        "time_boundary": {
            "chengdu_validation_start": pd.Timestamp(chengdu_validation_start).isoformat(),
            "eligible_for_chengdu_train": False,
        },
    }
    failure_reasons: list[str] = report["failure_reasons"]
    missing = sorted(set(CITY_REQUIRED_COLUMNS) - set(city_frame.columns))
    if missing:
        failure_reasons.append(f"missing required columns: {missing}")

    try:
        timestamps = _timestamp_series(city_frame)
        report["time_coverage"] = _time_coverage(timestamps)
        boundary = _compatible_boundary(timestamps, pd.Timestamp(chengdu_validation_start))
        if timestamps.duplicated().any():
            failure_reasons.append("duplicate city timestamps")
        elif timestamps.ge(boundary).any():
            failure_reasons.append("time boundary overlap with Chengdu validation")
        else:
            report["time_boundary"]["eligible_for_chengdu_train"] = True
    except ValueError as error:
        failure_reasons.append(str(error))
        timestamps = pd.Series(dtype="datetime64[ns]")

    coverage = report["time_coverage"]["rate"]
    if coverage < minimum_coverage:
        failure_reasons.append(
            f"hourly coverage {coverage:.3f} is below minimum {minimum_coverage:.2f}"
        )
    low_coverage_columns = sorted(
        column
        for column, rate in report["variable_non_null_rate"].items()
        if rate is not None and rate < minimum_coverage
    )
    if low_coverage_columns:
        failure_reasons.append(
            "variable non-null coverage below minimum: " + ", ".join(low_coverage_columns)
        )

    can_label = not missing and not timestamps.empty and not any(
        "timestamp" in reason or "time boundary" in reason for reason in failure_reasons
    )
    if can_label:
        try:
            labeled = label_city_data(city_frame)
            summary = label_summary(labeled)
            report.update(summary)
            report["labeling"]["performed"] = True
        except (TypeError, ValueError) as error:
            failure_reasons.append(f"independent label generation failed: {error}")

    if report["completed_events"] < minimum_complete_events:
        failure_reasons.append(
            f"completed events {report['completed_events']} is below minimum {minimum_complete_events}"
        )
    report["eligible"] = not failure_reasons
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _city_data_file(directory: Path) -> Path | None:
    preferred = [directory / "city_hourly.parquet", directory / "city_hourly.csv"]
    for path in preferred:
        if path.is_file():
            return path
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".parquet", ".csv"}
    )
    return files[0] if len(files) == 1 else None


def _read_city_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def audit_local_multicity_samples(
    city_data_dir: Path,
    chengdu_validation_start: pd.Timestamp | str,
    *,
    primary_city: str = "chengdu",
    minimum_coverage: float = DEFAULT_MINIMUM_COVERAGE,
    minimum_complete_events: int = 1,
) -> dict[str, Any]:
    """Discover local candidate-city files and return evidence without producing training data."""
    city_data_dir = Path(city_data_dir)
    report: dict[str, Any] = {
        "audit": "multicity sample feasibility",
        "eligible": False,
        "city_data_dir": str(city_data_dir),
        "chengdu_validation_start": pd.Timestamp(chengdu_validation_start).isoformat(),
        "primary_city": primary_city,
        "minimum_coverage": minimum_coverage,
        "minimum_complete_events": minimum_complete_events,
        "cities": [],
        "failure_reasons": [],
    }
    if not city_data_dir.is_dir():
        report["failure_reasons"].append("local city data directory does not exist")
        report["eligible_city_count"] = 0
        return report

    directories = sorted(path for path in city_data_dir.iterdir() if path.is_dir())
    if not directories:
        report["failure_reasons"].append("no local city directories found")
    for directory in directories:
        city = directory.name
        if city.casefold() == primary_city.casefold():
            report["cities"].append(
                {
                    "city": city,
                    "eligible": False,
                    "source_directory": str(directory),
                    "failure_reasons": ["primary city is excluded from multicity candidates"],
                }
            )
            continue
        source = _city_data_file(directory)
        if source is None:
            report["cities"].append(
                {
                    "city": city,
                    "eligible": False,
                    "source_directory": str(directory),
                    "failure_reasons": ["no supported city-hourly parquet/csv source file"],
                }
            )
            continue
        try:
            city_report = audit_city_frame(
                city,
                _read_city_data(source),
                chengdu_validation_start,
                minimum_coverage=minimum_coverage,
                minimum_complete_events=minimum_complete_events,
            )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            city_report = {
                "city": city,
                "eligible": False,
                "failure_reasons": [f"could not read source: {error}"],
            }
        city_report["source_directory"] = str(directory)
        city_report["source_file"] = str(source)
        city_report["source_sha256"] = _sha256(source)
        report["cities"].append(city_report)

    eligible_city_count = sum(bool(city.get("eligible")) for city in report["cities"])
    report["eligible_city_count"] = eligible_city_count
    report["eligible"] = eligible_city_count > 0
    if not report["eligible"]:
        report["failure_reasons"].append("no locally auditable city meets the feasibility criteria")
    return report


def chengdu_validation_start_from_metadata(path: Path) -> pd.Timestamp:
    """Read the frozen Chengdu final-test start rather than reconstructing a split."""
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return pd.Timestamp(metadata["time_ranges"]["test"][0])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("Chengdu metadata has no final validation start") from error
