"""Extract a bounded, time-safe GEFSv12 surface-feature pilot for whole haze events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.gefs_dataset import (
    GEFS_SURFACE_FEATURE_COLUMNS,
    enrich_event_rows_with_gefs_surface,
)

_RESEARCH_YEAR = 2015
_MAX_EVENTS = 15
_MAX_SAMPLES_PER_EVENT = 4


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须为正整数") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def _research_year(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("年份必须为 2015") from error
    if parsed != _RESEARCH_YEAR:
        raise argparse.ArgumentTypeError("研究范围固定为 2015 年")
    return parsed


def _bounded_positive_int(value: str, *, maximum: int, label: str) -> int:
    parsed = _positive_int(value)
    if parsed > maximum:
        raise argparse.ArgumentTypeError(f"{label} 不能超过 {maximum}")
    return parsed


def _max_events(value: str) -> int:
    return _bounded_positive_int(value, maximum=_MAX_EVENTS, label="事件数量")


def _samples_per_event(value: str) -> int:
    return _bounded_positive_int(value, maximum=_MAX_SAMPLES_PER_EVENT, label="每事件样本数")


def _validate_research_scope(*, year: int, max_events: int, samples_per_event: int) -> None:
    if (
        isinstance(year, bool)
        or isinstance(max_events, bool)
        or isinstance(samples_per_event, bool)
        or year != _RESEARCH_YEAR
        or not 1 <= max_events <= _MAX_EVENTS
        or not 1 <= samples_per_event <= _MAX_SAMPLES_PER_EVENT
    ):
        raise ValueError(
            "research scope requires year=2015, 1 <= max_events <= 15, "
            "and 1 <= samples_per_event <= 4"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/city_labeled.parquet"))
    parser.add_argument("--year", type=_research_year, default=_RESEARCH_YEAR)
    parser.add_argument("--max-events", type=_max_events, default=_MAX_EVENTS)
    parser.add_argument("--samples-per-event", type=_samples_per_event, default=_MAX_SAMPLES_PER_EVENT)
    parser.add_argument("--longitude", type=float, default=104.0665)
    parser.add_argument("--latitude", type=float, default=30.5723)
    parser.add_argument("--horizon-hours", type=_positive_int, default=6)
    parser.add_argument("--timeout-s", type=_positive_int, default=60)
    parser.add_argument("--wgrib2", type=Path, default=Path("tools/wgrib2/v3.1.3/wgrib2.exe"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/gefs_multivariable_event_pilot.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/gefs_multivariable_event_pilot/report.json")
    )
    return parser.parse_args()


def _select_event_samples(
    city_labeled: pd.DataFrame, *, year: int, max_events: int, samples_per_event: int
) -> tuple[pd.DataFrame, list[int], dict[str, int]]:
    _validate_research_scope(
        year=year, max_events=max_events, samples_per_event=samples_per_event
    )
    required = {"timestamp", "event_id", "remaining_hours", "aqi", "particle_primary"}
    missing = required - set(city_labeled.columns)
    if missing:
        raise ValueError(f"city_labeled 缺少事件抽取列: {sorted(missing)}")
    frame = city_labeled.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    eligible = frame.loc[
        frame["event_id"].gt(0)
        & frame["remaining_hours"].notna()
        & frame["aqi"].gt(100)
        & frame["particle_primary"].fillna(False)
        & frame["timestamp"].dt.year.eq(year)
    ].sort_values(["event_id", "timestamp"])
    if eligible.empty:
        raise ValueError(f"{year} 年没有可抽取的已标注污染事件")
    first_times = eligible.groupby("event_id", as_index=False)["timestamp"].min().sort_values("timestamp")
    selected_groups: list[pd.DataFrame] = []
    selected_ids: list[int] = []
    counts: dict[str, int] = {}
    for event_id in first_times["event_id"]:
        event = eligible.loc[eligible["event_id"].eq(event_id)].sort_values("timestamp")
        if len(event) < samples_per_event:
            continue
        indexes = np.linspace(0, len(event) - 1, samples_per_event).round().astype(int)
        sampled = event.iloc[indexes].copy()
        selected_groups.append(sampled)
        event_int = int(event_id)
        selected_ids.append(event_int)
        counts[str(event_int)] = len(sampled)
        if len(selected_ids) == max_events:
            break
    if len(selected_ids) != max_events:
        raise ValueError(
            f"GEFS extraction requires exactly {max_events} eligible independent events "
            f"with at least {samples_per_event} predictable rows each"
        )
    return pd.concat(selected_groups, ignore_index=True), selected_ids, counts


def extract_multivariable_event_features(
    city_labeled: pd.DataFrame,
    *,
    year: int,
    max_events: int,
    samples_per_event: int,
    horizon_hours: int,
    longitude: float,
    latitude: float,
    wgrib2: Path,
    timeout_s: int,
    output: Path,
    manifest: Path,
) -> dict[str, object]:
    """Download the bounded pilot and write only its parquet plus provenance manifest."""
    _validate_research_scope(
        year=year, max_events=max_events, samples_per_event=samples_per_event
    )
    selected, event_ids, sample_counts = _select_event_samples(
        city_labeled,
        year=year,
        max_events=max_events,
        samples_per_event=samples_per_event,
    )
    enriched = enrich_event_rows_with_gefs_surface(
        selected,
        longitude=longitude,
        latitude=latitude,
        wgrib2_executable=wgrib2,
        horizon_hours=horizon_hours,
        timeout_s=timeout_s,
    )
    run_time = pd.to_datetime(enriched["gefs_run_utc"], utc=True, errors="raise")
    timestamp = pd.to_datetime(enriched["timestamp"], utc=True, errors="raise")
    valid_time = pd.to_datetime(enriched["gefs_forecast_valid_utc"], utc=True, errors="raise")
    safe = (
        run_time.le(timestamp)
        & timestamp.lt(valid_time)
        & valid_time.ge(timestamp + pd.Timedelta(6, unit="h"))
    )
    if not safe.all():
        raise RuntimeError(
            "GEFS extraction requires run <= timestamp < valid and a forecast valid at least six hours ahead"
        )
    feature_columns = [name for name in GEFS_SURFACE_FEATURE_COLUMNS if name in enriched.columns]
    payload = {
        "source": "NOAA GEFSv12 retrospective public S3 archive",
        "retrospective_only": True,
        "year": year,
        "horizon_hours": horizon_hours,
        "event_ids": event_ids,
        "event_sample_counts": sample_counts,
        "event_rows": len(enriched),
        "time_safety": {
            "run_lte_timestamp_lt_valid": True,
            "forecast_at_least_horizon_hours_ahead": True,
        },
        "feature_columns": feature_columns,
        "source_field_names": [
            "gefs_run_utc",
            "gefs_lead_hours",
            "gefs_forecast_valid_utc",
            "gefs_grid_longitude",
            "gefs_grid_latitude",
            "gefs_source_manifest_json",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"frame": enriched, "event_ids": event_ids, "manifest": payload}


def main() -> None:
    args = parse_args()
    city_labeled = pd.read_parquet(args.input)
    result = extract_multivariable_event_features(
        city_labeled,
        year=args.year,
        max_events=args.max_events,
        samples_per_event=args.samples_per_event,
        horizon_hours=args.horizon_hours,
        longitude=args.longitude,
        latitude=args.latitude,
        wgrib2=args.wgrib2,
        timeout_s=args.timeout_s,
        output=args.output,
        manifest=args.manifest,
    )
    print(json.dumps(result["manifest"], ensure_ascii=False))


if __name__ == "__main__":
    main()
