"""Create a common Chengdu ERA5 reanalysis proxy, never as-of station weather."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests

from haze_forecast.open_meteo_archive import WEATHER_COLUMNS, parse_hourly_weather

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,"
    "wind_direction_10m,boundary_layer_height"
)


def parse_proxy_payload(payload: dict) -> pd.DataFrame:
    """Reject ambiguous units/timezones before using the existing weather parser."""
    if payload.get("timezone") != "Asia/Shanghai" or payload.get("utc_offset_seconds") != 28800:
        raise ValueError("weather timezone must be Asia/Shanghai with UTC+8 offset")
    if payload.get("hourly_units", {}).get("wind_speed_10m") != "m/s":
        raise ValueError("wind speed must be explicitly reported in m/s")
    payload = deepcopy(payload)
    hourly = payload["hourly"]
    if "boundary_layer_height" not in hourly:
        hourly["boundary_layer_height"] = [None] * len(hourly["time"])
    frame = parse_hourly_weather(payload)
    if frame.timestamp.duplicated().any():
        raise ValueError("duplicate weather timestamps")
    return frame


def align_proxy(frames: list[pd.DataFrame], grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Preserve all gaps rather than borrowing values from future times or years."""
    combined = pd.concat(frames, ignore_index=True)
    if combined.timestamp.duplicated().any():
        raise ValueError("duplicate weather timestamps across responses")
    return combined.set_index("timestamp").reindex(grid.rename("timestamp")).reset_index()


def year_ranges(start_date: str, end_date: str) -> list[tuple[int, str, str]]:
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start date must not follow end date")
    return [(year, max(start, date(year, 1, 1)).isoformat(),
             min(end, date(year, 12, 31)).isoformat())
            for year in range(start.year, end.year + 1)]


def validate_cache(path: Path, params: dict, entry: dict) -> bytes:
    if entry.get("parameters") != params or entry.get("status") != "validated":
        raise ValueError(f"cache identity is not validated: {path}")
    content = path.read_bytes()
    if entry.get("sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError(f"cache hash mismatch: {path}")
    return content


def cache_entry(path: Path) -> dict:
    sidecar = path.with_suffix(".identity.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    audit_path = path.parent / "audit.json"
    if audit_path.exists():
        entries = json.loads(audit_path.read_text(encoding="utf-8")).get("requests", [])
        for entry in entries:
            if Path(entry.get("response_path", "")).name == path.name:
                return entry
    raise ValueError(f"cache identity missing: {path}")


def validate_response(payload: dict, params: dict) -> None:
    frame = parse_proxy_payload(payload)
    units = payload.get("hourly_units", {})
    for column, expected in (("temperature_2m", "\u00b0C"),
                             ("relative_humidity_2m", "%"),
                             ("wind_direction_10m", "\u00b0")):
        if units.get(column) != expected:
            raise ValueError(f"unexpected unit for {column}: {units.get(column)}")
    grid = pd.date_range(params["start_date"], params["end_date"] + " 23:00",
                         freq="h", tz="Asia/Shanghai")
    if not pd.DatetimeIndex(frame.timestamp).equals(grid):
        raise ValueError("response timestamp grid does not match requested dates")


def fetch_response(params: dict, path: Path, audit: list, reuse_dir: Path | None = None) -> dict:
    url = requests.Request("GET", API_URL, params=params).prepare().url
    entry = {
        "url": url, "parameters": params,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "response_path": str(path.resolve()),
    }
    audit.append(entry)
    write_cache = False
    if path.exists():
        identity = cache_entry(path)
        content = validate_cache(path, params, identity)
        entry["checked_at_utc"] = entry["retrieved_at_utc"]
        entry["retrieved_at_utc"] = identity.get("retrieved_at_utc")
        if "reused_from" in identity:
            entry["reused_from"] = identity["reused_from"]
        entry["status"] = "cached"
    elif reuse_dir is not None and (reuse_dir / path.name).exists():
        source = reuse_dir / path.name
        identity = cache_entry(source)
        content = validate_cache(source, params, identity)
        entry["checked_at_utc"] = entry["retrieved_at_utc"]
        entry["retrieved_at_utc"] = identity.get("retrieved_at_utc")
        entry["reused_from"] = str(source.resolve())
        entry["status"] = "reused"
        write_cache = True
    else:
        try:
            response = requests.get(API_URL, params=params, timeout=90)
            entry["http_status"] = response.status_code
            content = response.content
            entry["status"] = "downloaded" if response.ok else "failed"
            entry["sha256"] = hashlib.sha256(content).hexdigest()
            response.raise_for_status()
            write_cache = True
        except requests.RequestException as error:
            entry["error"] = str(error)
            raise
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    payload = json.loads(content)
    validate_response(payload, params)
    entry["returned_grid_latitude"] = payload.get("latitude")
    entry["returned_grid_longitude"] = payload.get("longitude")
    entry["status"] = "validated"
    if write_cache:
        path.write_bytes(content)
        path.with_suffix(".identity.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/regional_air_models/weather"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/regional_air_models/weather_proxy.parquet"))
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-09-03")
    parser.add_argument("--metadata", type=Path,
                        default=Path("data/processed/regional_air_models/station_metadata.csv"))
    parser.add_argument("--reuse-cache-dir", type=Path)
    args = parser.parse_args()
    ranges = year_ranges(args.start_date, args.end_date)
    metadata = pd.read_csv(args.metadata)
    base_params = {
        "latitude": round(float(metadata.latitude.mean()), 7),
        "longitude": round(float(metadata.longitude.mean()), 7),
        "hourly": HOURLY, "models": "era5", "wind_speed_unit": "ms",
        "temperature_unit": "celsius", "timezone": "Asia/Shanghai",
    }
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "source_documentation": "https://open-meteo.com/en/docs/historical-weather-api",
        "model": "era5", "base_parameters": base_params,
        "interpretation": "Retrospective reanalysis; not weather available as of prediction issue time.",
        "spatial_scope": "One common Chengdu proxy at the mean of 9 local station coordinates, not 9 station measurements.",
        "future_weather_policy": "Downstream forecasting must not access proxy rows after its issue timestamp.",
        "source_delay": "ERA5 is documented as approximately five days delayed; latest requested hours may be null.",
        "imputation": "none", "requests": [],
        "start_date": args.start_date, "end_date": args.end_date,
        "metadata_path": str(args.metadata.resolve()),
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
    }
    try:
        params = {**base_params, "start_date": args.start_date, "end_date": args.start_date}
        probe = fetch_response(params, args.raw_dir / f"era5_probe_{args.start_date.replace('-', '')}.json",
                               audit["requests"], args.reuse_cache_dir)
        if len(parse_proxy_payload(probe)) != 24:
            raise ValueError("daily probe did not contain 24 local hours")
        print("Daily ERA5 probe validated: 24 hours, m/s, Asia/Shanghai", flush=True)
        frames = []
        for year, start, end in ranges:
            params = {**base_params, "start_date": start, "end_date": end}
            payload = fetch_response(params, args.raw_dir / f"era5_{year}.json", audit["requests"],
                                     args.reuse_cache_dir)
            frame = parse_proxy_payload(payload)
            frames.append(frame)
            print(f"ERA5 {year}: {len(frame)} hours", flush=True)
        result = align_proxy(frames, pd.date_range(
            args.start_date, args.end_date + " 23:00", freq="h", tz="Asia/Shanghai",
        ))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)
        audit["output"] = str(args.output.resolve())
        audit["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        audit["rows"] = len(result)
        audit["coverage_by_year"] = {
            str(year): {
                "hours": len(group),
                "available_hours": {column: int(group[column].notna().sum())
                                    for column in WEATHER_COLUMNS},
                "last_available_timestamp": {
                    column: str(group.loc[group[column].notna(), "timestamp"].max())
                    for column in WEATHER_COLUMNS
                },
            }
            for year, group in result.groupby(result.timestamp.dt.year)
        }
        audit["status"] = "complete"
        print(json.dumps(audit["coverage_by_year"], indent=2))
    except Exception as error:
        audit["status"] = "failed"
        audit["error"] = str(error)
        raise
    finally:
        (args.raw_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
