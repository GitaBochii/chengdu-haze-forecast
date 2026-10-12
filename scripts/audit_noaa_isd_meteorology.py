"""Audit nearby NOAA ISD stations before generating city meteorology features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from haze_forecast.noaa_isd import audit_noaa_isd

_MIN_NOAA_COVERAGE = 0.70
_REQUIRED_AUDIT_YEARS = list(range(2015, 2022))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _station_list(path: Path) -> pd.DataFrame:
    stations = pd.read_csv(path, dtype="string")
    aliases = {
        "USAF": "usaf",
        "WBAN": "wban",
        "LAT": "latitude",
        "LON": "longitude",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
    }
    stations = stations.rename(columns={key: aliases.get(key.upper(), key) for key in stations.columns})
    required = {"usaf", "wban", "latitude", "longitude"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"station list is missing columns: {sorted(missing)}")
    return stations


def _minimum_coverage(value: str) -> float:
    try:
        coverage = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be at least 0.70") from error
    if not math.isfinite(coverage) or coverage < _MIN_NOAA_COVERAGE:
        raise argparse.ArgumentTypeError("must be at least 0.70")
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NOAA ISD hourly meteorology for Chengdu.")
    parser.add_argument("--years", nargs="+", type=int, default=_REQUIRED_AUDIT_YEARS)
    parser.add_argument("--station-list", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/noaa_isd"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/noaa_isd_audit/report.json"))
    parser.add_argument("--maximum-distance-km", type=float, default=60.0)
    parser.add_argument("--minimum-coverage", type=_minimum_coverage, default=_MIN_NOAA_COVERAGE)
    args = parser.parse_args()
    if args.years != _REQUIRED_AUDIT_YEARS:
        parser.error("--years must be exactly: 2015 2016 2017 2018 2019 2020 2021")
    return args


def main() -> None:
    args = parse_args()
    city_hourly, report = audit_noaa_isd(
        _station_list(args.station_list),
        args.raw_dir,
        years=args.years,
        maximum_distance_km=args.maximum_distance_km,
        minimum_coverage=args.minimum_coverage,
    )
    report["eligible"] = bool(
        report.get("eligible_station_count", 0) > 0
        and args.minimum_coverage >= _MIN_NOAA_COVERAGE
        and args.years == _REQUIRED_AUDIT_YEARS
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hourly_output = args.output.parent / "noaa_isd_hourly.parquet"
    if city_hourly is not None:
        city_hourly.to_parquet(hourly_output, index=False)
        report["city_hourly_sha256"] = _file_sha256(hourly_output)
    else:
        hourly_output.unlink(missing_ok=True)
        report["city_hourly_sha256"] = None
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"eligible_station_count": report["eligible_station_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
