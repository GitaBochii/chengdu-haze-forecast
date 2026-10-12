"""Audit complete raw GEFS shards with immutable archive-timestamp evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from haze_forecast.operational_gefs_availability import (
    LIMITATIONS,
    POLICY,
    _raw_messages,
    audit_frame,
    head_evidence,
)
from haze_forecast.operational_gefs_quality import consolidate_verified_shards


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/operational_gefs_audited"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("artifacts/operational_gefs_ensemble/availability_cache"))
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/operational_gefs_ensemble/availability_report.json"))
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if args.report.exists():
        raise FileExistsError("report is immutable; select a new report path")
    result = {"policy": POLICY, "limitations": LIMITATIONS, "years": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for year in sorted(set(args.years)):
        paths = sorted(args.input_dir.glob(f"operational_gefs_event_features_{year}_*.parquet"))
        if not paths:
            raise ValueError(f"no raw shards for {year}")
        inputs = {str(path.resolve()): _hash(path) for path in paths}
        output = args.output_dir / f"operational_gefs_event_features_{year}.parquet"
        year_report = args.output_dir / f"availability_{year}.json"
        if output.exists() or year_report.exists():
            if not output.exists() or not year_report.exists():
                raise ValueError(f"incomplete output for {year}; use a new output directory")
            previous = json.loads(year_report.read_text(encoding="utf-8"))
            if previous["input_sha256"] != inputs or previous["output_sha256"] != _hash(output):
                raise ValueError(f"input/output identity changed for {year}; use a new output directory")
            result["years"][str(year)] = previous
            print(f"Resumed verified year {year}", flush=True)
            continue
        raw = consolidate_verified_shards([pd.read_parquet(path) for path in paths])
        if not raw.gefs_row_verified.all():
            raise ValueError(f"{year}: {int((~raw.gefs_row_verified).sum())} raw keys need extraction repair")
        urls = sorted({message["source_url"] for _, row in raw.iterrows()
                       for message in _raw_messages(row).values()})
        evidence = {}
        print(f"Auditing {year}: {len(raw)} rows, {len(urls)} unique objects", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(head_evidence, url, args.cache_dir): url for url in urls}
            for index, future in enumerate(as_completed(futures), start=1):
                evidence[futures[future]] = future.result()
                if index % 100 == 0 or index == len(futures):
                    print(f"HEAD evidence {year}: {index}/{len(futures)}", flush=True)
        audited, report = audit_frame(raw, evidence.__getitem__)
        if inputs != {str(path.resolve()): _hash(path) for path in paths}:
            raise ValueError("raw shards changed during audit; no output was written")
        report.update(input_sha256=inputs, output=str(output.resolve()),
                      output_rows=len(audited), unique_keys=len(audited))
        audited.to_parquet(output, index=False)
        report["output_sha256"] = _hash(output)
        _write_new(year_report, report)
        result["years"][str(year)] = report
        print(json.dumps({"year": year, "available_fraction": report["available_fraction"],
                          "unknown_fraction": report["unknown_fraction"],
                          "affected_rows": report["affected_rows"]}), flush=True)
    _write_new(args.report, result)
    print(f"Archive timestamp proxy audit: {args.report}", flush=True)


if __name__ == "__main__":
    main()
