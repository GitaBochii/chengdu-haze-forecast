"""Benchmark audited NOAA ISD hourly features without altering the default model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from haze_forecast.noaa_benchmark import (
    audit_is_eligible,
    audit_report_sha256,
    benchmark_noaa_isd_features,
    write_noaa_benchmark_artifacts,
)


def _exactly_three(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be exactly 3") from error
    if parsed != 3:
        raise argparse.ArgumentTypeError("must be exactly 3")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--city-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=_exactly_three, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "rejected", "reason": f"invalid audit report: {error}"}, ensure_ascii=False))
        return 0
    if not isinstance(audit, dict) or not audit_is_eligible(audit):
        print(
            json.dumps(
                {"status": "rejected", "eligible": False, "reason": "NOAA audit is not eligible"},
                ensure_ascii=False,
            )
        )
        return 0

    try:
        hourly = pd.read_parquet(args.hourly)
        city_labeled = pd.read_parquet(args.city_data)
        report, predictions = benchmark_noaa_isd_features(
            city_labeled,
            hourly,
            audit,
            audit_report_sha256=audit_report_sha256(args.audit),
            audit_report_path=args.audit,
            hourly_path=args.hourly,
            n_splits=args.splits,
            seed=args.seed,
        )
        write_noaa_benchmark_artifacts(report, predictions, args.output)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}, ensure_ascii=False))
        return 0

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(args.output),
                "training_rows": report["training_rows"],
                "decision": report["decision"]["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
