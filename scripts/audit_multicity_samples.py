"""Write a read-only feasibility report for locally available multi-city samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from haze_forecast.multicity_audit import (
    DEFAULT_MINIMUM_COVERAGE,
    audit_local_multicity_samples,
    chengdu_validation_start_from_metadata,
)
from haze_forecast.storage import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local multi-city samples without training a model.")
    parser.add_argument("--city-data-dir", type=Path, default=Path("data/raw/data"))
    parser.add_argument("--chengdu-metadata", type=Path, default=Path("artifacts/model_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/multicity_audit/report.json"))
    parser.add_argument("--minimum-coverage", type=float, default=DEFAULT_MINIMUM_COVERAGE)
    parser.add_argument("--minimum-complete-events", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.minimum_coverage <= 1:
        parser.error("--minimum-coverage must be in (0, 1]")
    if args.minimum_complete_events < 1:
        parser.error("--minimum-complete-events must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    validation_start = chengdu_validation_start_from_metadata(args.chengdu_metadata)
    report = audit_local_multicity_samples(
        args.city_data_dir,
        validation_start,
        minimum_coverage=args.minimum_coverage,
        minimum_complete_events=args.minimum_complete_events,
    )
    write_json_atomic(args.output, report)
    print({"eligible_city_count": report["eligible_city_count"], "eligible": report["eligible"]})


if __name__ == "__main__":
    main()
