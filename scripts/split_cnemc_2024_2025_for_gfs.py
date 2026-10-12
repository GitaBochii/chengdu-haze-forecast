"""Write one calendar year's labelled CNEMC rows for leakage-safe GFS extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, choices=(2024, 2025))
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/processed/cnemc_chengdu_2024_2025_training_features.parquet",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or ROOT / "data/processed" / f"cnemc_chengdu_{args.year}_training_features.parquet"
    frame = pd.read_parquet(args.input)
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    yearly = frame.loc[timestamps.dt.year.eq(args.year)].copy()
    if yearly.empty:
        raise ValueError(f"{args.year} has no labelled feature rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_parquet(output, index=False)
    print(f"wrote {len(yearly)} labelled rows to {output}")


if __name__ == "__main__":
    main()
