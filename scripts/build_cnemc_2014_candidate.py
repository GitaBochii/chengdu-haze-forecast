"""Build isolated deployable artifacts for the validated CNEMC 2014 candidate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from haze_forecast.cnemc_candidate import build_cnemc_2014_candidate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = pd.read_parquet(root / "data" / "processed" / "training_features.parquet")
    prefix = pd.read_parquet(
        root / "data" / "processed" / "cnemc_chengdu_2014_training_features.parquet"
    ).copy()
    prefix["event_id"] = "cnemc_2014_" + prefix["event_id"].astype(str)
    result = build_cnemc_2014_candidate(
        base,
        prefix,
        root / "artifacts" / "cnemc_2014_candidate",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
