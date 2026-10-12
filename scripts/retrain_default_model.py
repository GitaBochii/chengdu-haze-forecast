"""Rebuild the default overall-air-quality recovery models and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from haze_forecast.training import train_and_evaluate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_parquet(root / "data" / "processed" / "training_features.parquet")
    result = train_and_evaluate(frame, root / "artifacts", trials=30, seed=42)
    print(result["metrics"])


if __name__ == "__main__":
    main()
