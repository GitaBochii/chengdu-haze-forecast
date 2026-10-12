"""Materialize auditable event-isolated rolling predictions for the history screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators
from haze_forecast.strict_backtest import rolling_event_backtest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts" / "strict_history_backtest"
SOURCES = {
    "cnemc_2014": DATA / "cnemc_chengdu_2014_training_features.parquet",
    "cnemc_2015_2021": DATA / "training_features.parquet",
    "cnemc_2022": DATA / "cnemc_chengdu_2022_training_features.parquet",
    "cnemc_2023": DATA / "cnemc_chengdu_2023_training_features.parquet",
    "cnemc_2024_2025": DATA / "cnemc_chengdu_2024_2025_training_features.parquet",
}


def prepare(frame: pd.DataFrame, source: str, feature_names: list[str]) -> pd.DataFrame:
    """Preserve source-scoped event identity and match the deployed 87-feature schema."""
    prepared = frame.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="raise")
    prepared["event_key"] = source + ":" + prepared["event_id"].astype(str)
    pbl_sources = [
        name
        for name in feature_names
        if name.startswith("boundary_layer_height_m") and not name.endswith("_is_missing")
    ]
    prepared = add_missingness_indicators(prepared, pbl_sources)
    prepared["boundary_layer_height_m_change_3_is_missing"] = (
        prepared["boundary_layer_height_m"].isna()
        | prepared["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return prepared


def main() -> None:
    metadata = json.loads(
        (ROOT / "artifacts" / "cnemc_2025_deployment_candidate" / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    feature_names = list(metadata["feature_names"])
    frames = [
        prepare(pd.read_parquet(path), source, feature_names) for source, path in SOURCES.items()
    ]
    all_rows = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    predictions = rolling_event_backtest(all_rows, feature_names=feature_names)
    if predictions.empty:
        raise RuntimeError("没有足够的早期训练样本，无法生成严格历史回测")
    error = np.abs(predictions["strict_prediction_hours"] - predictions["remaining_hours"])
    predictions["strict_absolute_error_hours"] = error
    years = predictions["timestamp"].dt.year
    report = {
        "protocol": (
            "expanding event-isolated history backtest; every event is predicted by an AFT model "
            "trained only on completed events ending before that event begins"
        ),
        "feature_count": len(feature_names),
        "minimum_train_rows": 100,
        "eligible_rows": len(predictions),
        "eligible_events": int(predictions["event_key"].nunique()),
        "first_eligible_timestamp": predictions["timestamp"].min().isoformat(),
        "last_eligible_timestamp": predictions["timestamp"].max().isoformat(),
        "overall_row_mae": float(error.mean()),
        "yearly_row_mae": {
            str(year): float(predictions.loc[years.eq(year), "strict_absolute_error_hours"].mean())
            for year in sorted(years.unique())
        },
        "output": str(ARTIFACTS / "predictions.parquet"),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(ARTIFACTS / "predictions.parquet", index=False)
    (ARTIFACTS / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
