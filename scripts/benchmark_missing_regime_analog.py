"""Evaluate a conservative analog blend for rows without boundary-layer observations.

The candidate never changes predictions for the PBL-observed regime.  For a
PBL-missing row, it blends the AFT estimate with nearby completed historical
pollution states using only measurements available at the prediction time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.analog_blend import MissingRegimeAnalogBlend
from haze_forecast.features import add_missingness_indicators
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts" / "missing_regime_analog_benchmark"
RETRIEVAL_FEATURES = [
    "aqi",
    "pm2_5",
    "pm10",
    "so2",
    "no2",
    "co",
    "o3",
    "temperature_c",
    "relative_humidity_pct",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "wind_speed_ms",
    "ongoing_pollution_hours",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_24",
    "aqi_roll_3",
    "pm2_5_lag_1",
    "pm2_5_lag_3",
    "pm2_5_lag_24",
    "pm2_5_roll_3",
    "pm10_lag_1",
    "pm10_lag_3",
    "pm10_lag_24",
    "pm10_roll_3",
]


def prepare_frame(filename: str, source: str) -> pd.DataFrame:
    """Load one source while preserving source-scoped event identity."""
    frame = pd.read_parquet(PROCESSED / filename).copy()
    frame["event_key"] = source + ":" + frame["event_id"].astype(str)
    pbl_sources = [name for name in frame.columns if name.startswith("boundary_layer_height_m")]
    prepared = add_missingness_indicators(frame, pbl_sources)
    prepared["boundary_layer_height_m_change_3_is_missing"] = (
        prepared["boundary_layer_height_m"].isna()
        | prepared["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return prepared


def metrics(actual: pd.Series, prediction: np.ndarray, event_key: pd.Series) -> dict[str, float]:
    error = np.abs(np.asarray(prediction, dtype=float) - actual.to_numpy(float))
    by_event = pd.DataFrame({"event_key": event_key.to_numpy(), "error": error}).groupby("event_key")[
        "error"
    ].mean()
    return {"row_mae": float(error.mean()), "event_equal_mae": float(by_event.mean())}


def main() -> None:
    frames = [
        prepare_frame("cnemc_chengdu_2014_training_features.parquet", "prefix_2014"),
        prepare_frame("training_features.parquet", "base"),
        prepare_frame("cnemc_chengdu_2022_training_features.parquet", "external_2022"),
        prepare_frame("cnemc_chengdu_2023_training_features.parquet", "external_2023"),
    ]
    all_rows = pd.concat(frames, ignore_index=True)
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"))
    model_features = list(metadata["feature_names"])
    if set(RETRIEVAL_FEATURES) - set(all_rows.columns):
        raise ValueError("retrieval feature schema is incomplete")

    report: dict[str, object] = {
        "protocol": (
            "event-isolated annual forward evaluation; rows with observed PBL keep the AFT prediction; "
            "PBL-missing rows use a 0.1 blend with five inverse-distance historical analogs"
        ),
        "retrieval_features": RETRIEVAL_FEATURES,
        "neighbors": 5,
        "blend_weight": 0.1,
        "years": {},
    }
    for year in range(2018, 2024):
        train, test = event_isolated_year_split(all_rows, test_year=year, event_column="event_key")
        model = AFTSurvivalRegressor(seed=42, nthread=1).fit(
            train[model_features], train["remaining_hours"]
        )
        base_prediction = np.asarray(model.predict(test[model_features]), dtype=float)
        missing_pbl = test["boundary_layer_height_m_is_missing"].eq(1.0).to_numpy()
        candidate = MissingRegimeAnalogBlend(
            feature_names=RETRIEVAL_FEATURES, neighbors=5, blend_weight=0.1
        ).fit(train[RETRIEVAL_FEATURES], train["remaining_hours"])
        blended_prediction = candidate.predict(base_prediction, test[RETRIEVAL_FEATURES], missing_pbl)
        actual = test["remaining_hours"]
        report["years"][str(year)] = {
            "training_rows": len(train),
            "test_rows": len(test),
            "test_events": int(test["event_key"].nunique()),
            "pbl_missing_rows": int(missing_pbl.sum()),
            "baseline": metrics(actual, base_prediction, test["event_key"]),
            "analog_blend": metrics(actual, blended_prediction, test["event_key"]),
        }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for year, result in report["years"].items():
        baseline = result["baseline"]["row_mae"]
        candidate = result["analog_blend"]["row_mae"]
        print(f"{year}: {baseline:.3f} -> {candidate:.3f} ({1 - candidate / baseline:+.2%})")
    print(f"报告已写入 {ARTIFACTS / 'report.json'}")


if __name__ == "__main__":
    main()
