from pathlib import Path

import pandas as pd

from haze_forecast.trajectory_benchmark import benchmark_trajectory_features


def _training_frame() -> pd.DataFrame:
    rows = []
    for event_id in range(1, 9):
        for step, remaining in enumerate((30.0, 20.0, 10.0)):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2020-01-01", tz="Asia/Shanghai")
                    + pd.Timedelta(hours=(event_id - 1) * 4 + step),
                    "event_id": event_id,
                    "remaining_hours": remaining,
                    "aqi": 110.0 + event_id + step,
                    "pollution_age_hours": float(step + 1),
                    "event_peak_aqi": 110.0 + event_id + step,
                    "event_peak_pm2_5": 80.0 + event_id + step,
                    "event_peak_pm10": 160.0 + event_id + step,
                    "aqi_from_event_peak": 0.0,
                    "pm2_5_from_event_peak": 0.0,
                    "pm10_from_event_peak": 0.0,
                    "aqi_cumulative_excess": 10.0 + event_id + step,
                    "pm2_5_cumulative_excess": 5.0 + event_id + step,
                    "pm10_cumulative_excess": 10.0 + event_id + step,
                    "event_peak_age_hours": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_trajectory_benchmark_uses_identical_disjoint_event_folds():
    report, predictions = benchmark_trajectory_features(_training_frame(), n_splits=3)

    for fold in report["folds"]:
        assert not set(fold["train_event_ids"]).intersection(fold["validation_event_ids"])
    assert set(predictions["variant"]) == {"baseline", "with_trajectory"}


def test_trajectory_script_uses_isolated_candidate_paths():
    text = Path("scripts/benchmark_trajectory_features.py").read_text(encoding="utf-8")

    assert "trajectory_training_features.parquet" in text
    assert "artifacts/trajectory_feature_benchmark" in text
