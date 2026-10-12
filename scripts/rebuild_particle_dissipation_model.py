"""Build the particle-dissipation target dataset and both prediction models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import model_feature_names, snapshot_feature_names, training_rows
from haze_forecast.labels import label_city_data, label_summary
from haze_forecast.training import train_and_evaluate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    artifacts = root / "artifacts" / "particle_dissipation"
    city = pd.read_parquet(root / "data" / "interim" / "city_hourly.parquet")
    labeled = label_city_data(city, require_aqi_recovery=False)
    features = training_rows(labeled)

    processed.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(processed / "particle_dissipation_city_labeled.parquet", index=False)
    features.to_parquet(processed / "particle_dissipation_training_features.parquet", index=False)
    report = {
        "target": "颗粒物雾霾消散",
        "definition": "连续3小时颗粒物主导污染后开始，连续6小时非颗粒物主导后结束",
        "label_summary": label_summary(labeled),
        "training_rows": len(features),
        "history_feature_count": len(model_feature_names(features)),
        "snapshot_feature_count": len(snapshot_feature_names(features)),
    }
    (processed / "particle_dissipation_label_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = train_and_evaluate(features, artifacts, trials=30, seed=42)
    print(json.dumps({"label_report": report, "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
