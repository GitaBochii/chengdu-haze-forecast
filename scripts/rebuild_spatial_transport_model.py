"""Build and evaluate a spatial-transport candidate using the local nine-station network."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import model_feature_names, snapshot_feature_names, training_rows
from haze_forecast.spatial import enrich_city_with_spatial_features
from haze_forecast.training import train_and_evaluate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "data" / "processed"
    artifacts = root / "artifacts" / "spatial_transport"
    city_labeled = pd.read_parquet(processed / "city_labeled.parquet")
    station_hourly = pd.read_parquet(root / "data" / "interim" / "station_hourly.parquet")
    coordinates = json.loads((root / "data" / "raw" / "data" / "chengdu" / "stations.json").read_text("utf-8"))

    spatial_city = enrich_city_with_spatial_features(city_labeled, station_hourly, coordinates)
    features = training_rows(spatial_city)
    spatial_city.to_parquet(processed / "spatial_city_labeled.parquet", index=False)
    features.to_parquet(processed / "spatial_training_features.parquet", index=False)

    report = {
        "target": "整体空气质量恢复",
        "station_count": len(coordinates),
        "spatial_feature_columns": [name for name in features if name.startswith("spatial_")],
        "training_rows": len(features),
        "history_feature_count": len(model_feature_names(features)),
        "snapshot_feature_count": len(snapshot_feature_names(features)),
    }
    (processed / "spatial_feature_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = train_and_evaluate(features, artifacts, trials=30, seed=42)
    print(json.dumps({"report": report, "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
