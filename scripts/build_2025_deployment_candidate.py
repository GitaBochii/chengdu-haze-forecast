"""Refit a future-use model on every completed 2014-2025 observation.

The 2025 external-forward result is frozen before this script runs.  It remains
the candidate's stated external reference; this script never scores the model
on rows it has just fitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from haze_forecast.full_history_candidate import build_full_history_deployment_candidate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
SOURCES = {
    "cnemc_2014": DATA / "cnemc_chengdu_2014_training_features.parquet",
    "cnemc_2015_2021": DATA / "training_features.parquet",
    "cnemc_2022": DATA / "cnemc_chengdu_2022_training_features.parquet",
    "cnemc_2023": DATA / "cnemc_chengdu_2023_training_features.parquet",
    "cnemc_2024_2025": DATA / "cnemc_chengdu_2024_2025_training_features.parquet",
}
ARTIFACT_DIR = ARTIFACTS / "cnemc_2025_deployment_candidate"
COMBINED_OUTPUT = DATA / "cnemc_chengdu_2014_2025_deployment_training_features.parquet"


def main() -> None:
    metadata = json.loads((ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8"))
    forward_report = json.loads(
        (ARTIFACTS / "cnemc_2024_2025_forward" / "report.json").read_text(encoding="utf-8")
    )
    confirmation = forward_report["confirmation_year_2025"]["selected"]
    source_frames = {name: pd.read_parquet(path) for name, path in SOURCES.items()}
    result = build_full_history_deployment_candidate(
        source_frames,
        ARTIFACT_DIR,
        feature_names=list(metadata["feature_names"]),
        snapshot_feature_names=list(metadata["snapshot_feature_names"]),
        external_mae=float(confirmation["row_mae"]),
        external_metric_source=(
            "2025 event-isolated forward validation; model selection predeclared from 2024 "
            f"as {forward_report['confirmation_year_2025']['predeclared_from_2024']}"
        ),
    )
    temporary = COMBINED_OUTPUT.with_suffix(".parquet.part")
    result["training"].to_parquet(temporary, index=False)
    temporary.replace(COMBINED_OUTPUT)
    print(f"fitted through: {result['metadata']['fitted_through']}")
    print(f"full-history rows: {result['metadata']['final_training_rows']}")
    print(f"frozen external MAE reference: {confirmation['row_mae']:.3f} h")
    print(f"artifacts: {ARTIFACT_DIR}")
    print(f"combined feature data: {COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
