"""Compare deployed-aligned AFT against the same model augmented with issued GFS.

Historical rows without an archived forecast remain in the training set with
missing GFS columns.  This makes the comparison conservative: the base model
keeps every prior labelled event, while the augmented model can only benefit
where an issued forecast is actually available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators
from haze_forecast.gfs_external_validation import (
    GFS_F006_FEATURES,
    event_equal_mae,
    validate_gfs_forecast_timing,
)
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "gfs_augmented_aft" / "report.json"
PBL_COLUMNS = [
    "boundary_layer_height_m",
    "boundary_layer_height_m_lag_1",
    "boundary_layer_height_m_lag_3",
    "boundary_layer_height_m_lag_24",
    "boundary_layer_height_m_roll_3",
    "boundary_layer_height_m_roll_24",
    "boundary_layer_height_m_change_3",
]
GFS_SOURCES = {
    "base": DATA / "gfs_event_features_f006_2021_all.parquet",
    "external_2022": DATA / "gfs_event_features_f006_2022_all.parquet",
    "external_2023": DATA / "gfs_event_features_f006_2023_all.parquet",
    "external_2024": DATA / "gfs_event_features_f006_2024_all.parquet",
    "external_2025": DATA / "gfs_event_features_f006_2025_all.parquet",
}
GFS_F024_SOURCES = {
    "base": DATA / "gfs_event_features_f024_2021_all.parquet",
    "external_2022": DATA / "gfs_event_features_f024_2022_all.parquet",
    "external_2023": DATA / "gfs_event_features_f024_2023_all.parquet",
    "external_2024": DATA / "gfs_event_features_f024_2024_all.parquet",
    "external_2025": DATA / "gfs_event_features_f024_2025_all.parquet",
}
GFS_F024_FEATURES = tuple(column.replace("f006", "f024") for column in GFS_F006_FEATURES)


def prepare(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result["event_key"] = source + ":" + result["event_id"].astype(str)
    result = add_missingness_indicators(result, PBL_COLUMNS)
    result["boundary_layer_height_m_change_3_is_missing"] = (
        result["boundary_layer_height_m"].isna() | result["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return result


def attach_gfs(
    frame: pd.DataFrame,
    source: str,
    forecast_rows: pd.DataFrame | None,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    result = frame.copy()
    gfs_columns = list(feature_columns)
    for column in gfs_columns:
        result[column] = np.nan
    if forecast_rows is None:
        return result
    validate_gfs_forecast_timing(forecast_rows)
    forecast = forecast_rows.loc[:, ["timestamp", *gfs_columns]].copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True, errors="raise")
    if forecast["timestamp"].duplicated().any():
        raise ValueError(f"{source} GFS rows have duplicate timestamps")
    result = result.drop(columns=gfs_columns).merge(
        forecast, on="timestamp", how="left", validate="one_to_one"
    )
    if result.loc[result["event_key"].str.startswith(source + ":"), gfs_columns].isna().all(axis=None):
        raise ValueError(f"{source} did not receive any GFS features")
    return result


def score(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, float]:
    model = AFTSurvivalRegressor(seed=42, nthread=1).fit(train[features], train["remaining_hours"])
    prediction = np.asarray(model.predict(test[features]), dtype=float).clip(min=0.0)
    actual = test["remaining_hours"].to_numpy(float)
    return {
        "row_mae": float(np.abs(prediction - actual).mean()),
        "event_equal_mae": event_equal_mae(actual, prediction, test["event_key"]),
        "prediction_mean": float(prediction.mean()),
        "actual_mean": float(actual.mean()),
    }


def main() -> None:
    metadata = json.loads((ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8"))
    deployed_features = list(metadata["feature_names"])
    gfs_rows = {source: pd.read_parquet(path) for source, path in GFS_SOURCES.items()}
    gfs_f024_rows = {source: pd.read_parquet(path) for source, path in GFS_F024_SOURCES.items()}
    base = attach_gfs(
        prepare(pd.read_parquet(DATA / "training_features.parquet"), "base"),
        "base",
        gfs_rows["base"],
        GFS_F006_FEATURES,
    )
    prefix = attach_gfs(
        prepare(pd.read_parquet(DATA / "cnemc_chengdu_2014_training_features.parquet"), "prefix_2014"),
        "prefix_2014",
        None,
        GFS_F006_FEATURES,
    )
    y22 = attach_gfs(
        prepare(pd.read_parquet(DATA / "cnemc_chengdu_2022_training_features.parquet"), "external_2022"),
        "external_2022",
        gfs_rows["external_2022"],
        GFS_F006_FEATURES,
    )
    y23 = attach_gfs(
        prepare(pd.read_parquet(DATA / "cnemc_chengdu_2023_training_features.parquet"), "external_2023"),
        "external_2023",
        gfs_rows["external_2023"],
        GFS_F006_FEATURES,
    )
    y24 = attach_gfs(
        prepare(pd.read_parquet(DATA / "cnemc_chengdu_2024_training_features.parquet"), "external_2024"),
        "external_2024",
        gfs_rows["external_2024"],
        GFS_F006_FEATURES,
    )
    y25 = attach_gfs(
        prepare(pd.read_parquet(DATA / "cnemc_chengdu_2025_training_features.parquet"), "external_2025"),
        "external_2025",
        gfs_rows["external_2025"],
        GFS_F006_FEATURES,
    )
    base = attach_gfs(base, "base", gfs_f024_rows["base"], GFS_F024_FEATURES)
    prefix = attach_gfs(prefix, "prefix_2014", None, GFS_F024_FEATURES)
    y22 = attach_gfs(y22, "external_2022", gfs_f024_rows["external_2022"], GFS_F024_FEATURES)
    y23 = attach_gfs(y23, "external_2023", gfs_f024_rows["external_2023"], GFS_F024_FEATURES)
    y24 = attach_gfs(y24, "external_2024", gfs_f024_rows["external_2024"], GFS_F024_FEATURES)
    y25 = attach_gfs(y25, "external_2025", gfs_f024_rows["external_2025"], GFS_F024_FEATURES)
    all_rows = pd.concat([prefix, base, y22, y23, y24, y25], ignore_index=True)
    augmented_features = [*deployed_features, *GFS_F006_FEATURES]
    multi_horizon_features = [*augmented_features, *GFS_F024_FEATURES]
    result: dict[str, object] = {
        "protocol": (
            "event-isolated annual forward test; baseline uses the 87 deployed-aligned features; "
            "augmented AFT adds issued GFS +6h and +24h fields, with historical unavailable GFS left missing"
        ),
        "gfs_feature_coverage": {
            source: {
                "f006_rows": int(frame[list(GFS_F006_FEATURES)].notna().all(axis=1).sum()),
                "f024_rows": int(frame[list(GFS_F024_FEATURES)].notna().all(axis=1).sum()),
            }
            for source, frame in {
                "prefix_2014": prefix,
                "base": base,
                "external_2022": y22,
                "external_2023": y23,
                "external_2024": y24,
                "external_2025": y25,
            }.items()
        },
        "years": {},
    }
    for year in (2022, 2023, 2024, 2025):
        train, test = event_isolated_year_split(all_rows, year, event_column="event_key")
        baseline = score(train, test, deployed_features)
        augmented = score(train, test, augmented_features)
        multi_horizon = score(train, test, multi_horizon_features)
        row_change = 1.0 - multi_horizon["row_mae"] / baseline["row_mae"]
        event_change = 1.0 - multi_horizon["event_equal_mae"] / baseline["event_equal_mae"]
        result["years"][str(year)] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_events": int(test["event_key"].nunique()),
            "baseline_deployed_aligned": baseline,
            "plus_gfs_f006": augmented,
            "plus_gfs_f006_f024": multi_horizon,
            "relative_row_mae_improvement": row_change,
            "relative_event_equal_mae_improvement": event_change,
        }
        print(
            f"{year}: row MAE {baseline['row_mae']:.3f} -> {augmented['row_mae']:.3f} -> "
            f"{multi_horizon['row_mae']:.3f} ({row_change:+.2%}); event MAE "
            f"{baseline['event_equal_mae']:.3f} -> {augmented['event_equal_mae']:.3f} -> "
            f"{multi_horizon['event_equal_mae']:.3f} ({event_change:+.2%})"
        )
    result["decision"] = (
        "candidate_for_operational_gfs_design"
        if all(
            values["relative_row_mae_improvement"] > 0
            and values["relative_event_equal_mae_improvement"] > 0
            for values in result["years"].values()
        )
        else "reject_gfs_multi_horizon_for_deployed_aligned_aft"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结论: {result['decision']}")
    print(f"报告: {OUTPUT}")


if __name__ == "__main__":
    main()
