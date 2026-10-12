"""Select and confirm a label-free gate for issued multi-horizon GFS features."""

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
from haze_forecast.gfs_regime_gate import GATE_NAMES, gate_mask, select_gate
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts" / "gfs_regime_gate"
STRICT_PREDICTIONS = ROOT / "artifacts" / "strict_history_backtest" / "predictions.parquet"
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
    source: path.with_name(path.name.replace("f006", "f024")) for source, path in GFS_SOURCES.items()
}
GFS_F024_FEATURES = tuple(column.replace("f006", "f024") for column in GFS_F006_FEATURES)


def prepare(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Use the deployed missing-data schema and source-scoped event keys."""
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result["event_key"] = source + ":" + result["event_id"].astype(str)
    result = add_missingness_indicators(result, PBL_COLUMNS)
    result["boundary_layer_height_m_change_3_is_missing"] = (
        result["boundary_layer_height_m"].isna()
        | result["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return result


def attach_gfs(
    frame: pd.DataFrame,
    source: str,
    forecast_rows: pd.DataFrame | None,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Attach only forecast fields issued before their prediction timestamps."""
    result = frame.copy()
    for column in feature_columns:
        result[column] = np.nan
    if forecast_rows is None:
        return result
    validate_gfs_forecast_timing(forecast_rows)
    forecast = forecast_rows.loc[:, ["timestamp", *feature_columns]].copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], utc=True, errors="raise")
    if forecast["timestamp"].duplicated().any():
        raise ValueError(f"{source} GFS rows have duplicate timestamps")
    result = result.drop(columns=list(feature_columns)).merge(
        forecast, on="timestamp", how="left", validate="one_to_one"
    )
    if result.loc[result["event_key"].str.startswith(source + ":"), list(feature_columns)].isna().all(
        axis=None
    ):
        raise ValueError(f"{source} did not receive any GFS features")
    return result


def metrics(actual: pd.Series, prediction: np.ndarray, event_key: pd.Series) -> dict[str, float]:
    """Return row and event-balanced error metrics."""
    observed = actual.to_numpy(dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    return {
        "row_mae": float(np.abs(predicted - observed).mean()),
        "event_equal_mae": event_equal_mae(observed, predicted, event_key),
        "prediction_mean": float(predicted.mean()),
        "actual_mean": float(observed.mean()),
    }


def predictions(
    train: pd.DataFrame, test: pd.DataFrame, deployed_features: list[str], enhanced_features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the base and multi-horizon AFT paths on the same event-isolated rows."""
    base = AFTSurvivalRegressor(seed=42, nthread=1).fit(
        train[deployed_features], train["remaining_hours"]
    )
    enhanced = AFTSurvivalRegressor(seed=42, nthread=1).fit(
        train[enhanced_features], train["remaining_hours"]
    )
    return (
        np.asarray(base.predict(test[deployed_features]), dtype=float).clip(min=0.0),
        np.asarray(enhanced.predict(test[enhanced_features]), dtype=float).clip(min=0.0),
    )


def score_year(
    train: pd.DataFrame, test: pd.DataFrame, deployed_features: list[str], enhanced_features: list[str]
) -> dict[str, object]:
    """Score all predeclared gate rules from one pair of fitted models."""
    base, enhanced = predictions(train, test, deployed_features, enhanced_features)
    result: dict[str, object] = {
        "base": metrics(test["remaining_hours"], base, test["event_key"]),
        "enhanced": metrics(test["remaining_hours"], enhanced, test["event_key"]),
        "gates": {},
    }
    for gate_name in GATE_NAMES:
        mask = gate_mask(test, gate_name)
        gated = np.where(mask.to_numpy(), enhanced, base).clip(min=0.0)
        result["gates"][gate_name] = {
            **metrics(test["remaining_hours"], gated, test["event_key"]),
            "coverage": float(mask.mean()),
        }
    return result


def load_rows() -> pd.DataFrame:
    """Load all labelled sources with both time-safe forecast horizons attached."""
    source_frames = {
        "prefix_2014": prepare(pd.read_parquet(DATA / "cnemc_chengdu_2014_training_features.parquet"), "prefix_2014"),
        "base": prepare(pd.read_parquet(DATA / "training_features.parquet"), "base"),
        "external_2022": prepare(pd.read_parquet(DATA / "cnemc_chengdu_2022_training_features.parquet"), "external_2022"),
        "external_2023": prepare(pd.read_parquet(DATA / "cnemc_chengdu_2023_training_features.parquet"), "external_2023"),
        "external_2024": prepare(pd.read_parquet(DATA / "cnemc_chengdu_2024_training_features.parquet"), "external_2024"),
        "external_2025": prepare(pd.read_parquet(DATA / "cnemc_chengdu_2025_training_features.parquet"), "external_2025"),
    }
    for source, frame in list(source_frames.items()):
        source_frames[source] = attach_gfs(
            frame,
            source,
            pd.read_parquet(GFS_SOURCES[source]) if source in GFS_SOURCES else None,
            GFS_F006_FEATURES,
        )
        source_frames[source] = attach_gfs(
            source_frames[source],
            source,
            pd.read_parquet(GFS_F024_SOURCES[source]) if source in GFS_F024_SOURCES else None,
            GFS_F024_FEATURES,
        )
    return pd.concat(source_frames.values(), ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def strict_audit(
    selected_gate: str,
    all_rows: pd.DataFrame,
    deployed_features: list[str],
    enhanced_features: list[str],
) -> dict[str, object]:
    """Audit a fixed gate event-by-event without using each event's future label in training."""
    strict = pd.read_parquet(STRICT_PREDICTIONS).copy()
    strict["timestamp"] = pd.to_datetime(strict["timestamp"], utc=True, errors="raise")
    forecast_columns = [*GFS_F006_FEATURES, *GFS_F024_FEATURES]
    lookup = all_rows.loc[:, ["timestamp", "event_key", *forecast_columns]]
    strict = strict.merge(lookup, on=["timestamp", "event_key"], how="left", validate="one_to_one")
    event_end = all_rows.groupby("event_key", sort=False)["timestamp"].max()
    annual: dict[str, list[dict[str, float]]] = {str(year): [] for year in range(2022, 2026)}
    for event_key, test in strict.groupby("event_key", sort=False):
        year = int(test["timestamp"].dt.year.iloc[0])
        if year not in range(2022, 2026):
            continue
        completed = event_end[event_end.lt(test["timestamp"].min())].index
        train = all_rows[all_rows["event_key"].isin(completed)]
        base, enhanced = predictions(train, test, deployed_features, enhanced_features)
        gated = np.where(gate_mask(test, selected_gate).to_numpy(), enhanced, base).clip(min=0.0)
        annual[str(year)].append(
            {
                "baseline_absolute_error": float(np.abs(base - test["remaining_hours"].to_numpy(float)).mean()),
                "gated_absolute_error": float(np.abs(gated - test["remaining_hours"].to_numpy(float)).mean()),
            }
        )
    return {
        year: {
            "baseline_row_mae": float(np.mean([item["baseline_absolute_error"] for item in values])),
            "gated_row_mae": float(np.mean([item["gated_absolute_error"] for item in values])),
            "events": len(values),
        }
        for year, values in annual.items()
        if values
    }


def main() -> None:
    metadata_path = ROOT / "artifacts" / "cnemc_2025_deployment_candidate" / "model_metadata.json"
    deployed_features = list(json.loads(metadata_path.read_text(encoding="utf-8"))["feature_names"])
    if len(deployed_features) != 87:
        raise ValueError(f"expected 87 deployed features, got {len(deployed_features)}")
    enhanced_features = [*deployed_features, *GFS_F006_FEATURES, *GFS_F024_FEATURES]
    all_rows = load_rows()
    if set(enhanced_features) - set(all_rows.columns):
        raise ValueError("GFS feature schema is incomplete")

    yearly: dict[str, object] = {}
    for year in range(2022, 2026):
        train, test = event_isolated_year_split(all_rows, year, event_column="event_key")
        yearly[str(year)] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_events": int(test["event_key"].nunique()),
            **score_year(train, test, deployed_features, enhanced_features),
        }
    selection = {name: value["row_mae"] for name, value in yearly["2024"]["gates"].items()}
    confirmation = {name: value["row_mae"] for name, value in yearly["2025"]["gates"].items()}
    coverage = {name: value["coverage"] for name, value in yearly["2024"]["gates"].items()}
    selected_2024 = min(selection, key=selection.__getitem__)
    accepted_gate = select_gate(selection, confirmation, coverage)

    if accepted_gate is None:
        audit: dict[str, object] = {"status": "not_run", "reason": "forward_confirmation_rejected"}
        strict_not_worse = False
        strict_improved_years = 0
    else:
        audit = strict_audit(accepted_gate, all_rows, deployed_features, enhanced_features)
        strict_not_worse = all(
            values["gated_row_mae"] <= values["baseline_row_mae"] for values in audit.values()
        )
        strict_improved_years = sum(
            values["gated_row_mae"] < values["baseline_row_mae"] for values in audit.values()
        )
        if not strict_not_worse or strict_improved_years < 2:
            accepted_gate = None

    report = {
        "protocol": (
            "event-isolated annual forward validation; each GFS row is validated as issued before "
            "the prediction timestamp and valid after it. 2024 selects a predeclared gate and 2025 confirms it."
        ),
        "feature_count": len(deployed_features),
        "years": yearly,
        "selected_from_2024": selected_2024,
        "strict_history_2022_2025": audit,
        "decision": {
            "status": "accept" if accepted_gate is not None else "reject",
            "selected_gate": accepted_gate,
            "strict_not_worse_all_years": strict_not_worse,
            "strict_improved_years": strict_improved_years,
            "deployment_artifact_changed": False,
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"2024 selected: {selected_2024} ({selection[selected_2024]:.3f} MAE)")
    print(f"2025 baseline: {confirmation['never']:.3f} MAE")
    print(f"2025 selected: {confirmation[selected_2024]:.3f} MAE")
    print(f"decision: {report['decision']['status']}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
