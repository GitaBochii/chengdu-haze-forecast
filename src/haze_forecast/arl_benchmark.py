"""Leave-one-complete-event-out comparison of observation-only and ARL +3 hour features."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from haze_forecast.training import constrain_event_predictions

BASELINE_MODEL_NAME = "observations_only"
ARL_MODEL_NAME = "observations_plus_arl_f003"
EXPLORATORY_EVENT_MINIMUM = 7
DECISION_EVENT_MINIMUM = 15
REQUIRED_RELATIVE_GAIN = 0.05


def build_benchmark_model(**overrides: object) -> CatBoostRegressor:
    """Return the shared CatBoost regressor used by every compared feature set."""
    params: dict[str, object] = {
        "loss_function": "MAE",
        "iterations": 300,
        "learning_rate": 0.04,
        "depth": 4,
        "l2_leaf_reg": 10,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    params.update(overrides)
    return CatBoostRegressor(**params)


def _check_columns(frame: pd.DataFrame, feature_names: Sequence[str]) -> list[str]:
    missing = sorted({"timestamp", "event_id", "remaining_hours", *feature_names} - set(frame.columns))
    if missing:
        raise ValueError(f"benchmark frame is missing columns: {missing}")
    return list(feature_names)


def compare_feature_sets(
    frame: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    model_factory: Callable[[], CatBoostRegressor] = build_benchmark_model,
) -> dict[str, object]:
    """Evaluate every feature set with one identical leave-one-complete-event-out protocol.

    Each fold trains on all rows of every other event, so no event contributes to
    its own training set, and both models are constrained the same way afterwards.
    """
    if not feature_sets:
        raise ValueError("at least one feature set is required")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("benchmark frame is empty")
    checked = {
        name: _check_columns(ordered, names) for name, names in feature_sets.items()
    }
    events = sorted(int(value) for value in ordered["event_id"].unique())
    if len(events) < 2:
        raise ValueError("leave-one-event-out comparison needs at least two events")

    records: list[dict[str, float | int | str]] = []
    for event_id in events:
        train = ordered.loc[ordered["event_id"].ne(event_id)]
        test = ordered.loc[ordered["event_id"].eq(event_id)]
        for model_name, feature_names in checked.items():
            model = model_factory().fit(train[feature_names], train["remaining_hours"])
            raw_prediction = pd.Series(model.predict(test[feature_names]), index=test.index)
            prediction = constrain_event_predictions(test, raw_prediction)
            records.append(
                {
                    "event_id": event_id,
                    "model": model_name,
                    "rows": len(test),
                    "mae": float(mean_absolute_error(test["remaining_hours"], prediction)),
                }
            )
    comparison = pd.DataFrame(records)
    overall = (
        comparison.assign(weighted_error=comparison["mae"] * comparison["rows"])
        .groupby("model", as_index=False)
        .agg(rows=("rows", "sum"), weighted_error=("weighted_error", "sum"))
    )
    overall["mae"] = overall["weighted_error"] / overall["rows"]
    overall = overall.drop(columns="weighted_error")
    mae_by_model = {
        str(row["model"]): float(row["mae"]) for _, row in overall.iterrows()
    }
    if BASELINE_MODEL_NAME not in mae_by_model:
        raise ValueError(f"feature sets must include {BASELINE_MODEL_NAME}")

    baseline_mae = mae_by_model[BASELINE_MODEL_NAME]
    baseline_rows = comparison.loc[comparison["model"].eq(BASELINE_MODEL_NAME)]
    baseline_error = baseline_rows["mae"] * baseline_rows["rows"]
    total_baseline_error = float(baseline_error.sum())
    dominant_index = baseline_error.idxmax()
    rows_per_event = ordered["event_id"].value_counts()
    payload: dict[str, object] = {
        "protocol": (
            "leave-one-complete-event-out; identical folds, rows and constraint for both models"
        ),
        "rows": len(ordered),
        "events": len(events),
        "event_ids": events,
        "largest_event_row_share": float(rows_per_event.max() / len(ordered)),
        "largest_event_id": int(rows_per_event.idxmax()),
        "largest_event_error_share": float(baseline_error.max() / total_baseline_error)
        if total_baseline_error
        else 0.0,
        "largest_error_event_id": int(baseline_rows.loc[dominant_index, "event_id"]),
        "overall": overall.to_dict(orient="records"),
        "per_event": comparison.to_dict(orient="records"),
    }
    if ARL_MODEL_NAME in mae_by_model:
        arl_mae = mae_by_model[ARL_MODEL_NAME]
        relative_gain = (baseline_mae - arl_mae) / baseline_mae if baseline_mae else 0.0
        payload["mae_change_arl_minus_baseline"] = arl_mae - baseline_mae
        payload["relative_gain"] = relative_gain
        payload["decision"] = decide_from_mae(len(events), baseline_mae, arl_mae)
    return payload


def decide_from_mae(events: int, baseline_mae: float, arl_mae: float) -> str:
    """Apply the agreed acceptance gate to a completed ARL comparison."""
    relative_gain = (baseline_mae - arl_mae) / baseline_mae if baseline_mae else 0.0
    if arl_mae >= baseline_mae:
        return "reject_arl_for_main_model"
    if events < EXPLORATORY_EVENT_MINIMUM:
        return "exploratory_only_insufficient_events"
    if events < DECISION_EVENT_MINIMUM or relative_gain < REQUIRED_RELATIVE_GAIN:
        return "candidate_for_more_data"
    return "candidate_model"


def summarise_per_event_direction(per_event: pd.DataFrame) -> dict[str, int]:
    """Count how many events improve, worsen or stay level when ARL features are added."""
    wide = per_event.pivot(index="event_id", columns="model", values="mae")
    if ARL_MODEL_NAME not in wide.columns:
        return {}
    difference = wide[ARL_MODEL_NAME] - wide[BASELINE_MODEL_NAME]
    return {
        "improved": int(difference.lt(0).sum()),
        "worsened": int(difference.gt(0).sum()),
        "events": len(difference),
    }
