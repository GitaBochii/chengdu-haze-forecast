"""时间序列训练、调参与保留测试集评估。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from haze_forecast.evaluation import regression_metrics, write_evaluation_charts
from haze_forecast.features import model_feature_names, snapshot_feature_names
from haze_forecast.storage import dump_model_atomic, write_json_atomic


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows in time order into 60% train, 20% validation and 20% test."""
    if len(frame) < 5:
        raise ValueError("at least five samples are required for chronological splitting")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    train_end = max(1, int(len(ordered) * 0.6))
    validation_end = max(train_end + 1, int(len(ordered) * 0.8))
    validation_end = min(validation_end, len(ordered) - 1)
    return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]


def event_chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split completed events chronologically without placing one event in multiple partitions."""
    required = {"timestamp", "event_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for event split: {sorted(missing)}")

    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    event_sizes = ordered.groupby("event_id", sort=False).size()
    if len(event_sizes) < 3:
        raise ValueError("at least three completed events are required for event splitting")

    cumulative_rows = event_sizes.cumsum().to_numpy()
    event_count = len(event_sizes)
    train_event_end = min(
        max(1, int(np.searchsorted(cumulative_rows, len(ordered) * 0.6, side="left") + 1)),
        event_count - 2,
    )
    validation_event_end = min(
        max(
            train_event_end + 1,
            int(np.searchsorted(cumulative_rows, len(ordered) * 0.8, side="left") + 1),
        ),
        event_count - 1,
    )
    event_ids = event_sizes.index.to_list()
    train_ids = event_ids[:train_event_end]
    valid_ids = event_ids[train_event_end:validation_event_end]
    test_ids = event_ids[validation_event_end:]
    return (
        ordered[ordered["event_id"].isin(train_ids)].reset_index(drop=True),
        ordered[ordered["event_id"].isin(valid_ids)].reset_index(drop=True),
        ordered[ordered["event_id"].isin(test_ids)].reset_index(drop=True),
    )


def constrain_event_predictions(frame: pd.DataFrame, predictions: pd.Series) -> pd.Series:
    """Keep predicted remaining time non-increasing by at least one hour inside an event."""
    required = {"timestamp", "event_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for event constraint: {sorted(missing)}")
    if len(frame) != len(predictions):
        raise ValueError("prediction count must match frame rows")

    constrained = pd.Series(predictions.to_numpy(dtype=float), index=frame.index).clip(lower=0)
    ordered = frame.assign(_prediction=constrained).sort_values("timestamp")
    for _, event in ordered.groupby("event_id", sort=False):
        previous: float | None = None
        for index, value in event["_prediction"].items():
            current = float(value) if previous is None else min(float(value), max(0.0, previous - 1.0))
            constrained.loc[index] = current
            previous = current
    return constrained


def empirical_interval(prediction: float, validation_mae: float) -> tuple[float, float]:
    """Use validation MAE as a transparent empirical half-width."""
    return max(0.0, prediction - validation_mae), prediction + validation_mae


def aft_label_bounds(remaining_hours: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Convert exact observed dissipation times into XGBoost AFT label bounds."""
    observed = pd.to_numeric(remaining_hours, errors="raise").to_numpy(dtype=float)
    lower_bound = np.maximum(observed, 1e-3)
    return lower_bound, lower_bound.copy()


def rolling_time_splits(
    frame: pd.DataFrame, n_splits: int = 3
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return expanding-window validation folds in chronological order."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return [(ordered.iloc[train_index], ordered.iloc[valid_index]) for train_index, valid_index in splitter.split(ordered)]


def rolling_event_splits(
    frame: pd.DataFrame, n_splits: int = 3
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return expanding validation folds whose training and validation events are disjoint."""
    required = {"timestamp", "event_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns for rolling event split: {sorted(missing)}")

    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    event_ids = ordered["event_id"].drop_duplicates().to_numpy()
    if len(event_ids) <= n_splits:
        raise ValueError("more completed events than splits are required")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    return [
        (
            ordered[ordered["event_id"].isin(event_ids[train_index])].reset_index(drop=True),
            ordered[ordered["event_id"].isin(event_ids[valid_index])].reset_index(drop=True),
        )
        for train_index, valid_index in splitter.split(event_ids)
    ]


def _matrix(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return frame[feature_names].astype(float)


class AFTSurvivalRegressor:
    """Small XGBoost wrapper that predicts median hours until event recovery."""

    def __init__(
        self, num_boost_round: int = 500, seed: int = 42, nthread: int | None = None
    ) -> None:
        self.num_boost_round = num_boost_round
        self.seed = seed
        self.nthread = nthread
        self.booster: xgb.Booster | None = None

    def fit(
        self,
        features: pd.DataFrame,
        remaining_hours: pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> AFTSurvivalRegressor:
        """Fit the AFT model.

        ``sample_weight`` lets callers down-weight stale rows when the target
        distribution drifts over time.  It is optional so existing call sites
        keep their current behaviour.
        """
        lower_bound, upper_bound = aft_label_bounds(remaining_hours)
        matrix = xgb.DMatrix(features.astype(float), missing=np.nan)
        matrix.set_float_info("label_lower_bound", lower_bound)
        matrix.set_float_info("label_upper_bound", upper_bound)
        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=float)
            if weight.shape != (len(features),):
                raise ValueError("sample_weight must have one entry per row")
            if np.any(weight < 0) or not np.all(np.isfinite(weight)):
                raise ValueError("sample_weight must be finite and non-negative")
            matrix.set_weight(weight)
        params: dict[str, float | int | str] = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.2,
            "eta": 0.05,
            "max_depth": 4,
            "min_child_weight": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "lambda": 3.0,
            "seed": self.seed,
            "tree_method": "hist",
            "verbosity": 0,
        }
        if self.nthread is not None:
            params["nthread"] = self.nthread
        self.booster = xgb.train(
            params,
            matrix,
            num_boost_round=self.num_boost_round,
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise ValueError("AFT model must be fitted before prediction")
        hours = self.booster.predict(xgb.DMatrix(features.astype(float), missing=np.nan))
        return hours.clip(min=0.0)


def _build_model(params: dict[str, float | int], seed: int, n_estimators: int = 1000) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )


def _build_snapshot_catboost(iterations: int = 500) -> CatBoostRegressor:
    """Build the validated single-timestamp model without creating CatBoost side files."""
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=8,
        random_strength=1,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def _tune_lightgbm(
    train: pd.DataFrame, valid: pd.DataFrame, feature_names: list[str], trials: int, seed: int
) -> tuple[dict[str, float | int], int, float]:
    train_x, valid_x = _matrix(train, feature_names), _matrix(valid, feature_names)
    train_y, valid_y = train["remaining_hours"], valid["remaining_hours"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, float | int] = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        }
        model = _build_model(params, seed)
        model.fit(
            train_x,
            train_y,
            eval_X=valid_x,
            eval_y=valid_y,
            callbacks=[early_stopping(60, verbose=False)],
        )
        return float(mean_absolute_error(valid_y, model.predict(valid_x)))

    study.optimize(objective, n_trials=trials)
    best = {name: value for name, value in study.best_params.items()}
    model = _build_model(best, seed)
    model.fit(
        train_x,
        train_y,
        eval_X=valid_x,
        eval_y=valid_y,
        callbacks=[early_stopping(60, verbose=False)],
    )
    return best, int(model.best_iteration_ or model.n_estimators), float(study.best_value)


def _rolling_validation_metrics(
    frame: pd.DataFrame, feature_names: list[str], params: dict[str, float | int], iterations: int, seed: int
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for fold, (train, valid) in enumerate(rolling_event_splits(frame), start=1):
        model = _build_model(params, seed + fold, n_estimators=iterations)
        model.fit(_matrix(train, feature_names), train["remaining_hours"])
        results.append(regression_metrics(valid["remaining_hours"], model.predict(_matrix(valid, feature_names))))
    return results


def train_and_evaluate(
    frame: pd.DataFrame, artifacts_dir: Path, trials: int = 30, seed: int = 42
) -> dict[str, object]:
    """Train baselines and LightGBM, then evaluate only once on the held-out final period."""
    if len(frame) < 100:
        raise ValueError("at least 100 labeled samples are required for training")
    history_feature_names = model_feature_names(frame)
    static_feature_names = snapshot_feature_names(frame)
    if not history_feature_names or not static_feature_names:
        raise ValueError("no numeric model features are available")
    train, valid, test = event_chronological_split(frame)
    history_params, history_iterations, history_validation_mae = _tune_lightgbm(
        train, valid, history_feature_names, trials, seed
    )
    snapshot_catboost = _build_snapshot_catboost()
    snapshot_catboost.fit(
        _matrix(train, static_feature_names),
        train["remaining_hours"],
        eval_set=(_matrix(valid, static_feature_names), valid["remaining_hours"]),
        early_stopping_rounds=60,
        verbose=False,
    )
    snapshot_iterations = max(1, int(snapshot_catboost.get_best_iteration()))
    snapshot_validation_mae = float(
        mean_absolute_error(valid["remaining_hours"], snapshot_catboost.predict(_matrix(valid, static_feature_names)))
    )
    aft_validation_model = AFTSurvivalRegressor(seed=seed).fit(
        _matrix(train, history_feature_names), train["remaining_hours"]
    )
    aft_snapshot_validation_model = AFTSurvivalRegressor(seed=seed).fit(
        _matrix(train, static_feature_names), train["remaining_hours"]
    )
    aft_validation_mae = float(
        mean_absolute_error(
            valid["remaining_hours"], aft_validation_model.predict(_matrix(valid, history_feature_names))
        )
    )
    aft_snapshot_validation_mae = float(
        mean_absolute_error(
            valid["remaining_hours"],
            aft_snapshot_validation_model.predict(_matrix(valid, static_feature_names)),
        )
    )
    combined = pd.concat([train, valid], ignore_index=True)
    history_rolling_metrics = _rolling_validation_metrics(
        combined, history_feature_names, history_params, history_iterations, seed
    )
    history_model = _build_model(history_params, seed, n_estimators=history_iterations)
    history_model.fit(_matrix(combined, history_feature_names), combined["remaining_hours"])
    snapshot_model = _build_snapshot_catboost(snapshot_iterations)
    snapshot_model.fit(_matrix(combined, static_feature_names), combined["remaining_hours"])
    aft_model = AFTSurvivalRegressor(seed=seed).fit(
        _matrix(combined, history_feature_names), combined["remaining_hours"]
    )
    aft_snapshot_model = AFTSurvivalRegressor(seed=seed).fit(
        _matrix(combined, static_feature_names), combined["remaining_hours"]
    )

    baseline = DummyRegressor(strategy="median").fit(
        _matrix(train, history_feature_names), train["remaining_hours"]
    )
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)).fit(
        _matrix(train, history_feature_names), train["remaining_hours"]
    )
    history_test_x, test_y = _matrix(test, history_feature_names), test["remaining_hours"]
    static_test_x = _matrix(test, static_feature_names)
    predicted = pd.Series(history_model.predict(history_test_x), index=test.index).clip(lower=0)
    static_predicted = pd.Series(snapshot_model.predict(static_test_x), index=test.index).clip(lower=0)
    aft_predicted = pd.Series(aft_model.predict(history_test_x), index=test.index).clip(lower=0)
    aft_static_predicted = pd.Series(aft_snapshot_model.predict(static_test_x), index=test.index).clip(lower=0)
    state_constrained_predicted = constrain_event_predictions(test, predicted)
    predictions = pd.DataFrame(
        {
            "timestamp": test["timestamp"],
            "actual_remaining_hours": test_y,
            "predicted_remaining_hours": predicted,
        }
    )
    predictions["interval_low_hours"] = (predicted - history_validation_mae).clip(lower=0)
    predictions["interval_high_hours"] = predicted + history_validation_mae
    predictions["state_constrained_remaining_hours"] = state_constrained_predicted
    static_predictions = predictions.assign(
        predicted_remaining_hours=static_predicted,
        interval_low_hours=(static_predicted - snapshot_validation_mae).clip(lower=0),
        interval_high_hours=static_predicted + snapshot_validation_mae,
    )
    aft_predictions = predictions.assign(
        predicted_remaining_hours=aft_predicted,
        interval_low_hours=(aft_predicted - aft_validation_mae).clip(lower=0),
        interval_high_hours=aft_predicted + aft_validation_mae,
    )
    aft_static_predictions = predictions.assign(
        predicted_remaining_hours=aft_static_predicted,
        interval_low_hours=(aft_static_predicted - aft_snapshot_validation_mae).clip(lower=0),
        interval_high_hours=aft_static_predicted + aft_snapshot_validation_mae,
    )
    importance = pd.DataFrame(
        {"feature": history_feature_names, "importance": history_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dump_model_atomic(artifacts_dir / "lightgbm_model.joblib", history_model)
    dump_model_atomic(artifacts_dir / "static_catboost_model.joblib", snapshot_model)
    dump_model_atomic(artifacts_dir / "aft_model.joblib", aft_model)
    dump_model_atomic(artifacts_dir / "static_aft_model.joblib", aft_snapshot_model)
    predictions.to_csv(artifacts_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    static_predictions.to_csv(artifacts_dir / "static_test_predictions.csv", index=False, encoding="utf-8-sig")
    aft_predictions.to_csv(artifacts_dir / "aft_test_predictions.csv", index=False, encoding="utf-8-sig")
    aft_static_predictions.to_csv(
        artifacts_dir / "static_aft_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    importance.to_csv(artifacts_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    write_evaluation_charts(predictions, importance, artifacts_dir)

    metrics = {
        "median_baseline": regression_metrics(test_y, baseline.predict(history_test_x)),
        "ridge_baseline": regression_metrics(test_y, ridge.predict(history_test_x)),
        "lightgbm": regression_metrics(test_y, predicted),
        "lightgbm_state_constrained": regression_metrics(test_y, state_constrained_predicted),
        "catboost_snapshot": regression_metrics(test_y, static_predicted),
        "xgboost_aft": regression_metrics(test_y, aft_predicted),
        "xgboost_aft_snapshot": regression_metrics(test_y, aft_static_predicted),
        "validation_mae": history_validation_mae,
        "snapshot_validation_mae": snapshot_validation_mae,
        "aft_validation_mae": aft_validation_mae,
        "aft_snapshot_validation_mae": aft_snapshot_validation_mae,
    }
    selected_history_model = (
        ("XGBoostAFT", "aft_model.joblib", aft_validation_mae)
        if aft_validation_mae < history_validation_mae
        else ("LightGBMRegressor", "lightgbm_model.joblib", history_validation_mae)
    )
    selected_snapshot_model = (
        ("XGBoostAFT", "static_aft_model.joblib", aft_snapshot_validation_mae)
        if aft_snapshot_validation_mae < snapshot_validation_mae
        else ("CatBoostRegressor", "static_catboost_model.joblib", snapshot_validation_mae)
    )
    metadata: dict[str, object] = {
        "model": "LightGBMRegressor",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "feature_names": history_feature_names,
        "snapshot_feature_names": static_feature_names,
        "best_params": history_params,
        "best_iteration": history_iterations,
        "snapshot_model": "CatBoostRegressor",
        "snapshot_model_filename": "static_catboost_model.joblib",
        "snapshot_best_iteration": snapshot_iterations,
        "aft_model": "XGBoostAFT",
        "aft_model_filename": "aft_model.joblib",
        "aft_snapshot_model_filename": "static_aft_model.joblib",
        "aft_num_boost_round": aft_model.num_boost_round,
        "selected_history_model": selected_history_model[0],
        "selected_history_model_filename": selected_history_model[1],
        "selected_history_validation_mae": selected_history_model[2],
        "selected_snapshot_model": selected_snapshot_model[0],
        "selected_snapshot_model_filename": selected_snapshot_model[1],
        "selected_snapshot_validation_mae": selected_snapshot_model[2],
        "seed": seed,
        "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
        "time_ranges": {
            "train": [train["timestamp"].min().isoformat(), train["timestamp"].max().isoformat()],
            "validation": [valid["timestamp"].min().isoformat(), valid["timestamp"].max().isoformat()],
            "test": [test["timestamp"].min().isoformat(), test["timestamp"].max().isoformat()],
        },
        "rolling_validation": history_rolling_metrics,
    }
    write_json_atomic(artifacts_dir / "metrics.json", metrics)
    write_json_atomic(artifacts_dir / "model_metadata.json", metadata)
    return {"metrics": metrics, "metadata": metadata}
