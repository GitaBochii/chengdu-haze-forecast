"""Predeclared 2023 rolling AFT event-weight development comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost

from haze_forecast.training import AFTSurvivalRegressor
from scripts.build_strict_history_backtest import ROOT, SOURCES, prepare
from scripts.reproduce_common_event_aft import sha256

OUTPUT = ROOT / "artifacts/task01_recovery_head/aft_event_weight_development_v1"
ARMS = {"row_control": 0.0, "inverse_sqrt_event": 0.5, "inverse_event": 1.0}


def event_weights(keys: pd.Series, power: float) -> np.ndarray:
    if keys.empty or keys.isna().any() or not np.isfinite(power) or power < 0:
        raise ValueError("Expected nonempty valid event keys and nonnegative finite power")
    counts = keys.map(keys.value_counts()).to_numpy(dtype=float)
    weights = counts ** (-power)
    return weights / weights.mean()


def assert_mature(training: pd.DataFrame, test_start: pd.Timestamp) -> pd.Series:
    assert len(training) > 0
    assert np.isfinite(training.remaining_hours).all()
    assert training.remaining_hours.ge(0).all()
    known_at = training.timestamp + pd.to_timedelta(training.remaining_hours + 5, unit="h")
    assert known_at.notna().all() and (known_at < test_start).all()
    return known_at


def metrics(frame: pd.DataFrame) -> dict:
    result = {"rows": len(frame), "events": int(frame.event_key.nunique())}
    for name, cap in (("full", None), ("cap68", 68.0)):
        actual = frame.remaining_hours.to_numpy(float)
        predicted = frame.prediction_hours.to_numpy(float)
        if cap is not None:
            actual, predicted = np.minimum(actual, cap), np.minimum(predicted, cap)
        error = pd.Series(abs(actual - predicted), index=frame.index)
        result[name + "_row_mae"] = float(error.mean())
        result[name + "_event_equal_mae"] = float(error.groupby(frame.event_key).mean().mean())
        if cap is not None:
            tail = frame.remaining_hours.ge(68)
            result["cap68_tail_row_mae"] = float(error[tail].mean())
            result["cap68_tail_event_equal_mae"] = float(error[tail].groupby(frame.loc[tail, "event_key"]).mean().mean())
            result["tail_rows"] = int(tail.sum())
            result["tail_events"] = int(frame.loc[tail, "event_key"].nunique())
    return result


def main() -> None:
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = time.monotonic()
    metadata = ROOT / "artifacts/cnemc_2025_deployment_candidate/model_metadata.json"
    old_path = ROOT / "artifacts/strict_history_backtest/predictions.parquet"
    protected = list(SOURCES.values()) + [metadata, old_path, Path(__file__).resolve(),
        ROOT / "aft_event_weight_design_20260906.md",
        ROOT / "scripts/build_strict_history_backtest.py",
        ROOT / "scripts/reproduce_common_event_aft.py",
        ROOT / "src/haze_forecast/training.py", ROOT / "src/haze_forecast/features.py",
        ROOT / "src/haze_forecast/strict_backtest.py"]
    protected += [p for p in (ROOT / "artifacts/cnemc_2025_deployment_candidate").rglob("*") if p.is_file()]
    hashes = {str(p): sha256(p) for p in protected}
    names = json.loads(metadata.read_text(encoding="utf-8"))["feature_names"]
    assert len(names) == len(set(names)) == 87
    rows = pd.concat([prepare(pd.read_parquet(p), source, names) for source, p in SOURCES.items()],
                     ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    rows = rows.sort_values("timestamp").reset_index(drop=True)
    assert not rows.duplicated(["event_key", "timestamp"]).any()
    old = pd.read_parquet(old_path)
    old = old.loc[old.timestamp.dt.year.eq(2023)].copy()
    requested = set(old.event_key)
    assert requested and not old.duplicated(["event_key", "timestamp"]).any()
    events = rows.groupby("event_key", sort=False).timestamp.agg(event_start="min", event_end="max").sort_values(["event_start", "event_end"])
    assert requested.issubset(events.index)
    OUTPUT.mkdir(parents=True)
    protocol = {"input_sha256": hashes, "features": names, "arms": ARMS,
        "event_keys": sorted(requested), "year": 2023, "seed": 42, "nthread": 1,
        "num_boost_round": 500, "confirmation_hours": 5, "xgboost": xgboost.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        "weights": "training event row count ** -power, normalized to mean 1 per split",
        "gate": "full row AND event equal MAE strictly improve AND cap68 tail event equal MAE not worse",
        "selection": "full event equal MAE, full row MAE, lower weighting power",
        "future_candidate_scores_read": False}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    completed, summaries, predictions = [], [], []
    for key, event in events.iterrows():
        if key not in requested:
            completed.append(key)
            continue
        training = rows.loc[rows.event_key.isin(completed)]
        test = rows.loc[rows.event_key.eq(key)].copy()
        assert test.timestamp.dt.year.eq(2023).all()
        assert len(training) >= 100 and key not in set(training.event_key)
        known_at = assert_mature(training, event.event_start)
        directory = OUTPUT / key.replace(":", "__")
        directory.mkdir()
        provenance = training[["timestamp", "event_key", "remaining_hours"]].copy()
        provenance.insert(0, "ordered_row_index", training.index.to_numpy())
        provenance["label_available_at"] = known_at
        for arm, power in ARMS.items():
            provenance[arm + "_weight"] = event_weights(training.event_key, power)
        provenance.to_parquet(directory / "training_rows.parquet", index=False)
        forecast = test[["timestamp", "event_key", "remaining_hours"]].copy()
        forecast.insert(0, "ordered_row_index", test.index.to_numpy())
        summary = {"event_key": key, "event_start": event.event_start.isoformat(),
                   "train_rows": len(training), "train_events": int(training.event_key.nunique()),
                   "latest_training_label_available_at": known_at.max().isoformat(),
                   "test_rows": len(test), "models": {}}
        for arm in ARMS:
            weights = provenance[arm + "_weight"].to_numpy()
            model = AFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit(
                training[names], training.remaining_hours, sample_weight=weights)
            path = directory / (arm + ".joblib")
            joblib.dump(model, path)
            predicted = model.predict(test[names]).astype(float)
            assert np.isfinite(predicted).all()
            assert np.array_equal(predicted, joblib.load(path).predict(test[names]).astype(float))
            forecast[arm] = predicted
            summary["models"][arm] = sha256(path)
        merged = forecast.merge(old[["timestamp", "event_key", "remaining_hours", "strict_prediction_hours"]],
            on=["timestamp", "event_key"], how="left", validate="one_to_one", suffixes=("", "_old"), indicator=True)
        assert merged._merge.eq("both").all()
        assert np.array_equal(merged.remaining_hours, merged.remaining_hours_old)
        assert np.array_equal(merged.row_control, merged.strict_prediction_hours)
        merged = merged.drop(columns="_merge")
        merged.to_parquet(directory / "predictions.parquet", index=False)
        summary["training_rows_sha256"] = sha256(directory / "training_rows.parquet")
        summary["predictions_sha256"] = sha256(directory / "predictions.parquet")
        predictions.append(merged)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        completed.append(key)
    frame = pd.concat(predictions, ignore_index=True)
    assert len(frame) == len(old) and len(summaries) == len(requested)
    frame.to_parquet(OUTPUT / "predictions.parquet", index=False)
    scores = {arm: metrics(frame.rename(columns={arm: "prediction_hours"})) for arm in ARMS}
    base = scores["row_control"]
    passed = [arm for arm in ARMS if arm != "row_control" and
              scores[arm]["full_row_mae"] < base["full_row_mae"] and
              scores[arm]["full_event_equal_mae"] < base["full_event_equal_mae"] and
              scores[arm]["cap68_tail_event_equal_mae"] <= base["cap68_tail_event_equal_mae"]]
    passed.sort(key=lambda arm: (scores[arm]["full_event_equal_mae"], scores[arm]["full_row_mae"], ARMS[arm]))
    unchanged = {path: sha256(Path(path)) == value for path, value in hashes.items()}
    assert all(unchanged.values())
    report = {"status": "completed_2023_development_only", "events": summaries, "scores": scores,
              "passed_arms": passed, "selected_arm": passed[0] if passed else None,
              "input_and_deployment_unchanged": unchanged, "maximum_old_prediction_difference": 0.0,
              "predictions_sha256": sha256(OUTPUT / "predictions.parquet"),
              "protocol_sha256": sha256(OUTPUT / "protocol.json"), "elapsed_seconds": time.monotonic() - started,
              "limitations": ["2023 is development data, not independent confirmation", "No deployment changed"]}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("events", "input_and_deployment_unchanged")}), flush=True)


if __name__ == "__main__":
    main()
