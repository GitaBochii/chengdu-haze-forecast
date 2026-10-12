"""Retrain archived common-event AFT forecasts without modifying deployed artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost

from haze_forecast.training import AFTSurvivalRegressor
from scripts.build_strict_history_backtest import ROOT, SOURCES, prepare

OUTPUT = ROOT / "artifacts/task01_recovery_head/aft_common_event_reproduction_v1"
COMMON = ROOT / "artifacts/task01_recovery_head/conditional_strict_history_common_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    start = time.monotonic()
    metadata_path = ROOT / "artifacts/cnemc_2025_deployment_candidate/model_metadata.json"
    old_path = ROOT / "artifacts/strict_history_backtest/predictions.parquet"
    common_paths = [COMMON / f"{year}_common.parquet" for year in (2024, 2025)]
    protected = list(SOURCES.values()) + common_paths + [old_path, metadata_path]
    protected += list((ROOT / "artifacts/cnemc_2025_deployment_candidate").rglob("*"))
    protected += [
        Path(__file__).resolve(), ROOT / "scripts/build_strict_history_backtest.py",
        ROOT / "src/haze_forecast/training.py", ROOT / "src/haze_forecast/strict_backtest.py",
        ROOT / "src/haze_forecast/features.py",
    ]
    hashes = {str(path): sha256(path) for path in protected if path.is_file()}
    names = json.loads(metadata_path.read_text(encoding="utf-8"))["feature_names"]
    assert len(names) == 87 and len(set(names)) == 87
    # Reproduce both sorts from build_strict_history_backtest and rolling_event_backtest.
    rows = pd.concat([
        prepare(pd.read_parquet(path), source, names) for source, path in SOURCES.items()
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    rows = rows.sort_values("timestamp").reset_index(drop=True)
    assert not rows.duplicated(["event_key", "timestamp"]).any()
    common = pd.concat([pd.read_parquet(path) for path in common_paths], ignore_index=True)
    assert not common.duplicated(["event_key", "timestamp"]).any()
    requested = set(common.event_key)
    old = pd.read_parquet(old_path)
    assert not old.duplicated(["event_key", "timestamp"]).any()
    events = rows.groupby("event_key", sort=False)["timestamp"].agg(
        event_start="min", event_end="max"
    ).sort_values(["event_start", "event_end"])
    assert requested.issubset(events.index)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps({
        "input_sha256": hashes, "features": names,
        "event_keys": sorted(requested), "seed": 42, "nthread": 1,
        "num_boost_round": 500, "confirmation_hours": 5,
        "python": platform.python_version(), "xgboost": xgboost.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        "purpose": "reproduction_only_not_model_selection",
    }, indent=2), encoding="utf-8")
    completed = []
    summaries, predictions = [], []
    for key, event in events.iterrows():
        if key not in requested:
            completed.append(key)
            continue
        training = rows.loc[rows.event_key.isin(completed)]
        test = rows.loc[rows.event_key.eq(key)].copy()
        known_at = training.timestamp + pd.to_timedelta(training.remaining_hours + 5, unit="h")
        assert len(training) >= 100
        assert training.remaining_hours.notna().all() and np.isfinite(training.remaining_hours).all()
        assert (known_at < event.event_start).all(), f"Label maturity failed: {key}"
        assert training.timestamp.max() < event.event_start
        assert key not in set(training.event_key)
        directory = OUTPUT / key.replace(":", "__")
        directory.mkdir()
        provenance = training[["timestamp", "event_key", "remaining_hours"]].copy()
        provenance.insert(0, "ordered_row_index", training.index.to_numpy())
        provenance["label_available_at"] = known_at
        provenance.to_parquet(directory / "training_rows.parquet", index=False)
        model = AFTSurvivalRegressor(seed=42, nthread=1).fit(training[names], training.remaining_hours)
        predicted = model.predict(test[names]).astype(float)
        joblib.dump(model, directory / "model.joblib")
        replayed = joblib.load(directory / "model.joblib").predict(test[names]).astype(float)
        assert np.array_equal(predicted, replayed)
        assert np.isfinite(predicted).all()
        test = test[["timestamp", "event_key", "remaining_hours"]].copy()
        test["reproduced_prediction_hours"] = predicted
        merged = test.merge(old[["timestamp", "event_key", "remaining_hours", "strict_prediction_hours"]],
                            on=["timestamp", "event_key"], how="left", validate="one_to_one",
                            suffixes=("", "_old"), indicator=True)
        assert merged._merge.eq("both").all()
        assert np.array_equal(merged.remaining_hours, merged.remaining_hours_old)
        merged["absolute_reproduction_difference"] = abs(
            merged.reproduced_prediction_hours - merged.strict_prediction_hours)
        merged.drop(columns="_merge").to_parquet(directory / "predictions.parquet", index=False)
        summary = {
            "event_key": key, "event_start": event.event_start.isoformat(),
            "train_rows": len(training), "train_events": training.event_key.nunique(),
            "train_end_timestamp": training.timestamp.max().isoformat(),
            "latest_training_label_available_at": known_at.max().isoformat(),
            "test_rows": len(test), "maximum_old_prediction_difference": float(
                merged.absolute_reproduction_difference.max()), "maximum_reload_difference": 0.0,
            "model_sha256": sha256(directory / "model.joblib"),
            "training_rows_sha256": sha256(directory / "training_rows.parquet"),
            "predictions_sha256": sha256(directory / "predictions.parquet"),
        }
        summaries.append(summary)
        predictions.append(merged.drop(columns="_merge"))
        print(json.dumps(summary), flush=True)
        completed.append(key)
    assert len(summaries) == len(requested)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_parquet(OUTPUT / "all_event_predictions.parquet", index=False)
    comparison = common.merge(all_predictions[["timestamp", "event_key", "reproduced_prediction_hours"]],
                              on=["timestamp", "event_key"], how="left", validate="one_to_one")
    assert comparison.reproduced_prediction_hours.notna().all()
    comparison.to_parquet(OUTPUT / "common_predictions.parquet", index=False)
    unchanged = {path: sha256(Path(path)) == value for path, value in hashes.items()}
    assert all(unchanged.values())
    report = {
        "status": "completed_reproduction_only", "events": summaries,
        "event_count": len(summaries), "common_rows": len(comparison),
        "all_event_rows": len(all_predictions), "label_maturity_verified": True,
        "input_and_deployment_unchanged": unchanged,
        "maximum_old_prediction_difference": float(all_predictions.absolute_reproduction_difference.max()),
        "maximum_common_old_prediction_difference": float(abs(
            comparison.reproduced_prediction_hours - comparison.strict_prediction_hours).max()),
        "elapsed_seconds": time.monotonic() - start,
        "output_sha256": {p.name: sha256(p) for p in [
            OUTPUT / "common_predictions.parquet", OUTPUT / "all_event_predictions.parquet"]},
        "limitations": ["Reproduction is not evidence of improved accuracy",
                        "Current environment may differ from original training environment"],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}), flush=True)


if __name__ == "__main__":
    main()
