import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.hazard_benchmark import (
    benchmark_discrete_hazard,
    write_hazard_benchmark_artifacts,
)


def test_benchmark_hazard_uses_identical_complete_event_folds(monkeypatch):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="h"),
            "event_id": np.repeat(np.arange(1, 7), 2),
            "remaining_hours": np.arange(12, 0, -1, dtype=float),
            "history_feature": np.arange(12, 0, -1, dtype=float),
        }
    )
    calls = []

    def fake_aft(train_x, train_y, valid_x, seed):
        calls.append(("aft", train_x.columns.tolist(), train_y.index.equals(train_x.index), seed))
        return valid_x["history_feature"].to_numpy() + 2.0

    def fake_hazard(train_x, train_y, valid_x, seed):
        calls.append(("hazard", train_x.columns.tolist(), train_y.index.equals(train_x.index), seed))
        return valid_x["history_feature"].to_numpy() + 1.0, np.zeros(len(valid_x), dtype=bool)

    monkeypatch.setattr("haze_forecast.hazard_benchmark._fit_predict_aft", fake_aft)
    monkeypatch.setattr("haze_forecast.hazard_benchmark._fit_predict_hazard", fake_hazard)

    report, predictions = benchmark_discrete_hazard(frame, ["history_feature"], n_splits=3, seed=10)

    assert report["protocol"] == "expanding chronological complete-event rolling validation"
    assert len(calls) == 6
    assert all(call[1] == ["history_feature"] and call[2] for call in calls)
    assert all(
        set(fold["train_event_ids"]).isdisjoint(fold["validation_event_ids"])
        for fold in report["folds"]
    )
    assert set(predictions["model"]) == {"xgboost_aft", "discrete_hazard"}
    assert report["models"]["discrete_hazard"]["right_censor_rate"] == 0.0
    assert report["decision"]["decision"] == "candidate_for_replacement_review"


def test_write_hazard_benchmark_artifacts_exports_auditable_files(tmp_path):
    predictions = pd.DataFrame(
        {
            "fold": [1],
            "model": ["discrete_hazard"],
            "event_id": [1],
            "actual_remaining_hours": [4.0],
            "predicted_remaining_hours": [4.0],
            "right_censored": [False],
        }
    )
    output = tmp_path / "hazard"

    write_hazard_benchmark_artifacts({"protocol": "test"}, predictions, output)

    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == {"protocol": "test"}
    assert pd.read_csv(output / "fold_predictions.csv", encoding="utf-8-sig").shape == (1, 6)


def test_benchmark_discrete_hazard_script_writes_a_separate_output(monkeypatch, capsys):
    script_path = Path(__file__).parents[1] / "scripts" / "benchmark_discrete_hazard.py"
    spec = importlib.util.spec_from_file_location("benchmark_discrete_hazard_script", script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "event_id": [1, 1],
            "remaining_hours": [2.0, 1.0],
            "history_feature": [2.0, 1.0],
        }
    )
    report = {"decision": {"decision": "candidate_for_replacement_review"}}
    predictions = pd.DataFrame()
    captured = {}
    monkeypatch.setattr(script.pd, "read_parquet", lambda path: frame)
    monkeypatch.setattr(script, "model_feature_names", lambda value: ["history_feature"])
    monkeypatch.setattr(
        script,
        "benchmark_discrete_hazard",
        lambda value, feature_names, n_splits, seed: (report, predictions),
    )
    monkeypatch.setattr(
        script,
        "write_hazard_benchmark_artifacts",
        lambda received_report, received_predictions, output: captured.update(
            report=received_report, predictions=received_predictions, output=output
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script_path), "--data", "fixture.parquet", "--output", "hazard-output", "--splits", "3"],
    )

    script.main()

    assert captured == {"report": report, "predictions": predictions, "output": Path("hazard-output")}
    assert json.loads(capsys.readouterr().out)["decision"] == "candidate_for_replacement_review"
