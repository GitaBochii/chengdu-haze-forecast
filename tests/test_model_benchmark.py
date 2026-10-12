import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from haze_forecast.model_benchmark import (
    CANDIDATE_MODELS,
    benchmark_history_models,
    benchmark_prefix_data_effect,
    build_candidate_model,
    decide_candidate,
    fit_predict_candidate,
    write_benchmark_artifacts,
)
from haze_forecast.training import AFTSurvivalRegressor


@pytest.fixture
def candidate_model_data():
    train_x = pd.DataFrame(
        {
            "pm2_5": np.linspace(45, 145, 18),
            "wind_speed_ms": np.linspace(1, 8, 18),
            "relative_humidity_pct": np.linspace(88, 45, 18),
        }
    )
    train_y = pd.Series(np.linspace(48, 3, 18))
    predict_x = pd.DataFrame(
        {
            "pm2_5": [55, 95, 135],
            "wind_speed_ms": [1.5, 4.5, 7.5],
            "relative_humidity_pct": [84, 65, 48],
        }
    )
    return train_x, train_y, predict_x


class _NegativePredictionModel:
    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        return self

    def predict(self, valid_x):
        return np.array([-2.0, 1.5, 4.0])


class _FixedPredictionModel(_NegativePredictionModel):
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, valid_x):
        return self.predictions


@pytest.mark.parametrize("model_name", CANDIDATE_MODELS)
def test_fit_predict_candidate_returns_non_negative_predictions_for_each_model(
    monkeypatch, model_name, candidate_model_data
):
    train_x, train_y, valid_x = candidate_model_data
    model = _NegativePredictionModel()
    monkeypatch.setattr(
        "haze_forecast.model_benchmark.build_candidate_model", lambda name, seed: model
    )

    predictions = fit_predict_candidate(model_name, train_x, train_y, valid_x, seed=7)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(valid_x),)
    assert predictions.tolist() == [0.0, 1.5, 4.0]
    assert model.train_x.dtypes.eq(float).all()
    assert model.train_y.dtype == float


@pytest.mark.parametrize(
    ("model_name", "model_type", "expected"),
    [
        ("xgboost_aft", AFTSurvivalRegressor, {"num_boost_round": 500, "nthread": 1}),
        (
            "xgboost_absolute_error",
            xgb.XGBRegressor,
            {"objective": "reg:absoluteerror", "n_estimators": 500, "n_jobs": 1},
        ),
        (
            "lightgbm_l1",
            LGBMRegressor,
            {
                "objective": "regression_l1",
                "n_estimators": 13,
                "n_jobs": 1,
                "num_leaves": 11,
            },
        ),
        (
            "catboost_mae",
            CatBoostRegressor,
            {"iterations": 500, "loss_function": "MAE", "thread_count": 1},
        ),
    ],
)
def test_build_candidate_model_maps_each_name_to_its_fixed_backend(
    model_name, model_type, expected
):
    model = build_candidate_model(model_name, seed=7)

    assert isinstance(model, model_type)
    parameters = model.__dict__ if isinstance(model, AFTSurvivalRegressor) else model.get_params()
    for parameter, value in expected.items():
        assert parameters[parameter] == value


def test_fit_predict_candidate_rejects_unknown_model(candidate_model_data):
    train_x, train_y, valid_x = candidate_model_data

    with pytest.raises(ValueError, match="unsupported model"):
        fit_predict_candidate("not-a-candidate", train_x, train_y, valid_x)


def test_fit_predict_candidate_allows_missing_feature_values(monkeypatch, candidate_model_data):
    train_x, train_y, valid_x = candidate_model_data
    train_x.loc[0, "pm2_5"] = np.nan
    valid_x.loc[0, "wind_speed_ms"] = np.nan
    monkeypatch.setattr(
        "haze_forecast.model_benchmark.build_candidate_model",
        lambda name, seed: _NegativePredictionModel(),
    )

    predictions = fit_predict_candidate("lightgbm_l1", train_x, train_y, valid_x)

    assert predictions.tolist() == [0.0, 1.5, 4.0]


@pytest.mark.parametrize(
    ("feature_frame", "feature_value"),
    [("train", np.inf), ("train", -np.inf), ("valid", np.inf), ("valid", -np.inf)],
)
def test_fit_predict_candidate_rejects_infinite_feature_values(
    feature_frame, feature_value, candidate_model_data
):
    train_x, train_y, valid_x = candidate_model_data
    features = train_x if feature_frame == "train" else valid_x
    features["pm2_5"] = features["pm2_5"].astype(float)
    features.loc[0, "pm2_5"] = feature_value

    with pytest.raises(ValueError, match="candidate inputs"):
        fit_predict_candidate("lightgbm_l1", train_x, train_y, valid_x)


@pytest.mark.parametrize("target_value", [np.nan, np.inf, -np.inf])
def test_fit_predict_candidate_rejects_non_finite_training_targets(
    target_value, candidate_model_data
):
    train_x, train_y, valid_x = candidate_model_data
    train_y.iloc[0] = target_value

    with pytest.raises(ValueError, match="candidate inputs"):
        fit_predict_candidate("lightgbm_l1", train_x, train_y, valid_x)


@pytest.mark.parametrize(
    "predictions",
    [
        np.array([np.nan, 1.0, 2.0]),
        np.array([np.inf, 1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([[1.0], [2.0], [3.0]]),
    ],
)
def test_fit_predict_candidate_rejects_invalid_model_predictions(
    monkeypatch, predictions, candidate_model_data
):
    train_x, train_y, valid_x = candidate_model_data
    monkeypatch.setattr(
        "haze_forecast.model_benchmark.build_candidate_model",
        lambda name, seed: _FixedPredictionModel(predictions),
    )

    with pytest.raises(ValueError, match="candidate predictions"):
        fit_predict_candidate("lightgbm_l1", train_x, train_y, valid_x)


@pytest.mark.parametrize(
    "train_x,train_y,valid_x",
    [
        (np.ones((2, 2)), pd.Series([1.0, 2.0]), pd.DataFrame({"a": [1.0], "b": [2.0]})),
        (pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([1.0, 2.0]), np.ones((1, 1))),
        (pd.DataFrame({"a": []}), pd.Series(dtype=float), pd.DataFrame({"a": [1.0]})),
        (pd.DataFrame({"a": [1.0]}), pd.Series([1.0]), pd.DataFrame({"a": []})),
        (
            pd.DataFrame({"a": [1.0, 2.0]}),
            pd.Series([1.0]),
            pd.DataFrame({"a": [3.0]}),
        ),
        (
            pd.DataFrame([[1.0, 2.0]], columns=["a", "a"]),
            pd.Series([1.0]),
            pd.DataFrame([[3.0, 4.0]], columns=["a", "a"]),
        ),
        (
            pd.DataFrame({"a": [1.0], "b": [2.0]}),
            pd.Series([1.0]),
            pd.DataFrame({"b": [3.0], "a": [4.0]}),
        ),
        (pd.DataFrame({"a": [1.0]}), np.array([1.0]), pd.DataFrame({"a": [2.0]})),
    ],
)
def test_fit_predict_candidate_rejects_invalid_input_contract(train_x, train_y, valid_x):
    with pytest.raises(ValueError, match="candidate inputs"):
        fit_predict_candidate("lightgbm_l1", train_x, train_y, valid_x)


def test_fit_predict_candidate_rejects_misaligned_training_feature_and_target_indexes(
    candidate_model_data,
):
    train_x, train_y, valid_x = candidate_model_data
    train_y.index = train_y.index + 1

    with pytest.raises(ValueError, match="matching training feature and target indexes"):
        fit_predict_candidate("lightgbm_l1", train_x, train_y, valid_x)


def test_decide_candidate_requires_stable_three_fold_mae_improvement():
    result = decide_candidate(
        baseline_fold_mae=[10.0, 10.0, 10.0],
        candidate_fold_mae=[9.0, 9.5, 9.4],
    )

    assert result["decision"] == "candidate_for_replacement_review"
    assert result["relative_mae_improvement"] >= 0.03
    assert result["improved_folds"] == 3


def test_decide_candidate_rejects_one_fold_over_five_percent_regression():
    result = decide_candidate(
        baseline_fold_mae=[10.0, 10.0, 10.0],
        candidate_fold_mae=[9.0, 9.5, 10.6],
    )

    assert result["decision"] == "reject_unstable_or_insufficient_mae_improvement"
    assert result["worst_fold_relative_change"] == pytest.approx(0.06)


def test_decide_candidate_accepts_exact_gate_boundaries():
    result = decide_candidate(
        baseline_fold_mae=[10.0, 10.0, 10.0],
        candidate_fold_mae=[9.0, 9.6, 10.5],
    )

    assert result["decision"] == "candidate_for_replacement_review"
    assert result["relative_mae_improvement"] == pytest.approx(0.03)
    assert result["improved_folds"] == 2
    assert result["worst_fold_relative_change"] == pytest.approx(0.05)


def test_decide_candidate_uses_provided_overall_mae_instead_of_unweighted_fold_mean():
    baseline_fold_mae = [10.0, 10.0, 1.0]
    candidate_fold_mae = [9.0, 9.0, 1.05]

    fold_mean_decision = decide_candidate(baseline_fold_mae, candidate_fold_mae)
    overall_decision = decide_candidate(
        baseline_fold_mae,
        candidate_fold_mae,
        baseline_overall_mae=120.0 / 102.0,
        candidate_overall_mae=123.0 / 102.0,
    )

    assert fold_mean_decision["decision"] == "candidate_for_replacement_review"
    assert overall_decision["decision"] == "reject_unstable_or_insufficient_mae_improvement"
    assert overall_decision["baseline_overall_mae"] == pytest.approx(120.0 / 102.0)
    assert overall_decision["candidate_overall_mae"] == pytest.approx(123.0 / 102.0)


@pytest.mark.parametrize(
    ("baseline_overall_mae", "candidate_overall_mae"),
    [
        (1.0, None),
        (None, 1.0),
        (np.nan, 1.0),
        (1.0, np.inf),
        (0.0, 1.0),
        (1.0, -0.1),
    ],
)
def test_decide_candidate_rejects_invalid_optional_overall_mae(
    baseline_overall_mae, candidate_overall_mae
):
    with pytest.raises(ValueError):
        decide_candidate(
            [10.0, 10.0, 10.0],
            [9.0, 9.0, 9.0],
            baseline_overall_mae=baseline_overall_mae,
            candidate_overall_mae=candidate_overall_mae,
        )


@pytest.mark.parametrize(
    ("baseline_fold_mae", "candidate_fold_mae"),
    [
        ([10.0, 10.0], [9.0]),
        ([10.0, 10.0], [9.0, 9.0]),
        ([10.0, float("nan"), 10.0], [9.0, 9.0, 9.0]),
        ([10.0, 10.0, 10.0], [9.0, float("inf"), 9.0]),
        ([10.0, 0.0, 10.0], [9.0, 9.0, 9.0]),
        ([10.0, -1.0, 10.0], [9.0, 9.0, 9.0]),
        ([10.0, 10.0, 10.0], [9.0, -1.0, 9.0]),
    ],
)
def test_decide_candidate_rejects_invalid_fold_mae_inputs(
    baseline_fold_mae, candidate_fold_mae
):
    with pytest.raises(ValueError):
        decide_candidate(baseline_fold_mae, candidate_fold_mae)


def test_decide_candidate_rejects_nested_fold_mae_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        decide_candidate(
            [[10.0], [10.0], [10.0]],
            [[9.0], [9.0], [9.0]],
        )


def test_benchmark_history_models_uses_auditable_event_folds_and_aligned_candidates(
    monkeypatch,
):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="h"),
            "event_id": [event_id for event_id in range(1, 7) for _ in range(2)],
            "remaining_hours": [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "history_feature": [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    calls = []

    def fake_fit_predict(model_name, train_x, train_y, valid_x, seed):
        calls.append(
            {
                "model": model_name,
                "seed": seed,
                "train_rows": len(train_x),
                "valid_rows": len(valid_x),
                "columns": train_x.columns.tolist(),
            }
        )
        offset = 2.0 if model_name == "xgboost_aft" else 1.0
        return valid_x["history_feature"].to_numpy() + offset

    monkeypatch.setattr("haze_forecast.model_benchmark.fit_predict_candidate", fake_fit_predict)

    report, predictions = benchmark_history_models(frame, ["history_feature"], n_splits=3, seed=10)

    assert report["protocol"] == "expanding chronological complete-event rolling validation"
    json.dumps(report)
    assert len(report["folds"]) == 3
    assert len(calls) == 3 * len(CANDIDATE_MODELS)
    assert {call["seed"] for call in calls} == {11, 12, 13}
    assert all(call["columns"] == ["history_feature"] for call in calls)
    assert all(
        set(fold["train_event_ids"]).isdisjoint(fold["validation_event_ids"])
        for fold in report["folds"]
    )
    assert report["folds"][0]["train_event_ids"] == [1, 2, 3]
    assert report["folds"][0]["validation_event_ids"] == [4]
    assert all(
        fold["train_end_timestamp"] < fold["validation_start_timestamp"]
        for fold in report["folds"]
    )
    assert list(predictions.columns) == [
        "fold",
        "model",
        "timestamp",
        "event_id",
        "actual_remaining_hours",
        "predicted_remaining_hours",
    ]
    assert len(predictions) == 3 * len(CANDIDATE_MODELS) * 2
    assert report["models"]["xgboost_aft"]["fold_mae"] == [2.0, 2.0, 2.0]
    assert report["models"]["xgboost_aft"]["metrics"]["mae"] == 2.0
    assert report["models"]["lightgbm_l1"]["fold_mae"] == [1.0, 1.0, 1.0]
    assert (
        report["models"]["lightgbm_l1"]["decision_against_xgboost_aft"]["decision"]
        == "candidate_for_replacement_review"
    )
    assert "decision_against_xgboost_aft" not in report["models"]["xgboost_aft"]


@pytest.mark.parametrize(
    ("frame", "feature_names", "n_splits"),
    [
        (pd.DataFrame(), ["history_feature"], 3),
        (
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2025-01-01", periods=8, freq="h"),
                    "event_id": [1, 1, 2, 2, 3, 3, 4, 4],
                    "remaining_hours": np.arange(8, 0, -1),
                }
            ),
            [],
            3,
        ),
        (
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2025-01-01", periods=8, freq="h"),
                    "event_id": [1, 1, 2, 2, 3, 3, 4, 4],
                    "remaining_hours": np.arange(8, 0, -1),
                }
            ),
            ["missing_feature"],
            3,
        ),
    ],
)
def test_benchmark_history_models_validates_required_contract(frame, feature_names, n_splits):
    with pytest.raises(ValueError):
        benchmark_history_models(frame, feature_names, n_splits=n_splits)


def test_benchmark_history_models_requires_at_least_three_folds():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="h"),
            "event_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "remaining_hours": np.arange(8, 0, -1),
            "history_feature": np.arange(8, 0, -1),
        }
    )

    with pytest.raises(ValueError, match="at least 3"):
        benchmark_history_models(frame, ["history_feature"], n_splits=2)


@pytest.mark.parametrize("non_feature", ["timestamp", "event_id", "remaining_hours", "station_id"])
def test_benchmark_history_models_rejects_non_feature_columns(non_feature):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="h"),
            "event_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "remaining_hours": np.arange(8, 0, -1),
            "station_id": ["cd"] * 8,
            "history_feature": np.arange(8, 0, -1),
        }
    )

    with pytest.raises(ValueError, match="non-feature"):
        benchmark_history_models(frame, ["history_feature", non_feature], n_splits=3)


@pytest.mark.parametrize("invalid_target", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_benchmark_history_models_requires_positive_finite_labels(invalid_target):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="h"),
            "event_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "remaining_hours": np.arange(8, 0, -1, dtype=float),
            "history_feature": np.arange(8, 0, -1, dtype=float),
        }
    )
    frame.loc[0, "remaining_hours"] = invalid_target

    with pytest.raises(ValueError, match="positive and finite"):
        benchmark_history_models(frame, ["history_feature"], n_splits=3)


def test_benchmark_history_models_json_normalizes_timestamp_event_ids(monkeypatch):
    event_ids = pd.date_range("2025-02-01", periods=4, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="h"),
            "event_id": np.repeat(event_ids, 2),
            "remaining_hours": np.arange(8, 0, -1, dtype=float),
            "history_feature": np.arange(8, 0, -1, dtype=float),
        }
    )

    monkeypatch.setattr(
        "haze_forecast.model_benchmark.fit_predict_candidate",
        lambda model_name, train_x, train_y, valid_x, seed: valid_x["history_feature"].to_numpy() + 1,
    )

    report, predictions = benchmark_history_models(frame, ["history_feature"], n_splits=3)

    json.dumps(report)
    assert report["folds"][0]["train_event_ids"] == [event_ids[0].isoformat()]
    assert pd.api.types.is_datetime64_any_dtype(predictions["event_id"])


def test_benchmark_history_models_decision_uses_row_weighted_overall_mae(monkeypatch):
    event_sizes = [1, 1, 1, 100]
    event_ids = np.repeat([1, 2, 3, 4], event_sizes)
    fold_marker = np.repeat([10.0, 10.0, 10.0, 1.0], event_sizes)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(event_ids), freq="h"),
            "event_id": event_ids,
            "remaining_hours": np.full(len(event_ids), 100.0),
            "fold_marker": fold_marker,
        }
    )

    def fake_fit_predict(model_name, train_x, train_y, valid_x, seed):
        if model_name == "xgboost_aft":
            return 100.0 + valid_x["fold_marker"].to_numpy()
        return 100.0 + np.where(valid_x["fold_marker"].to_numpy() == 10.0, 9.0, 1.05)

    monkeypatch.setattr("haze_forecast.model_benchmark.fit_predict_candidate", fake_fit_predict)

    report, _ = benchmark_history_models(frame, ["fold_marker"], n_splits=3)

    decision = report["models"]["lightgbm_l1"]["decision_against_xgboost_aft"]
    assert decision["decision"] == "reject_unstable_or_insufficient_mae_improvement"
    assert decision["baseline_overall_mae"] == pytest.approx(120.0 / 102.0)
    assert decision["candidate_overall_mae"] == pytest.approx(123.0 / 102.0)


def test_write_benchmark_artifacts_writes_json_report_and_prediction_csv(tmp_path):
    report = {"protocol": "expanding validation", "models": {"xgboost_aft": {}}}
    predictions = pd.DataFrame(
        {
            "fold": [1, 1],
            "model": ["xgboost_aft", "xgboost_aft"],
            "event_id": [101, 101],
            "actual_remaining_hours": [4.0, 3.0],
            "predicted_remaining_hours": [4.5, 2.5],
        }
    )
    original_predictions = predictions.copy(deep=True)

    write_benchmark_artifacts(report, predictions, tmp_path / "history-benchmark")

    output_dir = tmp_path / "history-benchmark"
    assert json.loads((output_dir / "report.json").read_text(encoding="utf-8")) == report
    assert not (output_dir / "fold_predictions.csv.part").exists()
    pd.testing.assert_frame_equal(predictions, original_predictions)
    exported = pd.read_csv(output_dir / "fold_predictions.csv", encoding="utf-8-sig")
    assert exported.to_dict(orient="records") == predictions.to_dict(orient="records")


def test_write_benchmark_artifacts_restores_existing_files_when_csv_publish_fails(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "history-benchmark"
    output_dir.mkdir()
    report_path = output_dir / "report.json"
    predictions_path = output_dir / "fold_predictions.csv"
    report_path.write_text('{"version": "old"}', encoding="utf-8")
    predictions_path.write_text("old predictions\n", encoding="utf-8")
    predictions = pd.DataFrame(
        {
            "fold": [1],
            "model": ["xgboost_aft"],
            "event_id": [101],
            "actual_remaining_hours": [4.0],
            "predicted_remaining_hours": [4.5],
        }
    )
    original_replace = Path.replace

    def fail_csv_publication(path, target):
        if path == output_dir / "fold_predictions.csv.part" and Path(target) == predictions_path:
            raise OSError("simulated CSV publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_csv_publication)

    with pytest.raises(OSError, match="simulated CSV publication failure"):
        write_benchmark_artifacts({"version": "new"}, predictions, output_dir)

    assert report_path.read_text(encoding="utf-8") == '{"version": "old"}'
    assert predictions_path.read_text(encoding="utf-8") == "old predictions\n"
    assert not list(output_dir.glob("*.part*"))
    assert not list(output_dir.glob("*.backup"))


def test_benchmark_history_models_script_runs_the_benchmark_and_prints_json_summary(
    monkeypatch, capsys
):
    script_path = Path(__file__).parents[1] / "scripts" / "benchmark_history_models.py"
    spec = importlib.util.spec_from_file_location("benchmark_history_models_script", script_path)
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
    report = {"models": {"xgboost_aft": {}}}
    predictions = pd.DataFrame(
        {
            "fold": [1],
            "model": ["xgboost_aft"],
            "event_id": [1],
            "actual_remaining_hours": [1.0],
            "predicted_remaining_hours": [1.2],
        }
    )
    captured = {}
    monkeypatch.setattr(script.pd, "read_parquet", lambda path: frame)
    monkeypatch.setattr(script, "model_feature_names", lambda received: ["history_feature"])
    monkeypatch.setattr(
        script,
        "benchmark_history_models",
        lambda received, features, n_splits, seed: (report, predictions),
    )
    monkeypatch.setattr(
        script,
        "write_benchmark_artifacts",
        lambda received_report, received_predictions, output: captured.update(
            report=received_report, predictions=received_predictions, output=output
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--data",
            "fixture.parquet",
            "--output",
            "benchmark-output",
            "--splits",
            "4",
            "--seed",
            "7",
        ],
    )

    script.main()

    assert captured == {
        "report": report,
        "predictions": predictions,
        "output": Path("benchmark-output"),
    }
    assert json.loads(capsys.readouterr().out) == {
        "data": "fixture.parquet",
        "output_dir": "benchmark-output",
        "prediction_rows": 1,
        "models": ["xgboost_aft"],
    }


def test_benchmark_history_models_script_rejects_fewer_than_three_splits(monkeypatch):
    script_path = Path(__file__).parents[1] / "scripts" / "benchmark_history_models.py"
    spec = importlib.util.spec_from_file_location("benchmark_history_models_script_invalid", script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(
        script.pd,
        "read_parquet",
        lambda path: pytest.fail("invalid --splits must fail before reading parquet"),
    )
    monkeypatch.setattr(sys, "argv", [str(script_path), "--splits", "2"])

    with pytest.raises(SystemExit, match="2"):
        script.main()


def test_prefix_data_effect_keeps_validation_fixed_and_adds_only_historical_rows(monkeypatch):
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=8, freq="h"),
            "event_id": list(range(1, 9)),
            "remaining_hours": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "history_feature": np.arange(8, dtype=float),
        }
    )
    extension = pd.DataFrame(
        {
            "timestamp": pd.date_range("2019-01-01", periods=2, freq="h"),
            "event_id": ["cnemc_1", "cnemc_2"],
            "remaining_hours": [6.0, 5.0],
            "history_feature": [10.0, 11.0],
        }
    )
    calls = []

    def fake_predict(model_name, train_x, train_y, valid_x, seed):
        calls.append((model_name, len(train_x), len(valid_x), seed))
        return np.full(len(valid_x), float(train_y.median()))

    monkeypatch.setattr("haze_forecast.model_benchmark.fit_predict_candidate", fake_predict)

    report, predictions = benchmark_prefix_data_effect(
        base,
        extension,
        ["history_feature"],
        n_splits=3,
        seed=11,
    )

    assert report["protocol"] == "fixed future-event validation with optional historical prefix data"
    assert [fold["baseline_train_rows"] for fold in report["folds"]] == [2, 4, 6]
    assert [fold["candidate_train_rows"] for fold in report["folds"]] == [4, 6, 8]
    assert [fold["validation_rows"] for fold in report["folds"]] == [2, 2, 2]
    assert set(predictions["variant"]) == {"baseline", "with_historical_prefix"}
    assert all(call[0] == "xgboost_aft" for call in calls)
    assert [call[1] for call in calls] == [2, 4, 4, 6, 6, 8]


def test_prefix_data_effect_rejects_extension_that_reaches_a_validation_window():
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=8, freq="h"),
            "event_id": list(range(1, 9)),
            "remaining_hours": np.arange(8, 0, -1, dtype=float),
            "history_feature": np.arange(8, dtype=float),
        }
    )
    invalid_extension = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-01-06")],
            "event_id": ["cnemc_1"],
            "remaining_hours": [1.0],
            "history_feature": [1.0],
        }
    )

    with pytest.raises(ValueError, match="strictly precede every validation window"):
        benchmark_prefix_data_effect(base, invalid_extension, ["history_feature"])
