from collections.abc import Callable

import pandas as pd
import pytest

from haze_forecast.arl_benchmark import (
    ARL_MODEL_NAME,
    BASELINE_MODEL_NAME,
    build_benchmark_model,
    compare_feature_sets,
    decide_from_mae,
    summarise_per_event_direction,
)


class _SpyModel:
    """Constant predictor that records which events it was allowed to train on."""

    def __init__(self) -> None:
        self.trained_events: set[int] = set()

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "_SpyModel":
        self.trained_events = {int(value) for value in features["event_marker"]}
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(2.0, index=features.index)


def _spy_factory() -> _SpyModel:
    return _SpyModel()


def _frame(events: int = 3, rows_per_event: int = 3) -> pd.DataFrame:
    timestamps, event_ids, remaining, marker, noise, signal = [], [], [], [], [], []
    for event_index in range(events):
        for row_index in range(rows_per_event):
            timestamps.append(pd.Timestamp("2020-01-01T00:00:00Z", tz="UTC"))
            event_ids.append(100 + event_index)
            remaining.append(float(rows_per_event - row_index))
            marker.append(100 + event_index)
            noise.append(float((event_index * 3 + row_index) % 4))
            signal.append(float(rows_per_event - row_index))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "event_id": event_ids,
            "remaining_hours": remaining,
            "event_marker": marker,
            "observation_noise": noise,
            "arl_temperature_c_f003": signal,
        }
    )


def _recording_factory(seen: list[set[int]]) -> Callable[[], _SpyModel]:
    class Recording(_SpyModel):
        def fit(self, features, target):  # type: ignore[no-untyped-def]
            seen.append({int(value) for value in features["event_marker"]})
            return self

    return Recording


def test_training_folds_never_contain_the_held_out_event():
    frame = _frame(events=3)
    seen: list[set[int]] = []

    compare_feature_sets(
        frame, {BASELINE_MODEL_NAME: ["event_marker"]}, model_factory=_recording_factory(seen)
    )

    assert seen == [{101, 102}, {100, 102}, {100, 101}]
    for held_out, trained in zip(sorted(frame["event_id"].unique()), seen, strict=True):
        assert held_out not in trained


def test_compare_requires_the_baseline_feature_set_and_two_events():
    frame = _frame(events=2)
    with pytest.raises(ValueError, match="at least one feature set"):
        compare_feature_sets(frame, {}, model_factory=_spy_factory)

    single = frame.loc[frame["event_id"].eq(100)]
    with pytest.raises(ValueError, match="at least two events"):
        compare_feature_sets(single, {BASELINE_MODEL_NAME: ["event_marker"]}, model_factory=_spy_factory)


def test_compare_rejects_missing_feature_columns_and_empty_frames():
    frame = _frame(events=2)
    with pytest.raises(ValueError, match="missing columns"):
        compare_feature_sets(
            frame, {BASELINE_MODEL_NAME: ["not_a_column"]}, model_factory=_spy_factory
        )
    with pytest.raises(ValueError, match="empty"):
        compare_feature_sets(frame.iloc[0:0], {BASELINE_MODEL_NAME: ["event_marker"]}, model_factory=_spy_factory)


def test_compare_reports_both_models_and_a_weighted_overall_mae():
    frame = _frame(events=3)

    payload = compare_feature_sets(
        frame,
        {
            BASELINE_MODEL_NAME: ["observation_noise"],
            ARL_MODEL_NAME: ["observation_noise", "arl_temperature_c_f003"],
        },
        model_factory=lambda: build_benchmark_model(iterations=20, learning_rate=0.3),
    )

    assert payload["rows"] == 9
    assert payload["events"] == 3
    assert payload["largest_event_row_share"] == pytest.approx(1 / 3)
    assert payload["largest_event_id"] == 100
    assert {record["model"] for record in payload["overall"]} == {
        BASELINE_MODEL_NAME,
        ARL_MODEL_NAME,
    }
    assert len(payload["per_event"]) == 6
    assert payload["decision"] in {
        "reject_arl_for_main_model",
        "exploratory_only_insufficient_events",
        "candidate_for_more_data",
        "candidate_model",
    }


def test_compare_accepts_a_perfect_arl_signal_on_many_events():
    frame = _frame(events=15, rows_per_event=3)

    payload = compare_feature_sets(
        frame,
        {
            BASELINE_MODEL_NAME: ["observation_noise"],
            ARL_MODEL_NAME: ["observation_noise", "arl_temperature_c_f003"],
        },
        model_factory=lambda: build_benchmark_model(iterations=60, learning_rate=0.3),
    )

    assert payload["events"] == 15
    assert payload["mae_change_arl_minus_baseline"] < 0
    assert payload["decision"] == "candidate_model"


def test_decision_gate_follows_the_agreed_thresholds():
    assert decide_from_mae(20, 10.0, 10.0) == "reject_arl_for_main_model"
    assert decide_from_mae(20, 10.0, 10.5) == "reject_arl_for_main_model"
    assert decide_from_mae(5, 10.0, 9.0) == "exploratory_only_insufficient_events"
    assert decide_from_mae(10, 10.0, 9.9) == "candidate_for_more_data"
    assert decide_from_mae(10, 10.0, 9.0) == "candidate_for_more_data"
    assert decide_from_mae(20, 10.0, 9.0) == "candidate_model"
    assert decide_from_mae(20, 10.0, 9.6) == "candidate_for_more_data"


def test_per_event_direction_counts_improvements_and_regressions():
    per_event = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 2, 3, 3],
            "model": [BASELINE_MODEL_NAME, ARL_MODEL_NAME] * 3,
            "mae": [10.0, 9.0, 10.0, 11.0, 10.0, 10.0],
        }
    )

    assert summarise_per_event_direction(per_event) == {
        "improved": 1,
        "worsened": 1,
        "events": 3,
    }


def test_per_event_direction_ignores_baseline_only_runs():
    per_event = pd.DataFrame(
        {"event_id": [1], "model": [BASELINE_MODEL_NAME], "mae": [10.0]}
    )

    assert summarise_per_event_direction(per_event) == {}
