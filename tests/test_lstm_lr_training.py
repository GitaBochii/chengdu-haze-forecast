import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch


def training_module():
    assert importlib.util.find_spec("scripts.lstm_lr_training"), "learning-rate trainer missing"
    return importlib.import_module("scripts.lstm_lr_training")


def small_inputs():
    rng = np.random.default_rng(13)
    history = rng.normal(size=(8, 24, 3)).astype(np.float32)
    snapshot = rng.normal(size=(8, 5)).astype(np.float32)
    lower = np.linspace(8.0, 22.0, 8)
    upper = lower.copy()
    return history, snapshot, lower, upper


def test_learning_rate_must_be_finite_and_positive():
    module = training_module()
    history, snapshot, lower, upper = small_inputs()
    for value in (0.0, -0.001, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="learning_rate"):
            module.fit_lstm(24, history, snapshot, lower, upper, 1, value)


def test_epoch_count_and_training_shapes_are_validated_before_cuda():
    module = training_module()
    history, snapshot, lower, upper = small_inputs()
    for epochs in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="epochs"):
            module.fit_lstm(24, history, snapshot, lower, upper, epochs, 0.001)
    with pytest.raises(ValueError, match="aligned"):
        module.fit_lstm(24, history, snapshot, lower[:-1], upper, 1, 0.001)
    with pytest.raises(ValueError, match="shape"):
        module.fit_lstm(24, history[:, 0], snapshot, lower, upper, 1, 0.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LSTM fitting requires CUDA")
def test_adamw_uses_requested_learning_rate_and_frozen_training_constants(monkeypatch):
    module = training_module()
    history, snapshot, lower, upper = small_inputs()
    calls = []
    original_adamw = torch.optim.AdamW

    def observe_adamw(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_adamw(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", observe_adamw)
    module.fit_lstm(24, history, snapshot, lower, upper, 1, 0.003)
    assert calls == [{"lr": 0.003, "weight_decay": 0.01}]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LSTM fitting requires CUDA")
def test_original_learning_rate_replays_frozen_fit_exactly():
    frozen = importlib.import_module("scripts.benchmark_lstm_direct_crossyear")
    module = training_module()
    history, snapshot, lower, upper = small_inputs()
    expected, expected_trace, expected_epoch = frozen.fit(
        "lstm_history", 42, history, snapshot, lower, upper, 2
    )
    actual, actual_trace, actual_epoch = module.fit_lstm(
        42, history, snapshot, lower, upper, 2, 0.001
    )
    assert actual_trace == expected_trace
    assert actual_epoch == expected_epoch == 0
    for name, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="LSTM fitting requires CUDA")
def test_validation_restores_earliest_best_checkpoint(monkeypatch):
    module = training_module()
    history, snapshot, lower, upper = small_inputs()
    times = pd.date_range("2023-01-01", periods=2, freq="h", tz="Asia/Shanghai")
    validation_frame = pd.DataFrame(
        {
            "timestamp": times,
            "continuity_block": ["process-a", "process-b"],
            "lower_bound_hours": [12.0, 18.0],
            "upper_bound_hours": [12.0, 18.0],
            "exact_hours": [12.0, 18.0],
        }
    )
    scores = iter((4.0, 4.0, 5.0))
    monkeypatch.setattr(
        module, "score_predictions", lambda frame, prediction: {"exact_block_mae": next(scores)}
    )
    validation = (history[:2], snapshot[:2], validation_frame)
    actual, trace, best_epoch = module.fit_lstm(
        24, history, snapshot, lower, upper, 3, 0.001, validation=validation
    )
    reference, _, _ = module.fit_lstm(24, history, snapshot, lower, upper, 1, 0.001)
    assert best_epoch == 1
    assert [row["selection_block_mae"] for row in trace] == [4.0, 4.0, 5.0]
    for name, value in actual.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name
