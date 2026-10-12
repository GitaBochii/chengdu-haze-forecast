import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

path = Path(__file__).resolve().parents[1] / "scripts/benchmark_regional_air_models.py"
spec = importlib.util.spec_from_file_location("regional_runner_test", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("missing", [False, True])
def test_evaluation_ignores_offline_prediction_and_excludes_unverified_labels(tmp_path, missing):
    times = pd.date_range("2025-01-01", periods=8, freq="h", tz="Asia/Shanghai")
    raw = np.full((8, 2, 2), np.nan, dtype=np.float32)
    raw[:, 0] = [10, 80]
    # The offline station coming online in the future must not affect origin support.
    raw[5:, 1] = [100, 300]
    prediction = np.full((1, 6, 2, 2), 999.)
    prediction[:, :, 0] = [10, 80]
    city = pd.DataFrame({"pm2_5": 10., "aqi": 80.}, index=times)
    if missing:
        city.loc[times[3], "aqi"] = np.nan
    labels = pd.DataFrame({"event_id": 1, "remaining_hours": 1.}, index=times)
    result = runner.evaluate(prediction, {"timestamps": times, "targets": raw}, np.array([0]),
                             city, labels, SimpleNamespace(history=2, horizon=6),
                             tmp_path, "test", 2025)
    assert result["city_metrics"] == {"pm2_5": 0., "aqi": 0.}
    assert result["duration"]["event_rows"] == (0 if missing else 1)
    assert result["unverified_labeled_origins_excluded"] == int(missing)


def test_window_inputs_stop_at_origin_and_labels_remain_masked():
    inputs = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    targets = np.repeat(inputs, 2, axis=2)
    targets[4, 0, 0] = np.nan
    scaler = SimpleNamespace(mean=np.zeros(3), scale=np.ones(3))
    dataset = runner.Windows({"inputs": inputs, "targets": targets, "scaler": scaler},
                             np.array([3]), history=3, horizon=2)
    x, y, mask = dataset[0]
    np.testing.assert_array_equal(x[:, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(y[:, 0, 1], [4, 5])
    assert not mask[0, 0, 0]
    assert y[0, 0, 0] == 0


def test_masked_training_loss_does_not_learn_missing_targets():
    prediction = torch.tensor([2., 100.], requires_grad=True)
    loss = runner.masked_loss(prediction, torch.tensor([3., 0.]), torch.tensor([True, False]))
    assert loss.item() == 1
    loss.backward()
    assert prediction.grad.tolist() == [-1., 0.]


def test_paired_initialization_does_not_depend_on_prior_rng_or_model_order():
    panel = {"feature_names": list(range(5)), "coords": np.array([[104., 30.], [104.1, 30.]]),
             "scaler": SimpleNamespace(mean=np.array([0., 0., 0., 10., 270.]), scale=np.ones(5))}
    args = SimpleNamespace(history=4, horizon=6, seed=42)
    first = runner.build_model("pm25_gnn", panel, args)
    torch.rand(100)
    np.random.rand(100)
    second = runner.build_model("pm25_gnn_history", panel, args)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    third = runner.build_model("pm25_gnn", panel, args)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, third.state_dict()[name], rtol=0, atol=0)
