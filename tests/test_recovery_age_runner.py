from types import SimpleNamespace

import numpy as np
import torch

from haze_forecast.recovery_age import AgeRecoveryPM25GNN
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_age import AgeWindows
from scripts.benchmark_recovery_head import RecoveryWindows


def test_age_channel_uses_origin_only_preserving_all_base_tensors():
    rng = np.random.default_rng(42)
    panel = {'inputs': rng.normal(size=(100, 2, 5)).astype(np.float32),
             'targets': np.full((100, 2, 2), 150., dtype=np.float32),
             'scaler': SimpleNamespace(mean=np.zeros(5), scale=np.ones(5)),
             'recovery_age': np.arange(100, dtype=np.float32)}
    origins = np.array([24])
    aqi, eligible = np.full(100, 150.), np.ones(100, dtype=bool)
    baseline = RecoveryWindows(panel, origins, 24, 72, aqi, eligible)[0]
    candidate = AgeWindows(panel, origins, 24, 72, aqi, eligible)[0]
    assert candidate[0].shape == (24, 2, 6)
    torch.testing.assert_close(candidate[0][..., :-1], baseline[0], rtol=0, atol=0)
    assert torch.all(candidate[0][..., -1] == 24)
    for a, b in zip(candidate[1:], baseline[1:], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    panel['recovery_age'][25:] = 999
    again = AgeWindows(panel, origins, 24, 72, aqi, eligible)[0]
    torch.testing.assert_close(candidate[0], again[0], rtol=0, atol=0)


def test_same_seed_preserves_all_original_initial_weights():
    args = (5, np.array([[104., 30.], [104.1, 30.]]), np.zeros(2), np.ones(2))
    torch.manual_seed(42)
    base = RecoveryPM25GNN(*args)
    torch.manual_seed(42)
    model = AgeRecoveryPM25GNN(*args)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert torch.count_nonzero(model.age_projection.weight) == 0
