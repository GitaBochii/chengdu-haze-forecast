import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_head import RecoveryWindows, train


def test_cpu_training_selects_and_saves_duration_checkpoint(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    panel = {
        'timestamps': pd.date_range('2024-01-01', periods=30, freq='h', tz='Asia/Shanghai'),
        'inputs': np.random.default_rng(42).normal(size=(30, 3, 17)).astype(np.float32),
        'targets': np.ones((30, 3, 2), dtype=np.float32),
        'scaler': SimpleNamespace(mean=np.zeros(17), scale=np.ones(17)),
    }
    aqi = np.full(30, 80.)
    labels = pd.DataFrame({'event_id': np.ones(30), 'remaining_hours': np.ones(30)},
                          index=panel['timestamps'])
    training = RecoveryWindows(panel, np.array([3, 4, 5, 6]), 3, 6, aqi, np.ones(30, bool))
    validation = RecoveryWindows(panel, np.array([10, 11]), 3, 6, aqi, np.ones(30, bool))
    model = RecoveryPM25GNN(17, np.array([[104, 30], [104.1, 30.1], [104.2, 30.2]]),
                            np.zeros(2), np.ones(2), history=3, horizon=6)
    args = SimpleNamespace(learning_rate=.001, seed=42, batch_size=2, epochs=2,
                           auxiliary_weight=.1, horizon=6, patience=1)
    trace = train(model, training, validation, panel, labels, aqi, args, tmp_path, torch.device('cpu'))
    assert len(trace) == 2
    selection = json.loads((tmp_path / 'selection.json').read_text())
    assert selection['epoch'] == 1
    assert selection['rule'] == 'event_equal_mae_then_row_mae'
    state = torch.load(tmp_path / 'recovery_head_state.pt', weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key])
    assert training.audit()['informative_eligible'] == 4
