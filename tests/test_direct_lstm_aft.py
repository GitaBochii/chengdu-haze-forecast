import math

import torch
from torch import nn

from haze_forecast.direct_lstm_aft import DirectLSTMAFT
from haze_forecast.torch_interval_aft import interval_aft_nll


def test_model_uses_real_unidirectional_lstm_and_returns_batch_predictions():
    torch.manual_seed(42)
    model = DirectLSTMAFT(history_dim=24, snapshot_dim=208)
    history = torch.randn(3, 24, 24)
    snapshot = torch.randn(3, 208)

    prediction = model(history, snapshot)

    assert isinstance(model.history, nn.LSTM)
    assert model.history.num_layers == 1
    assert model.history.bidirectional is False
    assert prediction.shape == (3,)
    torch.testing.assert_close(prediction[:1], model(history[:1], snapshot[:1]))


def test_model_parameter_count_and_initial_duration_bias():
    model = DirectLSTMAFT(history_dim=24, snapshot_dim=208)

    assert sum(parameter.numel() for parameter in model.parameters()) == 10177
    torch.testing.assert_close(
        model.output[-1].bias.detach(), torch.tensor([math.log(24)])
    )


def test_interval_likelihood_backpropagates_through_history_and_snapshot():
    torch.manual_seed(42)
    model = DirectLSTMAFT(history_dim=24, snapshot_dim=208)
    history = torch.randn(3, 24, 24, requires_grad=True)
    snapshot = torch.randn(3, 208, requires_grad=True)
    mu = model(history, snapshot)
    lower = torch.tensor([12., 18., 24.])
    upper = torch.tensor([12., 30., float('inf')])

    interval_aft_nll(mu, lower, upper, scale=1.2).mean().backward()

    assert history.grad is not None and history.grad.abs().sum() > 0
    assert snapshot.grad is not None and snapshot.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_state_dict_restore_reproduces_predictions():
    torch.manual_seed(42)
    model = DirectLSTMAFT(history_dim=24, snapshot_dim=208).eval()
    history, snapshot = torch.randn(2, 24, 24), torch.randn(2, 208)
    expected = model(history, snapshot)
    restored = DirectLSTMAFT(history_dim=24, snapshot_dim=208).eval()
    restored.load_state_dict(model.state_dict())

    torch.testing.assert_close(restored(history, snapshot), expected)
