import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from haze_forecast.regional_air_networks import PM25GNNHistoryAdapter
from haze_forecast.regional_recovery_network import RecoveryPM25GNN


def model_args():
    return (5, np.array([[104., 30.], [104.1, 30.], [104., 30.1]]),
            np.array([10., 270.]), np.ones(2))


def make_model(**kwargs):
    torch.manual_seed(42)
    return RecoveryPM25GNN(*model_args(), history=4, horizon=8, **kwargs)


def test_shared_weights_preserve_concentration_exactly_and_initialize_hazard():
    baseline = PM25GNNHistoryAdapter(*model_args(), history=4, horizon=8)
    model = make_model()
    incompatible = model.load_state_dict(baseline.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {"risk_head.weight", "risk_head.bias"}
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(model.risk_head.bias,
                               torch.full((3,), math.log(1 / 23)))
    for batch in (2, 1):
        x = torch.randn(batch, 4, 3, 5)
        concentration, logits = model(x, torch.ones(batch, 3, 2, dtype=torch.bool))
        assert concentration.shape == (batch, 8, 3, 2)
        assert logits.shape == (batch, 3)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(concentration, baseline(x), rtol=0, atol=0)


def test_risk_head_uses_early_history_and_backpropagates_to_both_encoders():
    model = make_model()
    x = torch.randn(2, 4, 3, 5, requires_grad=True)
    active = torch.ones(2, 3, 2, dtype=torch.bool)
    _, logits = model(x, active)
    changed = x.detach().clone()
    changed[:, :-1] += 3
    assert not torch.allclose(logits, model(changed, active)[1])
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits)).backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :-1].abs().sum() > 0
    for head in model.heads:
        for module in (head.graph_gnn, head.gru_cell):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads) > 0


def test_pooling_uses_target_specific_active_nodes_at_last_observation():
    model = make_model()
    states = [[], []]
    handles = []
    for index, head in enumerate(model.heads):
        handles.append(head.gru_cell.register_forward_hook(
            lambda module, args, output, index=index: states[index].append(output)))
    x = torch.randn(2, 4, 3, 5)
    active = torch.tensor([[[1, 0], [0, 1], [0, 1]],
                           [[0, 1], [1, 0], [0, 0]]], dtype=torch.bool)
    concentration, logits = model(x, active)
    for handle in handles:
        handle.remove()
    pooled = []
    for index, head in enumerate(model.heads):
        encoded = states[index][3].reshape(2, 3, head.hid_dim)
        mask = active[:, :, index].unsqueeze(-1)
        pooled.append((encoded * mask).sum(1) / mask.sum(1))
    torch.testing.assert_close(logits, model.risk_head(torch.cat(pooled, dim=-1)),
                               rtol=0, atol=0)
    all_concentration, all_logits = model(x, torch.ones_like(active))
    torch.testing.assert_close(concentration, all_concentration, rtol=0, atol=0)
    assert not torch.allclose(logits, all_logits)


@pytest.mark.parametrize("shape", [(2, 3, 3, 5), (2, 4, 2, 5), (2, 4, 3, 4), (4, 3, 5)])
def test_invalid_input_shape_rejected(shape):
    with pytest.raises(ValueError, match="input"):
        make_model()(torch.zeros(shape), torch.ones(2, 3, 2, dtype=torch.bool))


@pytest.mark.parametrize("active", [torch.ones(2, 3, 1, dtype=torch.bool),
                                    torch.ones(2, 3, 2),
                                    torch.zeros(2, 3, 2, dtype=torch.bool)])
def test_invalid_activity_rejected(active):
    with pytest.raises(ValueError, match="active"):
        make_model()(torch.zeros(2, 4, 3, 5), active)


def test_single_target_without_active_nodes_rejected():
    active = torch.ones(2, 3, 2, dtype=torch.bool)
    active[1, :, 0] = False
    with pytest.raises(ValueError, match="active"):
        make_model()(torch.zeros(2, 4, 3, 5), active)


@pytest.mark.parametrize("history,horizon", [(0, 8), (4, 5)])
def test_insufficient_history_or_horizon_rejected(history, horizon):
    with pytest.raises(ValueError):
        RecoveryPM25GNN(*model_args(), history=history, horizon=horizon)


@pytest.mark.parametrize("indices", [(0,), (0, 5), (-1, 2)])
def test_invalid_target_indices_rejected(indices):
    with pytest.raises(ValueError, match="target"):
        make_model(target_indices=indices)


def test_single_observation_and_full_horizon_contract():
    model = RecoveryPM25GNN(*model_args(), history=1, horizon=72)
    concentration, logits = model(torch.zeros(1, 1, 3, 5),
                                  torch.ones(1, 3, 2, dtype=torch.bool))
    assert concentration.shape == (1, 72, 3, 2)
    assert logits.shape == (1, 67)
    assert sum(p.numel() for p in model.risk_head.parameters()) == 8643
