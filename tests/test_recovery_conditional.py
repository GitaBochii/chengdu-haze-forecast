import numpy as np
import pytest
import torch

from haze_forecast.recovery_conditional import StateAgeRecoveryPM25GNN, StateOnlyRecoveryPM25GNN
from haze_forecast.regional_recovery_network import RecoveryPM25GNN


def args():
    return (5, np.array([[104., 30.], [104.1, 30.], [104., 30.1]]),
            np.array([10., 270.]), np.ones(2))


def make(cls):
    torch.manual_seed(42)
    return cls(*args(), history=4, horizon=8)


def inputs():
    x = torch.randn(2, 4, 3, 5)
    age = torch.full((2, 4, 3, 1), 1.5)
    return x, torch.cat([x, age], -1), torch.ones(2, 3, 2, dtype=torch.bool)


def golden_forward(network, x, active):
    # Frozen pre-refactor computation, deliberately independent of the helper.
    batch, _, nodes, _ = x.shape
    predictions, pooled = [], []
    for target, (index, model) in enumerate(zip(network.target_indices, network.heads, strict=True)):
        model.graph_gnn.device = x.device
        hidden = x.new_zeros(batch * nodes, model.hid_dim)

        def step(values, state, model=model):
            spatial = model.graph_gnn(values.contiguous())
            return model.gru_cell(torch.cat([spatial, values], dim=-1), state)

        for hour in range(network.history - 1):
            observed = torch.cat([x[:, hour, :, index:index + 1], x[:, hour]], dim=-1)
            hidden = step(observed, hidden)
        current, decoded = x[:, -1, :, index:index + 1], []
        for lead in range(network.horizon):
            hidden = step(torch.cat([current, x[:, -1]], dim=-1), hidden)
            encoded = hidden.view(batch, nodes, model.hid_dim)
            if lead == 0:
                mask = active[:, :, target].unsqueeze(-1)
                pooled.append((encoded * mask).sum(dim=1) / mask.sum(dim=1))
            current = model.fc_out(encoded)
            decoded.append(current)
        predictions.append(torch.stack(decoded, dim=1))
    state = torch.cat(pooled, dim=-1)
    return torch.cat(predictions, dim=-1), state, network.risk_head(state)


def test_helper_matches_frozen_forward_exactly():
    base = make(RecoveryPM25GNN)
    x, _, active = inputs()
    active[0, 1, 0] = False
    names_before = list(base.state_dict())
    curve, state, logits = golden_forward(base, x, active)
    actual_curve, actual_state = base.forward_with_state(x, active)
    torch.testing.assert_close(actual_curve, curve, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, state, rtol=0, atol=0)
    torch.testing.assert_close(base(x, active)[1], logits, rtol=0, atol=0)
    assert state.shape == (2, 128)
    assert list(base.state_dict()) == names_before


def test_same_seed_full_initial_states_and_zero_branch_match_base():
    base, no_age, age = (make(cls) for cls in (RecoveryPM25GNN, StateOnlyRecoveryPM25GNN, StateAgeRecoveryPM25GNN))
    assert list(no_age.state_dict()) == list(age.state_dict())
    for key, value in no_age.state_dict().items():
        torch.testing.assert_close(value, age.state_dict()[key], rtol=0, atol=0)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, age.state_dict()[key], rtol=0, atol=0)
    assert age.conditional_head[0].weight.shape == (16, 129)
    assert age.conditional_head[2].weight.shape == (3, 16)
    assert torch.count_nonzero(age.conditional_head[2].weight) == 0
    assert torch.count_nonzero(age.conditional_head[2].bias) == 0
    x, augmented, active = inputs()
    for model in (no_age, age):
        actual = model(augmented, active)
        expected = base(x, active)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_state_only_remains_age_invariant_with_nonzero_trained_branch():
    model = make(StateOnlyRecoveryPM25GNN)
    _, x, active = inputs()
    loss = model(x, active)[1].square().sum()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-.01)
    assert model.conditional_head[2].weight.abs().sum() > 0
    changed = x.clone()
    changed[..., -1] = 20
    for a, b in zip(model(x, active), model(changed, active), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_nonzero_branch_can_express_state_age_interaction():
    model = make(StateAgeRecoveryPM25GNN)
    with torch.no_grad():
        for p in model.conditional_head.parameters():
            p.zero_()
        model.conditional_head[0].weight[0, 0] = 1
        model.conditional_head[0].weight[0, -1] = 1
        model.conditional_head[2].weight[0, 0] = 1
    values = torch.zeros(4, 129)
    values[:, 0] = torch.tensor([0., 0., 1., 1.])
    values[:, -1] = torch.tensor([0., 1., 0., 1.])
    outputs = model.conditional_head(values)[:, 0]
    assert not torch.isclose(outputs[1] - outputs[0], outputs[3] - outputs[2])


def test_second_backward_reaches_first_layer():
    model = make(StateAgeRecoveryPM25GNN)
    _, x, active = inputs()
    model(x, active)[1].sum().backward()
    first, last = model.conditional_head[0], model.conditional_head[2]
    assert first.weight.grad.abs().sum() == 0
    assert last.weight.grad.abs().sum() > 0
    with torch.no_grad():
        last.weight.add_(last.weight.grad, alpha=-.01)
    model.zero_grad(set_to_none=True)
    model(x, active)[1].sum().backward()
    assert torch.isfinite(first.weight.grad).all()
    assert first.weight.grad.abs().sum() > 0
    assert first.weight.grad[:, -1].abs().sum() > 0


@pytest.mark.parametrize('cls', [StateOnlyRecoveryPM25GNN, StateAgeRecoveryPM25GNN])
def test_invalid_age_and_shapes_rejected(cls):
    model = make(cls)
    base, x, active = inputs()
    with pytest.raises(ValueError, match='input'):
        model(base, active)
    bad = x.clone()
    bad[0, -1, 1, -1] = 9
    with pytest.raises(ValueError, match='nodes'):
        model(bad, active)
    bad[..., -1] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        model(bad, active)
