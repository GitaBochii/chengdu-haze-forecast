import pytest
import torch

from scripts.audit_recovery_gradients import gradient_metrics


def test_opposing_shared_gradients_and_global_clipping():
    params = [torch.zeros(2), torch.zeros(1), torch.zeros(1)]
    survival = (torch.tensor([3., 4.]), torch.tensor([2.]), None)
    auxiliary = (torch.tensor([-6., -8.]), None, torch.tensor([4.]))
    result = gradient_metrics(params, survival, auxiliary, 0.1)
    assert result['shared_cosine'] == pytest.approx(-1)
    assert result['weighted_aux_shared_ratio'] == pytest.approx(0.2)
    assert result['combined_global_norm'] == pytest.approx((16 + 4 + .16) ** .5)
    assert result['clip_multiplier'] == pytest.approx(1 / ((20.16 ** .5) + 1e-6))
    assert result['shared_parameter_elements'] == 2


def test_zero_gradient_cosine_is_undefined():
    result = gradient_metrics([torch.zeros(1)], (torch.zeros(1),), (torch.ones(1),), .1)
    assert result['shared_cosine'] is None
    assert result['weighted_aux_shared_ratio'] is None
    assert result['clip_multiplier'] == 1


def test_combined_norm_matches_backward_and_torch_clipping():
    shared = torch.nn.Parameter(torch.tensor([2., -3.]))
    head = torch.nn.Parameter(torch.tensor([1.]))
    params = (shared, head)
    survival = shared.square().sum() + head.square().sum()
    auxiliary = (shared - 1).square().sum()
    gs = torch.autograd.grad(survival, params, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(auxiliary, params, retain_graph=True, allow_unused=True)
    result = gradient_metrics(params, gs, ga, .1)
    (survival + .1 * auxiliary).backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
    assert result['combined_global_norm'] == pytest.approx(float(norm))
    after = torch.cat([p.grad.flatten() for p in params]).norm()
    assert float(after) == pytest.approx(float(norm) * result['clip_multiplier'])
