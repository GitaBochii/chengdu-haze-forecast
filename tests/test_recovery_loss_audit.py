import numpy as np
import pytest
import torch

from haze_forecast.recovery_survival import masked_survival_loss
from scripts.audit_recovery_loss import contribution_arrays


def test_contributions_match_loss_and_autograd():
    logits = torch.tensor([[0., -1., 2.], [1., 0., -2.], [4., 3., 2.]], requires_grad=True)
    target = torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
    mask = torch.tensor([[True, True, False], [True, True, True], [True, False, False]])
    eligible = torch.tensor([True, True, False])
    loss = masked_survival_loss(logits, target, mask, eligible)
    loss.backward()
    result = contribution_arrays(logits.detach().numpy(), target.numpy(), mask.numpy(), eligible.numpy())
    assert result['nll'].sum() / 2 == pytest.approx(loss.item())
    np.testing.assert_allclose(result['gradient'] / 2, logits.grad.numpy(), atol=1e-7)
    assert result['nll'][2] == 0
    assert result['gradient'][2].sum() == 0


def test_empty_observation_has_no_contribution():
    result = contribution_arrays(np.zeros((1, 2)), np.zeros((1, 2)),
                                 np.zeros((1, 2), dtype=bool), np.ones(1, dtype=bool))
    assert result['nll'].sum() == 0
    assert result['gradient'].sum() == 0
