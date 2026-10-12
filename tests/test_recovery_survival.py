import numpy as np
import pandas as pd
import pytest
import torch

from haze_forecast.recovery_survival import (
    causal_recovery_eligibility,
    hazard_distribution,
    hazard_median,
    masked_survival_loss,
    recovery_hazard_labels,
    validate_hourly_grid,
)


def test_exact_start_and_confirmation_and_late_start():
    future = np.full((3, 72), 150.0)
    future[0, 2:8] = 80
    future[1, :6] = 100
    future[2, -6:] = 90
    target, mask = recovery_hazard_labels(future)
    assert target.shape == (3, 67)
    assert np.where(target[0])[0].tolist() == [2]
    assert mask.sum(axis=1).tolist() == [3, 1, 67]
    assert target[1, 0] == target[2, -1] == 1


def test_missing_future_is_censored_not_a_negative_or_late_event():
    future = np.full((4, 72), 150.0)
    future[0, 9] = np.nan
    future[0, 10:16] = 80
    future[1, :6] = 80
    future[1, 6] = np.nan
    future[2, 0] = np.nan
    target, mask = recovery_hazard_labels(future)
    assert mask.sum(axis=1).tolist() == [4, 1, 0, 67]
    assert target.sum(axis=1).tolist() == [0, 1, 0, 0]
    assert np.array_equal(mask, recovery_hazard_labels(future.copy())[1])


def test_five_clean_hours_are_not_a_recovery_and_invalid_horizon_rejected():
    target, mask = recovery_hazard_labels(np.array([[150, 80, 80, 80, 80, 80]]))
    assert target.tolist() == [[0]] and mask.tolist() == [[True]]
    with pytest.raises(ValueError):
        recovery_hazard_labels(np.zeros((1, 5)))


def test_nll_excludes_unknown_and_ineligible_rows_with_finite_gradients():
    logits = torch.zeros(3, 4, requires_grad=True)
    targets = torch.tensor([[0., 1, 0, 0], [0., 0, 0, 0], [1., 0, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    eligible = torch.tensor([True, True, False])
    loss = masked_survival_loss(logits, targets, mask, eligible)
    assert loss.item() == pytest.approx(2 * np.log(2))
    loss.backward()
    assert logits.grad[0, :2].abs().sum() > 0
    assert logits.grad[0, 2:].abs().sum() == 0
    assert logits.grad[1:].abs().sum() == 0
    assert masked_survival_loss(logits, targets, mask, ~torch.ones(3, dtype=torch.bool)) is None


def test_distribution_normalizes_including_tail_and_median_handles_extremes():
    logits = torch.tensor([[-100.] * 67, [100.] * 67, [0.] * 67])
    probability = hazard_distribution(logits)
    assert probability.shape == (3, 68)
    torch.testing.assert_close(probability.sum(dim=1), torch.ones(3))
    assert hazard_median(logits).tolist() == [68, 1, 1]
    assert torch.isfinite(probability).all()


def test_eligibility_depends_on_past_not_future_completion():
    aqi = np.array([150.] * 3 + [80.] * 6 + [150.] * 3)
    particle = aqi > 100
    eligible = causal_recovery_eligibility(aqi, particle)
    assert eligible.tolist() == [True] * 8 + [False] + [True] * 3
    np.testing.assert_array_equal(eligible[:5], causal_recovery_eligibility(aqi[:5], particle[:5]))
    assert causal_recovery_eligibility(np.full(20, 160.), np.ones(20, bool)).all()


def test_grid_rejects_missing_or_duplicate_hours():
    times = pd.date_range('2024-01-01', periods=10, freq='h', tz='Asia/Shanghai')
    validate_hourly_grid(times)
    with pytest.raises(ValueError):
        validate_hourly_grid(times.delete(3))
    with pytest.raises(ValueError):
        validate_hourly_grid(times.append(times[-1:]))
