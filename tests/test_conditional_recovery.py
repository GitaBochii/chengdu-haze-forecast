import pytest
import torch

from haze_forecast.conditional_recovery import conditional_log_probability, observed_prefix_nll
from haze_forecast.recovery_survival import hazard_distribution, masked_survival_loss


def test_zero_matches_base_and_normalizes():
    torch.manual_seed(2)
    logits = torch.randn(8, 67, dtype=torch.float64) - 3
    logp = conditional_log_probability(logits, torch.zeros_like(logits))
    torch.testing.assert_close(logp.exp(), hazard_distribution(logits))
    torch.testing.assert_close(logp.exp().sum(1), torch.ones(8, dtype=torch.float64))


def test_tail_unchanged_and_no_gradient():
    logits = torch.full((3, 67), -4.)
    correction = torch.randn_like(logits, requires_grad=True)
    logp = conditional_log_probability(logits, correction)
    reference = conditional_log_probability(logits, torch.zeros_like(logits))
    assert torch.equal(logp[:, -1], reference[:, -1])
    logp[:, -1].sum().backward()
    assert torch.count_nonzero(correction.grad) == 0


def test_full_censored_loss_constant_gradient_zero():
    logits = torch.full((2, 67), -3.)
    correction = torch.randn_like(logits, requires_grad=True)
    logp = conditional_log_probability(logits, correction)
    loss = observed_prefix_nll(logp, torch.zeros_like(logits), torch.ones_like(logits, dtype=torch.bool),
                               torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(loss, -logp[:, -1].mean())
    loss.backward()
    assert torch.count_nonzero(correction.grad) == 0


def test_exact_and_partial_likelihood():
    logits = torch.zeros(2, 67, dtype=torch.float64)
    logp = conditional_log_probability(logits, logits)
    target = torch.zeros_like(logits)
    target[0, 2] = 1
    risk = torch.zeros_like(logits, dtype=torch.bool)
    risk[:, :3] = True
    loss = observed_prefix_nll(logp, target, risk, torch.ones(2, dtype=torch.bool))
    expected = (-logp[0, 2] - torch.logsumexp(logp[1, 3:], 0)) / 2
    torch.testing.assert_close(loss, expected)


def test_no_observed_or_eligible_returns_none():
    x = torch.zeros(2, 67)
    logp = conditional_log_probability(x, x)
    assert observed_prefix_nll(logp, x, x.bool(), torch.ones(2, dtype=torch.bool)) is None
    assert observed_prefix_nll(logp, x, ~x.bool(), torch.zeros(2, dtype=torch.bool)) is None


@pytest.mark.parametrize('value', [-1000., 1000.])
def test_extreme_logits_are_finite(value):
    x = torch.full((2, 67), value)
    result = conditional_log_probability(x, torch.zeros_like(x))
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.exp().sum(1), torch.ones(2))


def test_invalid_prefix_rejected():
    x = torch.zeros(2, 67)
    mask = x.bool()
    mask[0, 2] = True
    with pytest.raises(ValueError):
        observed_prefix_nll(conditional_log_probability(x, x), x, mask, torch.ones(2, dtype=torch.bool))


def test_zero_correction_likelihood_matches_hazard_nll():
    torch.manual_seed(42)
    logits = torch.randn(4, 67, dtype=torch.float64) - 3
    target = torch.zeros_like(logits)
    target[0, 4] = 1
    risk = torch.zeros_like(logits, dtype=torch.bool)
    risk[0, :5] = True
    risk[1, :13] = True
    risk[2, :] = True
    eligible = torch.ones(4, dtype=torch.bool)
    actual = observed_prefix_nll(conditional_log_probability(logits, torch.zeros_like(logits)),
                                 target, risk, eligible)
    expected = masked_survival_loss(logits, target, risk, eligible)
    torch.testing.assert_close(actual, expected)


def test_partial_censor_has_nonzero_gradient():
    logits = torch.full((1, 67), -4., dtype=torch.float64)
    correction = torch.zeros_like(logits, requires_grad=True)
    risk = torch.zeros_like(logits, dtype=torch.bool)
    risk[:, :10] = True
    loss = observed_prefix_nll(conditional_log_probability(logits, correction),
                               torch.zeros_like(logits), risk, torch.ones(1, dtype=torch.bool))
    loss.backward()
    assert (correction.grad[:, :10] > 0).all()
    assert (correction.grad[:, 10:] < 0).all()


@pytest.mark.parametrize('kind', ['outside_risk', 'two_events', 'event_not_final', 'nonbinary'])
def test_invalid_targets_rejected(kind):
    x = torch.zeros(1, 67)
    risk = torch.zeros_like(x, dtype=torch.bool)
    risk[:, :5] = True
    target = torch.zeros_like(x)
    if kind == 'outside_risk':
        target[0, 6] = 1
    elif kind == 'two_events':
        target[0, 3:5] = 1
    elif kind == 'event_not_final':
        target[0, 1] = 1
    else:
        target[0, 4] = .5
    with pytest.raises(ValueError):
        observed_prefix_nll(conditional_log_probability(x, x), target, risk, torch.ones(1, dtype=torch.bool))
