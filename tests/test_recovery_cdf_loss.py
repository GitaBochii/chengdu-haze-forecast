import pytest
import torch

from haze_forecast.recovery_cdf_loss import observed_cdf_loss


def arrays(batch=1):
    return (torch.zeros(batch, 67, dtype=torch.float64, requires_grad=True),
            torch.zeros(batch, 67, dtype=torch.float64),
            torch.zeros(batch, 67, dtype=torch.bool), torch.ones(batch, dtype=torch.bool))


def test_exact_event_observes_all_cdf_times_handcomputed():
    logits, target, risk, eligible = arrays()
    target[0, 1] = 1
    risk[0, :2] = True
    expected_cdf = 1 - .5 ** torch.arange(1, 68, dtype=torch.float64)
    labels = torch.ones(67, dtype=torch.float64)
    labels[0] = 0
    expected = ((expected_cdf - labels) ** 2).mean()
    actual = observed_cdf_loss(logits, target, risk, eligible)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-15)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2:].abs().sum() > 0


def test_full_censor_uses_all_67_observed_zero_labels():
    logits, target, risk, eligible = arrays()
    risk[:] = True
    expected = ((1 - .5 ** torch.arange(1, 68, dtype=torch.float64)) ** 2).mean()
    torch.testing.assert_close(observed_cdf_loss(logits, target, risk, eligible), expected)


def test_partial_censor_ignores_unobserved_later_cdf_and_gradient():
    logits, target, risk, eligible = arrays()
    risk[0, :3] = True
    expected = torch.tensor((.5 ** 2 + .75 ** 2 + .875 ** 2) / 3, dtype=torch.float64)
    loss = observed_cdf_loss(logits, target, risk, eligible)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(logits.grad[0, 3:], torch.zeros(64, dtype=torch.float64), rtol=0, atol=0)
    assert logits.grad[0, :3].abs().sum() > 0
    changed = torch.cat([logits.detach()[:, :3], torch.full((1, 64), 100., dtype=torch.float64)], dim=1)
    torch.testing.assert_close(observed_cdf_loss(changed, target, risk, eligible), loss, rtol=0, atol=0)


def test_loss_is_row_equal_not_observed_bin_equal():
    logits, target, risk, eligible = arrays(3)
    risk[0, :1] = True
    risk[1, :3] = True
    risk[2, :] = True
    eligible[2] = False
    expected = torch.tensor((.25 + (.25 + .5625 + .765625) / 3) / 2, dtype=torch.float64)
    actual = observed_cdf_loss(logits, target, risk, eligible)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad[2], torch.zeros(67, dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize('case', ['empty_risk', 'ineligible', 'empty_batch'])
def test_no_effective_row_returns_none(case):
    logits, target, risk, eligible = arrays(0 if case == 'empty_batch' else 2)
    if case == 'ineligible':
        risk[:] = True
        eligible[:] = False
    assert observed_cdf_loss(logits, target, risk, eligible) is None


@pytest.mark.parametrize('case', ['logit_shape', 'target_shape', 'risk_shape', 'eligible_shape',
                                 'nonfinite', 'target_nan', 'target_fraction', 'two_events',
                                 'outside_risk', 'nonprefix_risk', 'event_not_last_risk',
                                 'risk_dtype', 'eligible_dtype', 'device', 'integer_logits'])
def test_invalid_arrays_rejected(case):
    logits, target, risk, eligible = arrays()
    risk[0, :3] = True
    if case == 'logit_shape':
        logits, target, risk = logits[:, :66], target[:, :66], risk[:, :66]
    elif case == 'target_shape':
        target = target[:, :66]
    elif case == 'risk_shape':
        risk = risk[:, :66]
    elif case == 'eligible_shape':
        eligible = eligible[:, None]
    elif case == 'nonfinite':
        logits = logits.detach()
        logits[0, 5] = float('nan')
    elif case == 'target_nan':
        target[0, 1] = float('nan')
    elif case == 'target_fraction':
        target[0, 1] = .5
    elif case == 'two_events':
        target[0, :2] = 1
    elif case == 'outside_risk':
        target[0, 4] = 1
    elif case == 'nonprefix_risk':
        risk[0, 1] = False
    elif case == 'event_not_last_risk':
        target[0, 1] = 1
    elif case == 'risk_dtype':
        risk = risk.float()
    elif case == 'eligible_dtype':
        eligible = eligible.float()
    elif case == 'device':
        target = target.to('meta')
    else:
        logits = logits.long()
    with pytest.raises(ValueError):
        observed_cdf_loss(logits, target, risk, eligible)
