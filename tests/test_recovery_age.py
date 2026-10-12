import numpy as np
import pandas as pd
import pytest
import torch

from haze_forecast.recovery_age import AgeRecoveryPM25GNN, causal_recovery_age
from haze_forecast.regional_recovery_network import RecoveryPM25GNN


def test_confirmation_age_and_six_clean_hours_reset():
    aqi = [150] * 4 + [90] * 6 + [150] * 3
    np.testing.assert_array_equal(causal_recovery_age(aqi, [True] * len(aqi)),
                                  [0, 0, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 3])


def test_missing_values_preserve_memory_but_break_confirmation_runs():
    aqi = [150, 150, np.nan, 150, 150, 150, np.nan, 90, 90, np.nan, 90]
    np.testing.assert_array_equal(causal_recovery_age(aqi, [True] * len(aqi)),
                                  [0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(causal_recovery_age([150] * 8,
                                  [True, True, np.nan, True, None, pd.NA, True, True]),
                                  np.zeros(8))


def test_future_changes_cannot_change_prefix_and_long_age_is_not_capped():
    original = np.full(200, 150.)
    changed = original.copy()
    changed[100:] = 80
    np.testing.assert_array_equal(causal_recovery_age(original, np.ones(200)),
                                  np.r_[0, 0, np.arange(3, 201)])
    np.testing.assert_array_equal(causal_recovery_age(original, np.ones(200))[:100],
                                  causal_recovery_age(changed, np.ones(200))[:100])


@pytest.mark.parametrize('aqi,particle', [([[150]], [True]), ([150], []), ([150], [[True]])])
def test_alignment_required(aqi, particle):
    with pytest.raises(ValueError, match='aligned'):
        causal_recovery_age(aqi, particle)


def model_args():
    return (5, np.array([[104., 30.], [104.1, 30.], [104., 30.1]]),
            np.array([10., 270.]), np.ones(2))


def test_zero_residual_preserves_base_and_projection_receives_gradient():
    torch.manual_seed(42)
    base = RecoveryPM25GNN(*model_args(), history=4, horizon=8)
    model = AgeRecoveryPM25GNN(*model_args(), history=4, horizon=8)
    result = model.load_state_dict(base.state_dict(), strict=False)
    assert result.missing_keys == ['age_projection.weight']
    assert result.unexpected_keys == []
    x = torch.randn(2, 4, 3, 5)
    age = torch.full((2, 4, 3, 1), 1.5)
    augmented = torch.cat([x, age], dim=-1)
    active = torch.ones(2, 3, 2, dtype=torch.bool)
    expected_curve, expected_logits = base(x, active)
    curve, logits = model(augmented, active)
    torch.testing.assert_close(curve, expected_curve, rtol=0, atol=0)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    logits.sum().backward()
    torch.testing.assert_close(model.age_projection.weight.grad, torch.full((3, 1), 3.))
    with torch.no_grad():
        model.age_projection.weight.fill_(2.)
    curve, logits = model(augmented, active)
    torch.testing.assert_close(curve, expected_curve, rtol=0, atol=0)
    torch.testing.assert_close(logits, expected_logits + 3., rtol=0, atol=0)
    augmented[:, :-1, :, -1] = 99
    torch.testing.assert_close(model(augmented, active)[1], logits, rtol=0, atol=0)


def test_age_channel_validation():
    model = AgeRecoveryPM25GNN(*model_args(), history=4, horizon=8)
    active = torch.ones(1, 3, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match='input'):
        model(torch.zeros(1, 4, 3, 5), active)
    x = torch.zeros(1, 4, 3, 6)
    x[0, -1, 1, -1] = 1
    with pytest.raises(ValueError, match='nodes'):
        model(x, active)
    x[..., -1] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        model(x, active)
