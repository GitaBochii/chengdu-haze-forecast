import numpy as np
import pytest
import torch

from haze_forecast.history_state_residual import HistoryStateResidual
from haze_forecast.recovery_survival import masked_survival_loss
from scripts.train_history_state_residual import shuffled_stream, training_pools


@pytest.mark.parametrize('nonlinear', [False, True])
def test_zero_initialization_and_history_only_gradient(nonlinear):
    model = HistoryStateResidual(nonlinear=nonlinear)
    state = torch.randn(5, 128)
    delta = model(state)
    assert delta.shape == (5, 67)
    assert torch.count_nonzero(delta) == 0
    loss = masked_survival_loss(delta, torch.zeros_like(delta),
                                torch.ones_like(delta, dtype=torch.bool), torch.ones(5, dtype=torch.bool))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize('nonlinear', [False, True])
def test_seed_recreates_initial_weights(nonlinear):
    torch.manual_seed(42)
    first = HistoryStateResidual(nonlinear=nonlinear)
    torch.manual_seed(42)
    second = HistoryStateResidual(nonlinear=nonlinear)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


@pytest.mark.parametrize('state', [torch.zeros(128), torch.zeros(2, 127),
                                  torch.zeros(2, 128, dtype=torch.int64),
                                  torch.full((2, 128), float('nan'))])
def test_invalid_states_rejected(state):
    with pytest.raises(ValueError):
        HistoryStateResidual()(state)


def test_stream_is_deterministic_full_permutation_cycles():
    indices = np.arange(275, dtype=np.int64) + 100
    first = shuffled_stream(indices)
    np.testing.assert_array_equal(first, shuffled_stream(indices))
    assert first.shape == (500, 64)
    flat = first.ravel()
    for start in range(0, len(flat) - len(indices) + 1, len(indices)):
        np.testing.assert_array_equal(np.sort(flat[start:start + len(indices)]), indices)
    assert np.isin(first, indices).all()


@pytest.mark.parametrize('indices', [np.array([], dtype=int), np.array([1, 1]),
                                    np.array([-1, 2]), np.array([1., 2.])])
def test_invalid_stream_indices_rejected(indices):
    with pytest.raises(ValueError):
        shuffled_stream(indices)


def fixture_arrays():
    return {'train_state': np.zeros((4, 128), np.float32),
            'train_logits': np.zeros((4, 67), np.float32),
            'train_target': np.zeros((4, 67), np.float32),
            'train_risk': np.array([[True] * 67, [True] + [False] * 66,
                                    [False] * 67, [True] * 67]),
            'train_eligible': np.array([True, True, True, False]),
            'train_mask': np.array([[[True]], [[False]], [[True]], [[True]]]),
            'train_origin_ns': np.array([1420041600000000000] * 4)}


def test_pools_keep_partial_censoring_but_exclude_no_risk_and_ineligible():
    pools = training_pools(fixture_arrays())
    np.testing.assert_array_equal(pools['covered'], [0])
    np.testing.assert_array_equal(pools['all'], [0, 1])


@pytest.mark.parametrize('corruption', ['future', 'nonfinite', 'nonprefix', 'event_outside', 'two_events'])
def test_invalid_training_data_rejected(corruption):
    arrays = fixture_arrays()
    if corruption == 'future':
        arrays['train_origin_ns'][0] = 1704067200000000000
    elif corruption == 'nonfinite':
        arrays['train_state'][0, 0] = np.nan
    elif corruption == 'nonprefix':
        arrays['train_risk'][1, 2] = True
    elif corruption == 'event_outside':
        arrays['train_target'][1, 5] = 1
    else:
        arrays['train_target'][0, :2] = 1
    with pytest.raises(ValueError):
        training_pools(arrays)
