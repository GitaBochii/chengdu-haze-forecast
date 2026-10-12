from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haze_forecast.recovery_conditional import StateAgeRecoveryPM25GNN, StateOnlyRecoveryPM25GNN
from haze_forecast.recovery_selection import development_gate
from scripts.verify_recovery_conditional import check_gate, check_parameters


def test_saved_parameter_shapes_and_unused_age_column(tmp_path):
    panel = {'feature_names': list('abcdef'), 'coords': np.array([[104., 30.], [105., 31.]]),
             'scaler': SimpleNamespace(mean=np.zeros(6), scale=np.ones(6))}
    for model_class in (StateOnlyRecoveryPM25GNN, StateAgeRecoveryPM25GNN):
        torch.manual_seed(42)
        model = model_class(6, panel['coords'], np.zeros(2), np.ones(2))
        torch.save(model.state_dict(), tmp_path / 'legacy_state.pt')
        assert check_parameters(tmp_path, panel, model_class)['added_parameters'] == 3219
    torch.manual_seed(42)
    model = StateOnlyRecoveryPM25GNN(6, panel['coords'], np.zeros(2), np.ones(2))
    state = model.state_dict()
    state['conditional_head.0.weight'][:, -1] += 1
    torch.save(state, tmp_path / 'legacy_state.pt')
    with pytest.raises(AssertionError):
        check_parameters(tmp_path, panel, StateOnlyRecoveryPM25GNN)


def test_independent_pair_gate_checks_each_saved_condition():
    results = {}
    for year, events in (('2022', [1, 2]), ('2023', [3])):
        results[year] = {
            'old': {'row_mae': 10., 'event_equal_mae': 10., 'tail_event_equal_mae': 20., 'tail_events': events},
            'new': {'row_mae': 9., 'event_equal_mae': 9., 'tail_event_equal_mae': 19., 'tail_events': events}}
    saved = development_gate([{'legacy': x['old'], 'balanced': x['new']} for x in results.values()])
    assert check_gate(saved, results, 'old', 'new')['accepted']
    saved['checks']['tail_better'] = False
    with pytest.raises(AssertionError):
        check_gate(saved, results, 'old', 'new')
