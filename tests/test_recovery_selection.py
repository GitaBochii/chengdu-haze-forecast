import numpy as np
import pandas as pd
import pytest

from haze_forecast.recovery_selection import development_gate, selection_keys, selection_metrics


def records():
    return pd.DataFrame({'actual_hours': [10., 20., 80.], 'predicted_hours': [10., 20., 20.],
                         'event_id': [1, 2, 2], 'label_verified_in_scoring_window': [True]*3})


def test_balanced_rule_gives_tail_explicit_weight_without_changing_cap():
    result = selection_metrics(records())
    assert result['row_mae'] == 16
    assert result['event_equal_mae'] == 12
    assert result['short_event_equal_mae'] == 0
    assert result['tail_event_equal_mae'] == 48
    assert result['balanced_mae'] == 24
    assert result['group_overlap_events'] == 1
    assert selection_keys(result)['legacy'] == (12, 16)


def test_no_tail_cannot_select_balanced_and_unverified_is_excluded():
    frame = records()
    frame.loc[2, 'label_verified_in_scoring_window'] = False
    result = selection_metrics(frame)
    assert result['tail_rows'] == 0
    with pytest.raises(ValueError):
        selection_keys(result)
    frame.loc[2, 'label_verified_in_scoring_window'] = True
    frame.loc[2, 'predicted_hours'] = np.nan
    assert selection_metrics(frame)['tail_event_equal_mae'] == 0


def test_development_gate_does_not_accept_tail_gain_with_overall_regression():
    old = {'row_mae': 10., 'event_equal_mae': 10., 'tail_event_equal_mae': 20., 'tail_events': [1, 2]}
    new = {'row_mae': 9., 'event_equal_mae': 9., 'tail_event_equal_mae': 19., 'tail_events': [1, 2]}
    folds = [{'legacy': old, 'balanced': new},
             {'legacy': {**old, 'tail_events': [3]}, 'balanced': {**new, 'tail_events': [3]}}]
    assert development_gate(folds)['accepted']
    folds[1]['balanced']['event_equal_mae'] = 15
    assert not development_gate(folds)['accepted']
    assert not development_gate(folds[:1])['accepted']
