import numpy as np
import pandas as pd
import pytest

from scripts.verify_recovery_auxiliary import independent_gate, independent_scores, probabilities


def test_scores_event_weighting_and_tail_cap():
    frame = pd.DataFrame({'actual_hours': [10., 20., 100., 5.],
                          'predicted_hours': [12., 24., 60., 99.],
                          'event_id': [1, 1, 2, 3],
                          'label_verified_in_scoring_window': [True, True, True, False]})
    result = independent_scores(frame)
    assert result['row_mae'] == pytest.approx(14 / 3)
    assert result['event_equal_mae'] == 5.5
    assert result['tail_event_equal_mae'] == 8
    assert result['tail_events'] == [2]


def test_gate_requires_support_and_both_overall_metrics():
    def arm(tail, row=3., event=4., ids=(1, 2)):
        return {'row_mae': row, 'event_equal_mae': event,
                'tail_event_equal_mae': tail, 'tail_events': list(ids)}
    folds = [{'auxiliary_0_1': arm(6), 'auxiliary_0': arm(5)},
             {'auxiliary_0_1': arm(6, ids=(3,)), 'auxiliary_0': arm(5, ids=(3,))}]
    assert independent_gate(folds)['accepted']
    folds[1]['auxiliary_0']['event_equal_mae'] = 5
    assert not independent_gate(folds)['accepted']
    folds[1]['auxiliary_0']['event_equal_mae'] = 4
    folds[1]['auxiliary_0']['tail_events'] = [1]
    folds[1]['auxiliary_0_1']['tail_events'] = [1]
    assert not independent_gate(folds)['accepted']


def test_probabilities_stable_and_normalized():
    actual = probabilities(np.array([[0., 0.], [-1000., 1000.]]))
    np.testing.assert_allclose(actual, [[.5, .25, .25], [0., 1., 0.]])
