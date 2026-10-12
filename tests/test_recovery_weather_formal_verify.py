import copy

import pytest

from scripts.verify_recovery_weather_formal import verify_trace


def fixture():
    trace = [{'epoch': epoch, 'training_windows': 100, 'informative_survival_origins': 8,
              'concentration_elements': 1000, 'training_survival_nll': 2., 'training_concentration_l1': .5,
              'validation_2023': {'event_equal_mae': event, 'row_mae': row}, 'selection_key': [event, row]}
             for epoch, event, row in ((1, 4., 2.), (2, 3., 2.), (3, 3., 2.))]
    selection = {'epoch': 2, 'candidate_epochs': [1, 2, 3], 'validation_year': 2023,
                 'validation_key': [3., 2.], 'rule': 'event_equal_mae_then_row_mae_earliest_tie'}
    return trace, selection


def test_trace_lexicographic_minimum_and_earliest_tie():
    trace, selection = fixture()
    assert verify_trace(trace, selection, 3, 100, 8)['epoch'] == 2


@pytest.mark.parametrize('kind', ['later_tie', 'window_drop', 'nonfinite', 'key_mismatch'])
def test_trace_rejects_invalid_selection_and_training_counts(kind):
    trace, selection = copy.deepcopy(fixture())
    if kind == 'later_tie':
        selection['epoch'] = 3
    elif kind == 'window_drop':
        trace[0]['training_windows'] = 99
    elif kind == 'nonfinite':
        trace[0]['validation_2023']['row_mae'] = float('nan')
    else:
        trace[0]['selection_key'] = [0., 0.]
    with pytest.raises(AssertionError):
        verify_trace(trace, selection, 3, 100, 8)
