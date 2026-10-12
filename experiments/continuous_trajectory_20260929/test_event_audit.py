import numpy as np
import pytest

from event_audit import transition_times, confusion, evaluate


def test_threshold_indexing_and_censoring():
    a = np.full((4, 24), 101.0)
    a[0, :6] = 100
    a[1, 18:24] = 100
    a[2, 19:24] = 100
    a[3] = 100
    onset, recovery = transition_times(a)
    assert onset.tolist() == [7, 1, 1, 25]
    assert recovery.tolist() == [1, 19, 20, 1]


def test_confusion_zero_denominators():
    m = confusion(np.array([True, True, False, False]), np.array([True, False, True, False]))
    assert [m[k] for k in ('tp', 'fn', 'fp', 'tn')] == [1, 1, 1, 1]
    assert m['precision'] == m['recall'] == m['f1'] == .5
    assert confusion(np.zeros(0, bool), np.zeros(0, bool))['recall'] is None


def test_missed_recovery_included_and_duration_distinct():
    y = np.full((2, 24), 120.)
    y[0, :6] = 80
    p = np.full((2, 24), 120.)
    m = evaluate(y, p, np.array([120, 80]), np.array(['a', 'b']))
    assert m['recovery']['fn'] == 1
    assert m['recovery']['capped_time_mae_h'] == 19
    assert m['recovery']['joint_detection_timing_mae_h'] is None
    assert m['polluted_hours']['mae_h'] == 3
    assert m['onset']['tp'] == 1


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        transition_times(np.full((1, 24), np.nan))


def test_last_hour_onset_and_interrupted_clean_window():
    a = np.full((2, 24), 100.)
    a[0, 23] = 101
    a[1] = 101
    a[1, :5] = 100
    a[1, 6:12] = 100
    onset, recovery = transition_times(a)
    assert onset.tolist() == [24, 6]
    assert recovery.tolist() == [1, 7]


def test_empty_subpopulation_and_perfect_predictions():
    y = np.full((1, 24), 80.)
    m = evaluate(y, y, np.array([80]), np.array(['a']))
    assert m['recovery']['n'] == 0
    assert m['recovery']['capped_time_mae_h'] is None
    assert m['onset']['tn'] == 1
    assert m['onset']['capped_time_mae_h'] == 0
