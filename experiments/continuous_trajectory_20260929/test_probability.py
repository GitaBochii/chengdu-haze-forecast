import numpy as np
from probability import binary_metrics, select_candidate


def test_metrics_include_false_negatives():
    m = binary_metrics(np.array([1, 1, 0, 0]), np.array([.9, .2, .8, .1]), .5)
    assert (m['tp'], m['fn'], m['fp'], m['tn']) == (1, 1, 1, 1)
    assert m['f2'] == .5
    assert m['false_positive_rate'] == .5


def test_selection_uses_supplied_development_only():
    y = np.array([1, 1, 0, 0])
    selected, grid = select_candidate(y, {'unweighted': np.array([.8, .7, .2, .1]),
                                         'balanced': np.array([.8, .2, .7, .1])})
    assert selected['candidate'] == 'unweighted'
    assert selected['threshold'] == .7
    assert len(grid) == 38


def test_no_positive_predictions():
    m = binary_metrics(np.array([1, 0]), np.array([.1, .1]), .5)
    assert m['precision'] is None
    assert m['recall'] == m['f2'] == 0
