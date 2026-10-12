import numpy as np

from scripts.audit_regional_input_provenance import fill_provenance


def test_initial_gap_and_seventh_missing_row_use_mean():
    values = np.array([np.nan, 5., *([np.nan] * 7), 8.])[:, None]
    status = fill_provenance(values)
    np.testing.assert_array_equal(status[:, 0], [2, 0, 1, 1, 1, 1, 1, 1, 2, 0])


def test_future_changes_cannot_change_past_provenance():
    values = np.array([[1., np.nan], [np.nan, 2.], [3., 4.]])
    before = fill_provenance(values)
    values[-1] = np.nan
    np.testing.assert_array_equal(before[:2], fill_provenance(values)[:2])
