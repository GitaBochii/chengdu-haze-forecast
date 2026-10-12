import numpy as np

from scripts.audit_recovery_windows import scan_windows


def test_missing_hour_makes_earlier_window_possible_not_confirmed():
    assert scan_windows([150, 80, np.nan, 80, 80, 80, 80, 80, 80]) == (1, 3)


def test_pollution_inside_window_rules_out_early_recovery():
    assert scan_windows([150, 80, 80, 120, 80, 80, 80, 80, 80, 80]) == (4, 4)


def test_archive_end_is_not_observed_recovery():
    assert scan_windows([150, 80, 80]) == (1, np.inf)
