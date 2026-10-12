import numpy as np
import pandas as pd

from scripts.evaluate_regional_common_intervals import cap_labels, curve_recovery


def test_curve_no_recovery_is_cap_not_observed_full_duration():
    assert curve_recovery(np.full(72, 120.)) == 68
    assert curve_recovery(np.full(72, 80.)) == 1
    values = np.full(72, 120.)
    values[-6:] = 80
    assert curve_recovery(values) == 67


def test_capping_preserves_crossing_interval_and_source():
    frame = pd.DataFrame({'lower_bound_hours': [80., 65., 10.],
                          'upper_bound_hours': [100., 75., 12.],
                          'exact_hours': [np.nan] * 3})
    result = cap_labels(frame)
    assert result.lower_bound_hours.tolist() == [68., 65., 10.]
    assert result.upper_bound_hours.tolist() == [68., 68., 12.]
    assert result.exact_hours.iloc[0] == 68
    assert result.exact_hours.iloc[1:].isna().all()
    assert frame.lower_bound_hours.iloc[0] == 80
