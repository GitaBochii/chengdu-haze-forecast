import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_crossyear_forecast_control import metadata_only


def test_control_preserves_missingness_age_labels_and_source():
    frame = pd.DataFrame({'gefs_value': [12., np.nan, -3.],
                          'age': [6., 12., 18.], 'exact_hours': [1., 2., 3.]})
    original = frame.copy(deep=True)
    result = metadata_only(frame, ['gefs_value'])
    np.testing.assert_equal(result.gefs_value.to_numpy(), [0., np.nan, 0.])
    pd.testing.assert_frame_equal(result[['age', 'exact_hours']], original[['age', 'exact_hours']])
    pd.testing.assert_frame_equal(frame, original)


def test_control_rejects_missing_field():
    with pytest.raises(KeyError):
        metadata_only(pd.DataFrame({'age': [6.]}), ['gefs_value'])
