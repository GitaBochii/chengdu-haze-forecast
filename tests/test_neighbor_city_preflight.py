import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_neighbor_city_preflight import arm_inputs


def test_controls_preserve_missingness_and_feature_count():
    frame = pd.DataFrame({'base': [1., 2.], **{f'city48_{i:02d}': [i+1., np.nan] for i in range(24)}})
    masked = arm_inputs(frame, ['base'], 'masked')
    local = arm_inputs(frame, ['base'], 'local')
    full = arm_inputs(frame, ['base'], 'regional')
    assert masked.shape == local.shape == full.shape == (2, 25)
    pd.testing.assert_frame_equal(masked.isna(), full.isna())
    pd.testing.assert_frame_equal(local.isna(), full.isna())
    assert masked.iloc[0, 1:].eq(0).all()
    assert local.iloc[0, 1:4].tolist() == [1., 2., 3.]
    assert local.iloc[0, 4:].eq(0).all()
    pd.testing.assert_frame_equal(frame, full)
    pd.testing.assert_frame_equal(arm_inputs(frame, ['base'], 'native'), frame[['base']])


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        arm_inputs(pd.DataFrame(), [], 'unknown')
