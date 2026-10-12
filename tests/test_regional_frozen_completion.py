import numpy as np
import pandas as pd
import pytest

from scripts.complete_regional_frozen_predictions import checked_positions


def test_positions_keep_order_and_require_history():
    times = pd.date_range('2025-01-01', periods=100, freq='h', tz='Asia/Shanghai')
    np.testing.assert_array_equal(checked_positions(times, times[[30, 25]]), [30, 25])
    with pytest.raises(ValueError):
        checked_positions(times, times[[1]])
    with pytest.raises(ValueError):
        checked_positions(times.delete(28), times[[30]])
    with pytest.raises(ValueError):
        checked_positions(times, times[[30, 30]])
