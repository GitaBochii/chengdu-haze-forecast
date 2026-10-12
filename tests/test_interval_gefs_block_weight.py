import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_interval_gefs_block_weight import block_weights


def test_equal_total_weight_and_mean_one():
    keys = pd.Series([1, 1, 1, 2], index=[5, 7, 8, 10])
    weights = block_weights(keys)
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2])
    assert weights.mean() == pytest.approx(1.)
    assert weights[:3].sum() == pytest.approx(weights[3])


@pytest.mark.parametrize('keys', [[], [1, None]])
def test_reject_invalid_keys(keys):
    with pytest.raises(ValueError):
        block_weights(pd.Series(keys))
