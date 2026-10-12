import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_aft_event_weight import assert_mature, event_weights


@pytest.mark.parametrize("power", [0.0, 0.5, 1.0])
def test_weights_normalized(power):
    keys = pd.Series(["a", "b", "b", "b", "b"])
    weights = event_weights(keys, power)
    assert weights.mean() == pytest.approx(1.0)
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert weights[0] / weights[1] == pytest.approx(4**power)
    if power == 1:
        assert weights[0] == pytest.approx(weights[1:].sum())
    if power == 0:
        assert np.array_equal(weights, np.ones(5))


@pytest.mark.parametrize("keys,power", [([], 1), ([None], 1), (["a"], -1), (["a"], float("nan"))])
def test_invalid_weights(keys, power):
    with pytest.raises(ValueError):
        event_weights(pd.Series(keys, dtype=object), power)


def test_label_maturity_requires_confirmation_and_strict_boundary():
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01"]), "remaining_hours": [2.]})
    assert assert_mature(rows, pd.Timestamp("2023-01-01 08:00")).iloc[0] == pd.Timestamp("2023-01-01 07:00")
    with pytest.raises(AssertionError):
        assert_mature(rows, pd.Timestamp("2023-01-01 07:00"))
    rows["remaining_hours"] = np.nan
    with pytest.raises(AssertionError):
        assert_mature(rows, pd.Timestamp("2023-01-02"))
