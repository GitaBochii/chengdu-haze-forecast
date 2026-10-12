import pandas as pd
import pytest

from haze_forecast.forward_evaluation import recency_sample_weights


def test_recency_weights_halve_after_one_half_life():
    timestamps = pd.Series(pd.to_datetime(["2024-01-01", "2023-01-01", "2022-01-01"]))

    weights = recency_sample_weights(timestamps, pd.Timestamp("2024-01-01"), half_life_years=1)

    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, rel=0.01)
    assert weights[2] == pytest.approx(0.25, rel=0.01)


def test_recency_weights_reject_non_positive_half_life():
    with pytest.raises(ValueError, match="positive"):
        recency_sample_weights(pd.Series(pd.to_datetime(["2024-01-01"])), pd.Timestamp("2024-01-01"), 0)
