import pandas as pd
import pytest

from haze_forecast.calibration import (
    MedianResidualCalibrator,
    prediction_bias_summary,
    remaining_time_segment_metrics,
)


def test_median_residual_calibrator_corrects_a_constant_underprediction():
    actual = pd.Series([10.0, 20.0, 30.0])
    predicted = pd.Series([5.0, 15.0, 25.0])

    calibrator = MedianResidualCalibrator.fit(actual, predicted)

    assert calibrator.offset_hours == 5.0
    assert calibrator.predict(pd.Series([1.0, 10.0])).tolist() == [6.0, 15.0]


def test_median_residual_calibrator_rejects_empty_or_misaligned_inputs():
    with pytest.raises(ValueError, match="equally sized"):
        MedianResidualCalibrator.fit(pd.Series([], dtype=float), pd.Series([], dtype=float))
    with pytest.raises(ValueError, match="equally sized"):
        MedianResidualCalibrator.fit(pd.Series([1.0]), pd.Series([1.0, 2.0]))


def test_bias_summary_and_time_segments_preserve_their_distinct_purposes():
    actual = pd.Series([12.0, 30.0, 72.0, 120.0])
    predicted = pd.Series([10.0, 20.0, 50.0, 80.0])

    bias = prediction_bias_summary(actual, predicted)
    segments = remaining_time_segment_metrics(actual, predicted)

    assert bias["median_signed_error_hours"] < 0
    assert [(row["segment"], row["rows"]) for row in segments] == [
        ("0-24h", 1),
        ("24-48h", 1),
        ("48-96h", 1),
        ("96h+", 1),
    ]
