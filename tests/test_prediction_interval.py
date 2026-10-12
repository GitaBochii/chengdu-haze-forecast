import numpy as np
import pandas as pd
import pytest

from haze_forecast.prediction_interval import (
    DEFAULT_COVERAGE,
    MIN_PREDICTION_HOURS,
    calibrate_ratio_intervals,
    coverage_summary,
    prediction_interval,
    residual_ratios,
    write_calibration,
)


def test_residual_ratios_floors_the_denominator():
    ratios = residual_ratios(np.array([10.0, 10.0]), np.array([0.0, 5.0]))
    assert ratios[0] == pytest.approx(10.0 / MIN_PREDICTION_HOURS)
    assert ratios[1] == pytest.approx(2.0)


def test_residual_ratios_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        residual_ratios(np.array([1.0, 2.0]), np.array([1.0]))


def test_calibration_is_two_sided_and_asymmetric():
    ratios = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    bands = calibrate_ratio_intervals(ratios, (0.80,))

    band = bands["0.80"]
    assert band["samples"] == 6
    assert band["lower_ratio"] < 1.0 < band["upper_ratio"]
    assert band["lower_ratio"] == pytest.approx(np.quantile(ratios, 0.10))
    assert band["upper_ratio"] == pytest.approx(np.quantile(ratios, 0.90))


def test_calibration_rejects_empty_and_invalid_levels():
    with pytest.raises(ValueError, match="at least one"):
        calibrate_ratio_intervals(np.array([]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrate_ratio_intervals(np.array([1.0]), (1.5,))
    with pytest.raises(ValueError, match="finite"):
        calibrate_ratio_intervals(np.array([np.nan]))


def test_prediction_interval_is_asymmetric_and_clamped():
    calibration = calibrate_ratio_intervals(np.linspace(0.3, 6.0, 200), (0.90,))
    lower, upper = prediction_interval(20.0, calibration, 0.90)

    assert lower < 20.0 < upper
    assert lower == pytest.approx(20.0 * calibration["0.90"]["lower_ratio"])
    assert upper == pytest.approx(20.0 * calibration["0.90"]["upper_ratio"])


def test_prediction_interval_clamps_negative_predictions():
    calibration = calibrate_ratio_intervals(np.linspace(0.3, 6.0, 200), (0.90,))
    lower, upper = prediction_interval(-5.0, calibration, 0.90)

    assert lower == 0.0
    assert upper == 0.0


def test_prediction_interval_reports_missing_levels():
    calibration = calibrate_ratio_intervals(np.linspace(0.3, 6.0, 200), (0.80,))
    with pytest.raises(KeyError, match="0.90"):
        prediction_interval(20.0, calibration, DEFAULT_COVERAGE)


def test_coverage_summary_counts_observations_inside_the_band():
    summary = coverage_summary(
        np.array([1.0, 5.0, 9.0]),
        np.array([0.0, 6.0, 0.0]),
        np.array([2.0, 7.0, 8.0]),
    )
    assert summary["rows"] == 3
    assert summary["coverage"] == pytest.approx(1 / 3)
    assert summary["mean_width"] == pytest.approx((2.0 + 1.0 + 8.0) / 3)


def test_coverage_summary_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        coverage_summary(np.array([1.0, 2.0]), np.array([0.0]), np.array([3.0]))


def test_calibrated_band_transfers_to_an_unseen_sample():
    rng = np.random.default_rng(7)
    predicted = rng.uniform(2.0, 60.0, size=4000)
    # Right-skewed multiplicative error, mirroring the dissipation target.
    actual = predicted * rng.lognormal(mean=0.1, sigma=0.9, size=4000)
    calibration_ratios = residual_ratios(actual[:2000], predicted[:2000])

    calibration = calibrate_ratio_intervals(calibration_ratios, (0.80, 0.90))
    held_out_pred = predicted[2000:]
    lower = np.array([prediction_interval(p, calibration, 0.90)[0] for p in held_out_pred])
    upper = np.array([prediction_interval(p, calibration, 0.90)[1] for p in held_out_pred])

    summary = coverage_summary(actual[2000:], lower, upper)
    assert summary["coverage"] == pytest.approx(0.90, abs=0.03)


def test_write_calibration_produces_sorted_json(tmp_path):
    path = tmp_path / "nested" / "calibration.json"
    write_calibration(path, {"b": 1, "a": {"d": 2, "c": 3}})
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index('"b"')


def test_calibration_accepts_pandas_input():
    ratios = pd.Series([0.5, 1.0, 1.0, 2.0])
    bands = calibrate_ratio_intervals(ratios, (0.80,))
    assert bands["0.80"]["samples"] == 4
