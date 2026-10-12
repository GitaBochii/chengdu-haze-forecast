"""Calibrate coverage-checked prediction intervals for the deployed AFT model.

Calibration sources are out-of-sample predictions in the *deployment regime*
(boundary-layer height available): the fixed base holdout and the rolling
event-wise folds of the 2014 candidate. The 2022/2023 external years are scored
as a stress test only -- their boundary-layer height is entirely missing, so
they belong to a different input regime and must not set the interval width.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators
from haze_forecast.prediction_interval import (
    CALIBRATION_LEVELS,
    MIN_PREDICTION_HOURS,
    calibrate_ratio_intervals,
    coverage_summary,
    prediction_interval,
    residual_ratios,
    write_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
HOLDOUT_PREDICTIONS = ARTIFACTS / "test_predictions.csv"
ROLLING_PREDICTIONS = ARTIFACTS / "cnemc_2014_prefix_benchmark" / "fold_predictions.csv"
ROLLING_VARIANT = "with_historical_prefix"
EXTERNAL_YEARS = (2022, 2023)
OUTPUT = ARTIFACTS / "prediction_interval_calibration.json"


def _floor(series: pd.Series) -> np.ndarray:
    return np.asarray(series, dtype=float).clip(min=MIN_PREDICTION_HOURS)


def load_calibration_sources() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{source: (actual, predicted)}`` for deployment-regime samples."""
    holdout = pd.read_csv(HOLDOUT_PREDICTIONS)
    rolling = pd.read_csv(ROLLING_PREDICTIONS)
    rolling = rolling.loc[rolling["variant"].eq(ROLLING_VARIANT)]
    if rolling.empty:
        raise ValueError(f"rolling predictions contain no {ROLLING_VARIANT} rows")
    return {
        "base_holdout": (
            holdout["actual_remaining_hours"].to_numpy(dtype=float),
            _floor(holdout["predicted_remaining_hours"]),
        ),
        "rolling_event_folds": (
            rolling["actual_remaining_hours"].to_numpy(dtype=float),
            _floor(rolling["predicted_remaining_hours"]),
        ),
    }


def load_external_sources() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Score the deployed model on held-out years that lack boundary-layer height."""
    metadata = json.loads((ARTIFACTS / "model_metadata.json").read_text(encoding="utf-8"))
    feature_names = list(metadata["feature_names"])
    model = joblib.load(ARTIFACTS / "aft_model.joblib")
    sources: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for year in EXTERNAL_YEARS:
        path = ROOT / "data" / "processed" / f"cnemc_chengdu_{year}_training_features.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        prefix = [name for name in frame.columns if name.startswith("boundary_layer_height_m")]
        frame = add_missingness_indicators(frame, prefix)
        prediction = _floor(pd.Series(model.predict(frame[feature_names])))
        sources[f"external_{year}"] = (
            frame["remaining_hours"].to_numpy(dtype=float),
            prediction,
        )
    return sources


def evaluate(
    calibration: dict[str, dict[str, float]],
    sources: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    """Score each source at every calibrated coverage level."""
    report: list[dict[str, object]] = []
    for name, (actual, predicted) in sources.items():
        for key in sorted(calibration):
            level = float(calibration[key]["coverage"])
            lower = np.array([prediction_interval(p, calibration, level)[0] for p in predicted])
            upper = np.array([prediction_interval(p, calibration, level)[1] for p in predicted])
            summary = coverage_summary(actual, lower, upper)
            report.append(
                {
                    "source": name,
                    "coverage_target": level,
                    "rows": summary["rows"],
                    "observed_coverage": round(summary["coverage"], 4),
                    "mean_width_hours": round(summary["mean_width"], 2),
                    "used_for_calibration": name
                    in {"base_holdout", "rolling_event_folds"},
                }
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=list(CALIBRATION_LEVELS),
        help="coverage levels to calibrate",
    )
    args = parser.parse_args()

    calibration_sources = load_calibration_sources()
    pooled = np.concatenate(
        [
            residual_ratios(actual, predicted)
            for actual, predicted in calibration_sources.values()
        ]
    )
    calibration = calibrate_ratio_intervals(pooled, tuple(args.levels))
    external = load_external_sources()

    payload = {
        "protocol": (
            "Asymmetric multiplicative interval calibrated on out-of-sample "
            "actual/median-prediction ratios from the deployment regime "
            "(boundary-layer height available). The deployed AFT model predicts a "
            "conditional median, so the band is not centred symmetrically."
        ),
        "default_coverage": max(args.levels),
        "calibration_sources": {
            name: int(np.asarray(actual).size) for name, (actual, _) in calibration_sources.items()
        },
        "calibration_samples": int(pooled.size),
        "stress_test_sources": {name: int(np.asarray(a).size) for name, (a, _) in external.items()},
        "calibration": calibration,
        "evaluation": evaluate(calibration, {**calibration_sources, **external}),
        "replaces": (
            "the previous symmetric prediction +/- validation-MAE band, which "
            "covered only 63.9% of the base holdout despite being presented as a "
            "90%-style empirical interval"
        ),
    }
    write_calibration(args.output, payload)

    print(f"校准样本 = {pooled.size} 行（{', '.join(calibration_sources)}）")
    for key in sorted(calibration):
        band = calibration[key]
        print(
            f"  {band['coverage']:.0%} 区间 = 预测 x "
            f"[{band['lower_ratio']:.3f}, {band['upper_ratio']:.3f}]"
        )
    print()
    header = f"{'来源':<22} {'目标':>6} {'行数':>6} {'实测覆盖率':>10} {'平均宽度(小时)':>13}"
    print(header)
    print("-" * len(header))
    for row in payload["evaluation"]:
        print(
            f"{row['source']:<22} {row['coverage_target']:>5.0%} {row['rows']:>6} "
            f"{row['observed_coverage']:>9.1%} {row['mean_width_hours']:>13.1f}"
        )
    print()
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
