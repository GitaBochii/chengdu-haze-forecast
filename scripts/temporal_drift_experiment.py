"""Quantify how much of the remaining-hours error comes from temporal drift.

Chengdu haze events got dramatically shorter across the study period: the
training rows (2015--2018) have a target median near 53 h while the held-out
rows (2020--2021) sit near 22 h.  A model fitted on the stale relationship is
biased, so this script measures how much MAE improves when old rows are trimmed
or down-weighted.

Selection protocol: 3-fold rolling event splits on the base frame.  The 1213-row
chronological hold-out and the 2022/2023 external years are reported only as
confirmation, so variant choice never uses them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators, model_feature_names
from haze_forecast.training import (
    AFTSurvivalRegressor,
    event_chronological_split,
    rolling_event_splits,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

YEAR_SECONDS = 365.25 * 24 * 3600


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    sources = [c for c in frame.columns if c.startswith("boundary_layer_height_m")]
    return add_missingness_indicators(frame, sources)


def recency_weight(timestamps: pd.Series, half_life_years: float) -> np.ndarray:
    """Exponential decay weight anchored on the newest training row."""
    newest = timestamps.max()
    age_years = (newest - timestamps).dt.total_seconds() / YEAR_SECONDS
    return np.power(0.5, age_years / half_life_years)


def recency_window(timestamps: pd.Series, years: float) -> pd.Series:
    """Keep only rows within ``years`` of the newest training row."""
    newest = timestamps.max()
    age_years = (newest - timestamps).dt.total_seconds() / YEAR_SECONDS
    return age_years <= years


def build_variants() -> list[dict]:
    """Variants are defined relatively so they apply to every fold.

    They act on the 2015--2021 rows only; the 2014 prefix is always kept so the
    comparison isolates the value of recent base rows rather than re-litigating
    whether the prefix helps at all.
    """
    variants: list[dict] = [{"name": "A 全量", "kind": "all"}]
    for years in (2.0, 3.0, 4.0):
        variants.append({"name": f"R{years:.0f} 最近{years:.0f}年", "kind": "window", "years": years})
    for half_life in (1.5, 3.0, 6.0):
        variants.append(
            {"name": f"E{half_life} 衰减半衰期{half_life}年", "kind": "weight", "years": half_life}
        )
    return variants


def select_rows_and_weights(
    base_rows: pd.DataFrame, prefix: pd.DataFrame, variant: dict
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Return the fitting frame and optional per-row weights for one variant."""
    timestamps = base_rows["timestamp"]
    if variant["kind"] == "all":
        rows, weight = base_rows, None
    elif variant["kind"] == "window":
        rows, weight = base_rows[recency_window(timestamps, variant["years"])], None
    else:
        rows, weight = base_rows, recency_weight(timestamps, variant["years"])
    frame = pd.concat([prefix, rows], ignore_index=True).sort_values("timestamp")
    if weight is None:
        return frame, None
    # Prefix rows keep full weight: they are few and already resemble the target regime.
    return frame, np.concatenate([np.ones(len(prefix)), weight])


def evaluate(model: AFTSurvivalRegressor, frame: pd.DataFrame, features: list[str]) -> dict:
    prediction = np.asarray(model.predict(frame[features]), dtype=float).clip(min=0.0)
    actual = frame["remaining_hours"].to_numpy()
    error = prediction - actual
    return {
        "rows": len(frame),
        "mae": float(np.abs(error).mean()),
        "median_prediction": float(np.median(prediction)),
        "median_actual": float(np.median(actual)),
        "bias": float(error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default=str(ARTIFACTS / "temporal_drift_experiment" / "report.json")
    )
    args = parser.parse_args()

    base = _prepare(pd.read_parquet(PROCESSED / "training_features.parquet"))
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    prefix = _prepare(pd.read_parquet(PROCESSED / "cnemc_chengdu_2014_training_features.parquet"))
    external = {
        str(year): _prepare(
            pd.read_parquet(PROCESSED / f"cnemc_chengdu_{year}_training_features.parquet")
        )
        for year in (2022, 2023)
    }
    train, valid, test = event_chronological_split(base)
    features = model_feature_names(base)
    print(f"特征数={len(features)}  留出测试={len(test)} 行  2014 前缀={len(prefix)} 行\n")

    folds = rolling_event_splits(base, n_splits=args.splits)
    print(f"=== 选择阶段：{len(folds)} 折滚动验证（不使用留出测试集）===")
    selection: dict[str, dict] = {}
    for variant in build_variants():
        fold_metrics = []
        for index, (fold_train, fold_valid) in enumerate(folds, start=1):
            fit_frame, weight = select_rows_and_weights(fold_train, prefix, variant)
            model = AFTSurvivalRegressor(seed=args.seed, nthread=1).fit(
                fit_frame[features], fit_frame["remaining_hours"], sample_weight=weight
            )
            metrics = evaluate(model, fold_valid, features)
            metrics["fold"] = index
            metrics["training_rows"] = len(fit_frame)
            fold_metrics.append(metrics)
        total_rows = sum(m["rows"] for m in fold_metrics)
        weighted = sum(m["mae"] * m["rows"] for m in fold_metrics) / total_rows
        selection[variant["name"]] = {
            "folds": fold_metrics,
            "rolling_mae": weighted,
            "rolling_rows": total_rows,
        }
        detail = "  ".join(
            f"fold{m['fold']}={m['mae']:.2f}({m['training_rows']}行)" for m in fold_metrics
        )
        print(f'{variant["name"]:<22} 滚动加权MAE={weighted:>7.3f}   {detail}', flush=True)

    best = min(selection, key=lambda name: selection[name]["rolling_mae"])
    print(f"\n滚动折最优: {best}  (MAE={selection[best]['rolling_mae']:.4f})")

    print("\n=== 确认阶段：留出测试集与 2022/2023 外部年份 ===")
    confirmations: dict[str, dict] = {}
    holdout_pool = pd.concat([train, valid], ignore_index=True).sort_values("timestamp")
    header = f'{"变体":<22} {"训练行":>7} |'
    names = ["base_test", *external]
    for name in names:
        header += f" {name + ' MAE':>16}"
    print(header)
    print("-" * len(header))
    for variant in build_variants():
        rows, weight = select_rows_and_weights(holdout_pool, prefix, variant)
        started = time.time()
        model = AFTSurvivalRegressor(seed=args.seed, nthread=1).fit(
            rows[features], rows["remaining_hours"], sample_weight=weight
        )
        line = f'{variant["name"]:<22} {len(rows):>7} |'
        entry = {
            "training_rows": len(rows),
            "fit_seconds": round(time.time() - started, 1),
            "rolling_mae": selection[variant["name"]]["rolling_mae"],
            "evaluations": {},
        }
        for name, frame in [("base_test", test), *external.items()]:
            metrics = evaluate(model, frame, features)
            entry["evaluations"][name] = metrics
            line += f' {metrics["mae"]:>9.3f} ({metrics["median_prediction"]:.0f}/{metrics["median_actual"]:.0f})'
        print(line, flush=True)
        confirmations[variant["name"]] = entry

    payload = {
        "protocol": (
            "variant selection uses rolling event splits only; the chronological hold-out and "
            "2022/2023 external years are confirmation. Median column shows predicted/actual median."
        ),
        "seed": args.seed,
        "splits": args.splits,
        "feature_count": len(features),
        "selected_variant": best,
        "selection": selection,
        "confirmation": confirmations,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {output}")


if __name__ == "__main__":
    main()
