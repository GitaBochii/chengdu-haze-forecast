"""Reconstruct and verify a frozen bridge run without importing its runner.

All eligibility, feature, recovery-label, member and scoring calculations below
are independent of the production helpers. The CLI is deliberately explicit:
it only verifies an existing run and writes one new verification.json file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MEASUREMENTS = (
    "aqi", "pm2_5", "pm10", "so2", "no2", "co", "o3", "temperature_c",
    "relative_humidity_pct", "wind_u_ms", "wind_v_ms", "boundary_layer_height_m",
)
LABEL_COLUMNS = (
    "lower_bound_hours", "upper_bound_hours", "exact_hours", "informative",
    "label_available_at", "reason",
)
ARMS = ("archive", "legacy_control", "interval_candidate")
RAW_SOURCES = {
    "2014": "data/interim/cnemc_chengdu_2014_weather.parquet",
    "2015_2021": "data/processed/city_labeled.parquet",
    "2022": "data/interim/cnemc_chengdu_2022_weather.parquet",
    "2023": "data/interim/cnemc_chengdu_2023_weather.parquet",
    "2024_2025": "data/interim/cnemc_chengdu_2024_2025_weather.parquet",
}
LEGACY_SOURCES = {
    "cnemc_2014": "data/processed/cnemc_chengdu_2014_training_features.parquet",
    "cnemc_2015_2021": "data/processed/training_features.parquet",
    "cnemc_2022": "data/processed/cnemc_chengdu_2022_training_features.parquet",
    "cnemc_2023": "data/processed/cnemc_chengdu_2023_training_features.parquet",
    "cnemc_2024_2025": "data/processed/cnemc_chengdu_2024_2025_training_features.parquet",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(expected, base):
    if not isinstance(expected, dict) or not expected:
        raise ValueError("nonempty hash lock required")
    for name, digest in expected.items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(base) / path
        if not path.is_file() or sha256(path).lower() != str(digest).lower():
            raise ValueError(f"hash lock mismatch: {path}")
    return len(expected)


def write_exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _hourly_times(values):
    times = pd.DatetimeIndex(values)
    if times.empty or times.tz is None or times.hasnans or not times.is_unique:
        raise ValueError("unique nonmissing timezone-aware hours required")
    if not times.equals(times.floor("h")):
        raise ValueError("exact hourly timestamps required")
    return times


def _iaqi(values, breaks):
    raw = np.asarray(values, dtype=float)
    out = np.interp(raw, breaks, [0, 50, 100, 150, 200, 300, 400, 500])
    out[(raw < 0) | np.isnan(raw)] = np.nan
    return np.rint(out)


def rebuild_grid(sources):
    parts = []
    for source, frame in sources.items():
        times = _hourly_times(frame["timestamp"])
        part = pd.DataFrame({"timestamp": times})
        for column in MEASUREMENTS:
            part[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float) if column in frame else np.nan
        if np.isinf(part[list(MEASUREMENTS)].to_numpy()).any():
            raise ValueError("infinite measurements")
        part["source"] = str(source)
        parts.append(part)
    if not parts:
        raise ValueError("measurement sources required")
    original = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    if original.timestamp.duplicated().any():
        raise ValueError("overlapping measurement timestamps")
    time_grid = pd.date_range(original.timestamp.min(), original.timestamp.max(), freq="h")
    present = time_grid.isin(original.timestamp)
    result = original.set_index("timestamp").reindex(time_grid.rename("timestamp")).reset_index()
    result["input_hour_observed"] = present
    result["pm2_5_iaqi"] = _iaqi(result.pm2_5.rolling(24, min_periods=24).mean(),
                                 [0, 35, 75, 115, 150, 250, 350, 500])
    result["pm10_iaqi"] = _iaqi(result.pm10.rolling(24, min_periods=24).mean(),
                                [0, 50, 150, 250, 350, 420, 500, 600])
    result["particle_primary"] = result.aqi.gt(100) & result[["pm2_5_iaqi", "pm10_iaqi"]].max(axis=1).gt(100)
    for name, period, values in (("hour", 24, result.timestamp.dt.hour), ("month", 12, result.timestamp.dt.month)):
        result[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        result[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    result["wind_speed_ms"] = np.hypot(result.wind_u_ms, result.wind_v_ms)
    generated = {}
    for name in MEASUREMENTS:
        series = result[name]
        prior = series.shift(1)
        for lag in (1, 3, 24):
            generated[f"{name}_lag_{lag}"] = series.shift(lag)
        generated[f"{name}_roll_3"] = prior.rolling(3, min_periods=1).mean()
        generated[f"{name}_roll_24"] = prior.rolling(24, min_periods=6).mean()
    result = pd.concat([result, pd.DataFrame(generated)], axis=1)
    for name in ("boundary_layer_height_m", "boundary_layer_height_m_lag_1", "boundary_layer_height_m_lag_3",
                 "boundary_layer_height_m_lag_24", "boundary_layer_height_m_roll_3", "boundary_layer_height_m_roll_24"):
        result[f"{name}_is_missing"] = result[name].isna().astype(float)
    result["boundary_layer_height_m_change_3_is_missing"] = (
        result.boundary_layer_height_m.isna() | result.boundary_layer_height_m_lag_3.isna()
    ).astype(float)
    return rebuild_states(result)


def rebuild_states(grid):
    times = _hourly_times(grid.timestamp)
    if not times.equals(pd.date_range(times[0], times[-1], freq="h")):
        raise ValueError("states require a complete ordered grid")
    values = grid[["aqi", "pm2_5_iaqi", "pm10_iaqi"]].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("infinite state inputs")
    active, episode, block, dirty_streak, clean_streak = 0, 0, 1, 0, 0
    split_next = False
    rows = []
    for aqi, pm25, pm10 in values:
        if split_next:
            block += 1
            split_next = False
        dirty = bool(aqi > 100 and (pm25 > 100 or pm10 > 100))
        known_clear = bool(aqi <= 100 or (aqi > 100 and pm25 <= 100 and pm10 <= 100))
        dirty_streak = dirty_streak + 1 if dirty else 0
        clean_streak = clean_streak + 1 if np.isfinite(aqi) and aqi <= 100 else 0
        if not np.isfinite(aqi):
            active = 0
        elif clean_streak == 6:
            active = 0
            split_next = True
        elif active == 0 and dirty_streak == 3:
            episode += 1
            active = episode
        rows.append((active, block, bool(active and dirty),
                     "confirmed_pollution" if dirty else "confirmed_not_pollution" if known_clear else "unknown"))
    result = grid.copy()
    result[["episode_id", "continuity_block", "eligible", "pollution_state"]] = pd.DataFrame(rows, index=result.index)
    return result


def recover_bounds(grid, origins, stable_hours=6):
    times = _hourly_times(grid.timestamp)
    origin_times = _hourly_times(origins).tz_convert(times.tz)
    chronological = pd.date_range(times.min(), times.max(), freq="h")
    values = pd.Series(pd.to_numeric(grid.aqi, errors="raise").to_numpy(dtype=float), index=times).reindex(chronological).to_numpy()
    if np.isinf(values).any() or isinstance(stable_hours, bool) or stable_hours < 1:
        raise ValueError("invalid recovery inputs")
    positions = chronological.get_indexer(origin_times)
    if (positions < 0).any():
        raise ValueError("origin outside observation grid")
    windows = np.lib.stride_tricks.sliding_window_view(np.r_[values, np.full(stable_hours, np.nan)], stable_hours)
    possible = ~np.any(np.isfinite(windows) & (windows > 100), axis=1)
    confirmed = np.all(np.isfinite(windows) & (windows <= 100), axis=1)
    indices = np.arange(len(windows))
    next_possible = np.minimum.accumulate(np.where(possible, indices, len(windows))[::-1])[::-1]
    next_confirmed = np.minimum.accumulate(np.where(confirmed, indices, len(windows))[::-1])[::-1]
    rows = []
    for origin, position in zip(origin_times, positions, strict=True):
        lower_position = int(next_possible[position + 1])
        upper_position = int(next_confirmed[position + 1])
        lower = float(lower_position - position)
        upper = float(upper_position - position) if upper_position < len(chronological) else np.inf
        exact = lower == upper
        rows.append({"timestamp": origin, "lower_bound_hours": lower, "upper_bound_hours": upper,
                     "exact_hours": lower if exact else np.nan, "informative": bool(np.isfinite(upper) or lower > 1),
                     "label_available_at": chronological[upper_position + stable_hours - 1] if np.isfinite(upper) else chronological[-1],
                     "reason": "exact" if exact else "interval" if np.isfinite(upper) else "right_censored"})
    return pd.DataFrame(rows)


def expected_members(population, cutoff, block):
    mask = (population.timestamp < pd.Timestamp(cutoff)) & (population.label_available_at < pd.Timestamp(cutoff))
    mask &= population.continuity_block != block
    return population.loc[mask].copy()


def _validate_score_inputs(frame, prediction):
    prediction = np.asarray(prediction, dtype=float)
    if frame.empty or prediction.shape != (len(frame),) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("all origins require aligned finite nonnegative predictions")
    if frame.timestamp.isna().any() or frame.timestamp.duplicated().any() or frame.continuity_block.isna().any():
        raise ValueError("unique origin and process identity required")
    lo, hi, exact = (frame[name].to_numpy(dtype=float) for name in ("lower_bound_hours", "upper_bound_hours", "exact_hours"))
    if np.isinf(exact).any():
        raise ValueError("invalid exact labels")
    selected = np.isfinite(exact)
    if not np.isfinite(lo).all() or (lo <= 0).any() or np.isnan(hi).any() or (hi < lo).any():
        raise ValueError("invalid bounds")
    if not np.array_equal(selected, lo == hi) or not np.array_equal(exact[selected], lo[selected]):
        raise ValueError("invalid exact labels")
    return prediction, lo, hi, exact, selected


def _mean(values, mask):
    return float(np.mean(np.asarray(values)[mask])) if np.any(mask) else None


def _block_mean(values, blocks, mask):
    return float(pd.Series(np.asarray(values)[mask]).groupby(np.asarray(blocks)[mask]).mean().mean()) if np.any(mask) else None


def score_predictions(frame, prediction):
    p, lo, hi, exact, known = _validate_score_inputs(frame, prediction)
    blocks = frame.continuity_block.to_numpy()
    uncertain, tail, all_rows = ~known, known & (exact >= 24), np.ones(len(frame), dtype=bool)
    low = np.maximum(np.maximum(lo - p, p - hi), 0)
    high = np.maximum(np.abs(lo - p), np.abs(hi - p))
    finite_upper = np.isfinite(hi).all()
    bias = p - exact
    result = {"rows": len(frame), "blocks": int(pd.Series(blocks).nunique()), "prediction_coverage": 1.0,
            "exact_rows": int(known.sum()), "exact_blocks": int(pd.Series(blocks[known]).nunique()),
            "finite_interval_rows": int((uncertain & np.isfinite(hi)).sum()), "right_censored_rows": int(np.isposinf(hi).sum()),
            "exact_row_mae": _mean(low, known), "exact_block_mae": _block_mean(low, blocks, known),
            "tail24_rows": int(tail.sum()), "tail24_block_mae": _block_mean(low, blocks, tail),
            "all_row_error_lower": _mean(low, all_rows), "all_row_error_upper": _mean(high, all_rows) if finite_upper else None,
            "all_block_error_lower": _block_mean(low, blocks, all_rows),
            "all_block_error_upper": _block_mean(high, blocks, all_rows) if finite_upper else None,
            "uncertain_blocks": int(pd.Series(blocks[uncertain]).nunique()), "uncertain_row_distance": _mean(low, uncertain),
            "uncertain_block_distance": _block_mean(low, blocks, uncertain), "uncertain_below_fraction": _mean(p < lo, uncertain),
            "exact_error_p90": float(np.quantile(low[known], .9)) if known.any() else None,
            "exact_signed_bias": _mean(bias, known), "exact_early_fraction": _mean(bias < 0, known),
            "exact_late_fraction": _mean(bias > 0, known), "tail24_blocks": int(pd.Series(blocks[tail]).nunique())}
    result["subgroups"] = {}
    for name, mask in (("finite_interval", uncertain & np.isfinite(hi)), ("right_censored", np.isposinf(hi))):
        result["subgroups"][name] = {"rows": int(mask.sum()), "blocks": int(pd.Series(blocks[mask]).nunique()),
                                     "row_distance": _mean(low, mask), "block_distance": _block_mean(low, blocks, mask),
                                     "below_fraction": _mean(p < lo, mask), "above_fraction": _mean(p > hi, mask)}
    return result


def paired_bounds(frame, baseline, candidate):
    lo, hi = frame.lower_bound_hours.to_numpy(dtype=float), frame.upper_bound_hours.to_numpy(dtype=float)
    baseline, candidate = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if baseline.shape != lo.shape or candidate.shape != lo.shape or not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("aligned finite paired predictions required")
    left, right = np.minimum(baseline, candidate), np.maximum(baseline, candidate)
    truth_low, truth_high = np.maximum(left, np.minimum(right, lo)), np.maximum(left, np.minimum(right, hi))
    delta_low = np.abs(candidate - truth_low) - np.abs(baseline - truth_low)
    delta_high = np.abs(candidate - truth_high) - np.abs(baseline - truth_high)
    return np.minimum(delta_low, delta_high), np.maximum(delta_low, delta_high)


def paired_summary(frame, baseline, candidate):
    _validate_score_inputs(frame, baseline)
    _validate_score_inputs(frame, candidate)
    low, high = paired_bounds(frame, baseline, candidate)
    exact = frame.exact_hours.notna().to_numpy()
    uncertain, all_rows = ~exact, np.ones(len(frame), dtype=bool)
    blocks = frame.continuity_block.to_numpy()
    result = {"rows": len(frame), "blocks": int(pd.Series(blocks).nunique()), "prediction_coverage": 1.0,
              "exact_rows": int(exact.sum()), "exact_blocks": int(pd.Series(blocks[exact]).nunique()),
              "uncertain_rows": int(uncertain.sum()), "uncertain_blocks": int(pd.Series(blocks[uncertain]).nunique()),
              "finite_interval_rows": int((uncertain & np.isfinite(frame.upper_bound_hours)).sum()),
              "right_censored_rows": int(np.isposinf(frame.upper_bound_hours).sum()),
              "exact_row_delta": _mean(low, exact), "exact_block_delta": _block_mean(low, blocks, exact)}
    for subgroup, mask in (("all", all_rows), ("uncertain", uncertain)):
        for bound, values in (("lower", low), ("upper", high)):
            result[f"{subgroup}_row_delta_{bound}"] = _mean(values, mask)
            result[f"{subgroup}_block_delta_{bound}"] = _block_mean(values, blocks, mask)
    return result


def acceptance_gate(archive, candidate, paired, control=None):
    def nonworse(name):
        return archive.get(name) is not None and candidate.get(name) is not None and candidate[name] <= archive[name]

    checks = {"coverage_100": all(s.get("prediction_coverage") == 1.0 for s in (archive, candidate, control or archive)),
              "nonempty_exact_block_and_tail": all(candidate.get(key, 0) > 0 for key in ("exact_rows", "exact_blocks", "tail24_rows", "tail24_blocks")),
              "exact_row_mae_20pct": archive.get("exact_row_mae") is not None and candidate.get("exact_row_mae") is not None
                  and archive["exact_row_mae"] > 0 and candidate["exact_row_mae"] <= .8 * archive["exact_row_mae"],
              "exact_block_not_worse": nonworse("exact_block_mae"), "tail24_block_not_worse": nonworse("tail24_block_mae"),
              "paired_all_row_upper_not_positive": paired.get("all_row_delta_upper") is not None and paired["all_row_delta_upper"] <= 0,
              "paired_all_block_upper_not_positive": paired.get("all_block_delta_upper") is not None and paired["all_block_delta_upper"] <= 0}
    return {"checks": checks, "passed": bool(all(checks.values())),
            "failure_reasons": [key for key, value in checks.items() if not value],
            "exact_row_relative_reduction": 1 - candidate["exact_row_mae"] / archive["exact_row_mae"]
                if candidate.get("exact_row_mae") is not None and archive.get("exact_row_mae") is not None and archive["exact_row_mae"] > 0 else None}


def compare_array(expected, actual, name):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape:
        raise ValueError(f"{name}: shape mismatch {expected.shape} != {actual.shape}")
    if np.issubdtype(expected.dtype, np.integer) or expected.dtype == bool:
        if not np.array_equal(expected, actual):
            raise ValueError(f"{name}: exact integer/boolean value mismatch")
    elif np.issubdtype(expected.dtype, np.number):
        left, right = expected.astype(float), actual.astype(float)
        if (not np.array_equal(np.isnan(left), np.isnan(right))
                or not np.array_equal(np.isposinf(left), np.isposinf(right))
                or not np.array_equal(np.isneginf(left), np.isneginf(right))):
            raise ValueError(f"{name}: NaN/infinity pattern mismatch")
        finite = np.isfinite(left)
        if not np.allclose(left[finite], right[finite], rtol=1e-10, atol=1e-10):
            raise ValueError(f"{name}: numeric value mismatch")
    elif not np.array_equal(expected, actual):
        raise ValueError(f"{name}: value mismatch")


def compare_frame(expected, actual, columns, name):
    if len(expected) != len(actual) or not set(columns).issubset(actual.columns):
        raise ValueError(f"{name}: row count or schema mismatch")
    try:
        pd.testing.assert_frame_equal(expected[columns].reset_index(drop=True), actual[columns].reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    except AssertionError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def compare_json(expected, actual, name):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"{name}: JSON schema mismatch")
        for key, value in expected.items():
            compare_json(value, actual[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{name}: JSON list mismatch")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare_json(left, right, f"{name}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not np.isclose(expected, actual, rtol=1e-9, atol=1e-9):
            raise ValueError(f"{name}: numeric mismatch")
    elif expected != actual:
        raise ValueError(f"{name}: value mismatch")


def matrix_digest(values):
    array = np.array(values, dtype="<f8", order="C", copy=True)
    array[np.isnan(array)] = np.nan
    header = json.dumps({"shape": list(array.shape), "dtype": "<f8"}, sort_keys=True).encode("ascii")
    return hashlib.sha256(header + b"\n" + array.tobytes(order="C")).hexdigest()


def run_file(run, name):
    destination = (Path(run) / name).resolve()
    if not destination.is_relative_to(Path(run).resolve()):
        raise ValueError("artifact path outside run")
    return destination


def validate_run_directory(run, experiment):
    run, allowed = Path(run).resolve(), (Path(experiment) / "runs").resolve()
    if run == allowed or not run.is_dir() or not run.is_relative_to(allowed):
        raise ValueError("verification is restricted to existing experiment runs")
    return run


def _add_bounds(frame, grid):
    result = frame.drop(columns=[name for name in LABEL_COLUMNS if name in frame]).copy().reset_index(drop=True)
    bounds = recover_bounds(grid, result.timestamp)
    for name in LABEL_COLUMNS:
        result[name] = bounds[name].to_numpy() if name != "label_available_at" else bounds[name]
    return result


def rebuild_legacy(frames, grid, feature_names):
    parts = []
    for source, original in frames.items():
        part = original.copy()
        part["timestamp"] = pd.to_datetime(part.timestamp, utc=True).dt.tz_convert("Asia/Shanghai")
        part["event_key"] = str(source) + ":" + part.event_id.astype(str)
        for name in feature_names:
            if name.endswith("_is_missing"):
                base = name.removesuffix("_is_missing")
                if base == "boundary_layer_height_m_change_3":
                    part[name] = (part.boundary_layer_height_m.isna() | part.boundary_layer_height_m_lag_3.isna()).astype(float)
                else:
                    part[name] = part[base].isna().astype(float)
        parts.append(part[["timestamp", "event_key", "remaining_hours", *feature_names]])
    result = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if result.timestamp.duplicated().any():
        raise ValueError("duplicate legacy origins")
    result = result.rename(columns={"remaining_hours": "legacy_remaining_hours"})
    result["continuity_block"] = result.timestamp.map(grid.set_index("timestamp").continuity_block)
    if result.continuity_block.isna().any():
        raise ValueError("legacy origin outside grid")
    result = _add_bounds(result, grid)
    result["population_row"] = np.arange(len(result), dtype=np.int64)
    return result


def rebuild_evaluation(archive, grid):
    archive = archive.sort_values("timestamp").reset_index(drop=True)
    result = archive[["timestamp", "event_key", "remaining_hours", "strict_prediction_hours"]].rename(
        columns={"remaining_hours": "legacy_remaining_hours", "strict_prediction_hours": "archive_prediction_hours"}).copy()
    aligned = grid.set_index("timestamp").reindex(result.timestamp)
    if aligned.continuity_block.isna().any():
        raise ValueError("evaluation origin outside grid")
    result["eval_row"] = np.arange(len(result), dtype=np.int64)
    result["year"] = result.timestamp.dt.year
    result["grid_index"] = pd.DatetimeIndex(grid.timestamp).get_indexer(result.timestamp)
    for name in ("continuity_block", "pollution_state", "input_hour_observed"):
        result[name] = aligned[name].to_numpy()
    result["causal_eligible"] = aligned.eligible.to_numpy()
    result["cutoff"] = result.continuity_block.map(grid.groupby("continuity_block").timestamp.min())
    return _add_bounds(result, grid)


def combine_locks(groups):
    combined = {}
    for group in groups:
        for name, digest in group.items():
            if name in combined and combined[name].lower() != str(digest).lower():
                raise ValueError(f"hash lock conflict: {name}")
            combined[name] = str(digest).lower()
    return combined


def validate_inventory(records, evaluation):
    wanted = {(arm, int(block)) for block in evaluation.continuity_block.unique() for arm in ARMS[1:]}
    got = [(record["arm"], int(record["continuity_block"])) for record in records]
    ids = [record["model_id"] for record in records]
    if len(got) != len(wanted) or set(got) != wanted or len(set(ids)) != len(ids):
        raise ValueError("model inventory missing, duplicated or contains an undeclared arm/block")
    for record in records:
        if record["model_id"] != f"{record['arm']}_b{record['continuity_block']}":
            raise ValueError("model inventory identity mismatch")


def validate_trained_config(config, feature_count):
    learner = config["learner"]
    generic = learner["generic_param"]
    if int(generic["seed"]) != 42:
        raise ValueError("trained seed differs from 42")
    if int(generic["nthread"]) != 1:
        raise ValueError("trained nthread differs from 1")
    if int(learner["learner_model_param"]["num_feature"]) != feature_count:
        raise ValueError("trained feature count mismatch")
    booster = learner["gradient_booster"]
    if int(booster["gbtree_model_param"]["num_trees"]) != 500:
        raise ValueError("trained round count differs from 500")
    if booster["gbtree_train_param"]["tree_method"] != "hist":
        raise ValueError("trained tree method differs from hist")
    objective = learner["objective"]
    if objective["name"] != "survival:aft" or objective["aft_loss_param"]["aft_loss_distribution"] != "normal":
        raise ValueError("trained objective/distribution mismatch")
    if not np.isclose(float(objective["aft_loss_param"]["aft_loss_distribution_scale"]), 1.2, rtol=1e-6, atol=1e-6):
        raise ValueError("trained AFT scale mismatch")
    parameters = booster["tree_train_param"]
    for name, wanted in {"eta": .05, "max_depth": 4, "min_child_weight": 8, "subsample": .9,
                         "colsample_bytree": .8, "lambda": 3}.items():
        if not np.isclose(float(parameters[name]), wanted, rtol=1e-6, atol=1e-6):
            raise ValueError(f"trained parameter mismatch: {name}")


def compute_scores(evaluation, predictions, years=(2024, 2025)):
    def subset_score(frame, values):
        return {"arms": {arm: score_predictions(frame, values[arm]) for arm in ARMS},
                "paired": {f"interval_candidate_vs_{reference}": paired_summary(frame, values[reference], values["interval_candidate"])
                           for reference in ARMS[:2]}}

    annual, gates = {}, {}
    for year in years:
        mask = evaluation.year.eq(year).to_numpy()
        metrics = subset_score(evaluation.loc[mask], {arm: np.asarray(values)[mask] for arm, values in predictions.items()})
        annual[str(year)] = metrics
        gates[str(year)] = acceptance_gate(metrics["arms"]["archive"], metrics["arms"]["interval_candidate"],
                                           metrics["paired"]["interval_candidate_vs_archive"], metrics["arms"]["legacy_control"])
    scores = {"annual": annual, "pooled": subset_score(evaluation, predictions),
              "gates": {"annual": gates, "passed": all(item["passed"] for item in gates.values()),
                        "failure_reasons": [f"{year}:{reason}" for year in years for reason in gates[str(year)]["failure_reasons"]]}}
    rows = []
    for (year, block), process in evaluation.groupby(["year", "continuity_block"], sort=True):
        for arm in ARMS:
            metrics = score_predictions(process, np.asarray(predictions[arm])[process.index.to_numpy()])
            record = {"year": int(year), "continuity_block": int(block), "arm": arm,
                      "old_events": int(process.event_key.nunique()), **{key: value for key, value in metrics.items() if key != "subgroups"}}
            for name, subgroup in metrics["subgroups"].items():
                record.update({f"{name}_{key}": value for key, value in subgroup.items()})
            rows.append(record)
    return scores, pd.DataFrame(rows)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fixed_parameters():
    return {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.2, "eta": .05, "max_depth": 4, "min_child_weight": 8,
            "subsample": .9, "colsample_bytree": .8, "lambda": 3.0, "seed": 42,
            "tree_method": "hist", "verbosity": 0, "nthread": 1}


def historical_drift_records(historical, current, root):
    app = str((Path(root) / "src/haze_forecast/app.py").resolve())
    old_digest = "92bc58a1a818ce6ecaa58d6d8c0d4d01e6fac603eb21b346f2e8b4ad0396e1f7"
    new_digest = "7f222647d2a9bc31ee3040b121f987da5b95621fe2913a9c697e175b1568724c"
    _require(set(historical) == set(current), "historical input inventory drift")
    _require(historical.get(app) == old_digest and current.get(app) == new_digest,
             "historical app audit drift is not the exact predeclared pair")
    for path in historical:
        if path != app:
            _require(historical[path] == current[path], f"unapproved historical input drift: {path}")
    return [{"path": app, "historical_sha256": old_digest, "current_sha256": new_digest,
             "reason": "Noncomputational webpage audit entry in the old interval report; the exact historical/current pair was approved in PROTOCOL before fitting"}]


def validate_run_report(report, root, run):
    groups = ("input_sha256", "protected_sha256", "code_sha256", "environment_sha256")
    _require(all(isinstance(report.get(name), dict) and report[name] for name in (*groups, "frozen_sha256", "output_sha256")),
             "required nonempty hash groups missing")
    union = combine_locks([report[name] for name in groups])
    compare_json(union, {key: value.lower() for key, value in report["frozen_sha256"].items()}, "frozen hash union")
    required_inputs = [*RAW_SOURCES.values(), *LEGACY_SOURCES.values(),
        "artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet",
        "artifacts/task01_recovery_head/observed_business_dataset_v1/report.json",
        "artifacts/task01_recovery_head/interval_business_labels_v1/origins.parquet",
        "artifacts/task01_recovery_head/interval_business_labels_v1/report.json",
        "artifacts/strict_history_backtest/predictions.parquet", "sites-haze/dist/history.json",
        "artifacts/cnemc_2025_deployment_candidate/model_metadata.json"]
    input_paths = {Path(name).resolve() for name in report["input_sha256"]}
    _require(all((root / name).resolve() in input_paths for name in required_inputs), "original measurement/input hash coverage incomplete")
    dataset_report = _read_json(root / "artifacts/task01_recovery_head/observed_business_dataset_v1/report.json")
    label_report = _read_json(root / "artifacts/task01_recovery_head/interval_business_labels_v1/report.json")
    historical = combine_locks([dataset_report["input_sha256"], dataset_report["deployment_sha256"], label_report["input_sha256"]])
    compare_json(historical, report.get("historical_input_sha256"), "preserved historical input hash map")
    current = {path: sha256(path) for path in historical}
    drift = historical_drift_records(historical, current, root)
    compare_json(drift, report.get("historical_audit_drift"), "exact predeclared historical audit drift")
    _require(all(report["input_sha256"].get(path) == digest for path, digest in current.items()),
             "current historical-audit inputs are not locked throughout run")
    required_code = [root / "experiments/web_recovery_bridge_20261002" / name for name in (
        "PROTOCOL.md", "bridge_runner.py", "test_bridge_runner.py", "verify_bridge.py", "test_verify_bridge.py",
        "sequence_clock_adapter.py", "test_sequence_clock_adapter.py")]
    required_code.append(root / "pyproject.toml")
    code_paths = {Path(name).resolve() for name in report["code_sha256"]}
    _require(all(path.resolve() in code_paths for path in required_code), "protocol/code hash coverage incomplete")
    protected_paths = {Path(name).resolve() for name in report["protected_sha256"]}
    for name in ("artifacts/cnemc_2025_deployment_candidate", "artifacts/strict_history_backtest",
                 "project-showcase", "sites-haze/dist", "releases/runtime_20261002"):
        directory = root / name
        files = [path.resolve() for path in directory.rglob("*") if path.is_file()] if directory.is_dir() else []
        _require(bool(files), f"protected directory missing or empty: {directory}")
        _require(all(path in protected_paths for path in files), f"protected hash coverage incomplete: {directory}")
    pinned = {
        "artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet": "52efbd1bfef81643c317c193895104baff61a3eee1b0ca465066e3e39cec473e",
        "artifacts/task01_recovery_head/interval_business_labels_v1/origins.parquet": "11aa1f83d6d8e3eb884c0f17a72a9428266192c6e8073481c74e9252f9efcf44",
        "artifacts/strict_history_backtest/predictions.parquet": "016edd04e5cac982aa78d31a998a2bc8834215e67977e32e8ba1dd95b915dbd6",
        "sites-haze/dist/history.json": "c457a1a96fd8f62b9601492d08f48461db7c6bf432ccea03f9af7f4274bc7214",
    }
    verify_hashes({str(root / path): digest for path, digest in pinned.items()}, root)
    verify_hashes(union, root)
    outputs = report["output_sha256"]
    required_outputs = {"evaluation.parquet", "candidate_population.parquet", "legacy_population.parquet", "features.npz",
                        "environment.json", "execution_record_at_start.md", "training_members.parquet", "models.json",
                        "predictions.parquet", "process_scores.parquet", "scores.json"}
    _require(required_outputs.issubset(outputs), "output hash coverage incomplete")
    actual_outputs = {path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file() and path.name != "report.json"}
    _require(actual_outputs == set(outputs), "output inventory/hash mismatch")
    for name, digest in outputs.items():
        _require(sha256(run_file(run, name)) == digest.lower(), f"output hash mismatch: {name}")
    _require(report.get("schema_version") == 1 and report.get("status") == "pending_independent_verification", "run is not completed pending verification")
    compare_json({"seed": 42, "num_boost_round": 500, "nthread": 1}, report.get("configuration"), "fixed configuration")
    compare_json(_fixed_parameters(), report.get("aft_parameters"), "fixed AFT parameters")
    _require(report.get("years") == [2024, 2025] and report.get("historical_shadow_evaluation") is True,
             "both historical years required")
    _require(report.get("blind_test") is False and report.get("deployed") is False
             and report.get("archive_is_primary_reference") is True and report.get("protected_unchanged") is True,
             "run scope/protection status mismatch")
    return union


def _verify_environment(environment, locked):
    wanted_packages = {"numpy", "pandas", "pyarrow", "xgboost", "scipy", "scikit-learn", "lightgbm", "catboost", "optuna", "joblib"}
    _require(set(environment["packages"]) == wanted_packages, "dependency roster mismatch")
    _require(environment["python_version"] == sys.version, "Python version differs from frozen runtime")
    _require(Path(environment["python_executable"]).resolve() == Path(sys.executable).resolve(), "Python executable differs from frozen runtime")
    expected_records = {}
    for package, version in environment["packages"].items():
        _require(importlib.metadata.version(package) == version, f"dependency version mismatch: {package}")
        distribution = importlib.metadata.distribution(package)
        records = [Path(distribution.locate_file(item)).resolve() for item in distribution.files if item.name == "RECORD"]
        _require(bool(records), f"dependency RECORD missing: {package}")
        expected_records.update({str(path): sha256(path) for path in records})
    compare_json(expected_records, environment["record_sha256"], "dependency RECORD locks")
    expected = combine_locks([expected_records, {environment["python_executable"]: environment["python_executable_sha256"]}])
    compare_json(expected, locked, "dependency/executable frozen hashes")
    verify_hashes(expected, Path.cwd())


def _label_counts(frame):
    lower, upper = frame.lower_bound_hours.to_numpy(), frame.upper_bound_hours.to_numpy()
    return {"exact": int((lower == upper).sum()), "interval": int(((lower != upper) & np.isfinite(upper)).sum()),
            "right_censored": int(np.isposinf(upper).sum())}


def _validate_native_trees(path, feature_count):
    model = _read_json(path)
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    _require(len(trees) == 500, "native tree inventory differs from 500")
    for tree in trees:
        left, right, feature = (np.asarray(tree[key], dtype=int) for key in ("left_children", "right_children", "split_indices"))
        _require(len(left) == len(right) == len(feature), "invalid native tree shape")
        _require(np.isfinite(np.asarray(tree["split_conditions"], dtype=float)).all(), "nonfinite native tree values")
        stack, visited = [(0, 0)], set()
        while stack:
            node, depth = stack.pop()
            _require(node not in visited and 0 <= node < len(left) and depth <= 4, "native tree cycle/shape/depth mismatch")
            visited.add(node)
            if left[node] == -1:
                _require(right[node] == -1, "invalid native leaf")
            else:
                _require(0 <= feature[node] < feature_count and right[node] >= 0, "invalid native split")
                stack.extend(((int(left[node]), depth + 1), (int(right[node]), depth + 1)))
        _require(len(visited) == len(left), "unreachable native tree nodes")


def verify_run(run, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    experiment = root / "experiments/web_recovery_bridge_20261002"
    run = validate_run_directory(run, experiment)
    _require(not (run / "verification.json").exists(), "verification.json already exists; exclusive verification cannot overwrite")
    report_path = run / "report.json"
    report_hash = sha256(report_path)
    report = _read_json(report_path)
    frozen = validate_run_report(report, root, run)
    _verify_environment(_read_json(run / "environment.json"), report["environment_sha256"])
    metadata = _read_json(root / "artifacts/cnemc_2025_deployment_candidate/model_metadata.json")
    names = metadata["feature_names"]
    _require(len(names) == len(set(names)) == 87, "frozen feature order is not unique 87 columns")
    compare_json(names, report["feature_names"], "report feature order")
    original = {source: pd.read_parquet(root / path) for source, path in RAW_SOURCES.items()}
    grid = rebuild_grid(original)
    stored_grid = pd.read_parquet(root / "artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet")
    grid_columns = ["timestamp", *names, "continuity_block", "episode_id", "eligible", "pollution_state", "input_hour_observed"]
    compare_frame(grid, stored_grid, grid_columns, "independent complete grid")
    candidate = _add_bounds(grid.loc[grid.eligible].copy(), grid)
    candidate["population_row"] = np.arange(len(candidate), dtype=np.int64)
    saved_origins = pd.read_parquet(root / "artifacts/task01_recovery_head/interval_business_labels_v1/origins.parquet")
    compare_frame(candidate, saved_origins, ["timestamp", *names, "continuity_block", *LABEL_COLUMNS], "independent canonical origins")
    legacy = rebuild_legacy({source: pd.read_parquet(root / path) for source, path in LEGACY_SOURCES.items()}, grid, names)
    archive = pd.read_parquet(root / "artifacts/strict_history_backtest/predictions.parquet")
    archive["timestamp"] = pd.to_datetime(archive.timestamp, utc=True).dt.tz_convert("Asia/Shanghai")
    archive = archive.loc[archive.timestamp.dt.year.isin([2024, 2025])].sort_values("timestamp").reset_index(drop=True)
    evaluation = rebuild_evaluation(archive, grid)
    compare_frame(evaluation, pd.read_parquet(run / "evaluation.parquet"), list(evaluation), "all fixed evaluation origins")
    compare_frame(candidate, pd.read_parquet(run / "candidate_population.parquet"),
                  ["timestamp", *names, "population_row", "continuity_block", *LABEL_COLUMNS], "candidate complete population")
    compare_frame(legacy, pd.read_parquet(run / "legacy_population.parquet"), list(legacy), "legacy complete population")
    expected_counts = {"grid_rows": len(grid), "candidate_population": len(candidate), "legacy_population": len(legacy),
        "evaluation_rows": len(evaluation), "evaluation_blocks": int(evaluation.continuity_block.nunique()),
        "planned_fits": int(evaluation.continuity_block.nunique() * 2),
        "annual": {str(year): {"rows": int(evaluation.year.eq(year).sum()),
            "old_events": int(evaluation.loc[evaluation.year.eq(year), "event_key"].nunique()),
            "blocks": int(evaluation.loc[evaluation.year.eq(year), "continuity_block"].nunique()),
            "causal_eligible": int(evaluation.loc[evaluation.year.eq(year), "causal_eligible"].sum()),
            "labels": _label_counts(evaluation.loc[evaluation.year.eq(year)])} for year in (2024, 2025)}}
    _require((len(grid), len(candidate), len(legacy), len(evaluation), evaluation.continuity_block.nunique()) == (102024, 7890, 8638, 701, 32),
             "frozen full-population/block counts differ")
    _require(expected_counts["annual"]["2024"]["rows"] == 383 and expected_counts["annual"]["2025"]["rows"] == 318,
             "fixed annual population differs")
    compare_json(expected_counts, report["counts"], "cohort counts")
    candidate_features = grid.set_index("timestamp").loc[evaluation.timestamp, names].to_numpy(dtype=float)
    legacy_features = legacy.set_index("timestamp").loc[evaluation.timestamp, names].to_numpy(dtype=float)
    compare_array(legacy_features, archive[names].to_numpy(dtype=float), "legacy source/archive feature vectors")
    with np.load(run / "features.npz", allow_pickle=False) as features:
        _require(set(features.files) == {"timestamps", "feature_names", "candidate", "legacy"}, "evaluation feature archive schema mismatch")
        compare_array(np.asarray(names), features["feature_names"], "NPZ feature order")
        compare_array(pd.DatetimeIndex(evaluation.timestamp).asi8, features["timestamps"], "NPZ fixed origin timestamps")
        compare_array(candidate_features, features["candidate"], "all independently reconstructed candidate features")
        compare_array(legacy_features, features["legacy"], "all independently reconstructed legacy features")
    models = _read_json(run / "models.json")
    _require(models["schema_version"] == 1, "model manifest schema mismatch")
    compare_json(names, models["feature_names"], "model feature order")
    validate_inventory(models["models"], evaluation)
    _require(len(models["models"]) == report["fitted_models"] == 64, "all 64 declared models required")
    members = pd.read_parquet(run / "training_members.parquet")
    _require(set(members.model_id) == {record["model_id"] for record in models["models"]}, "foreign/missing model membership")
    saved_predictions = pd.read_parquet(run / "predictions.parquet")
    compare_frame(evaluation, saved_predictions, list(evaluation), "prediction fixed cohort and labels")
    predictions = {"archive": evaluation.archive_prediction_hours.to_numpy(dtype=float),
                   "legacy_control": np.full(len(evaluation), np.nan), "interval_candidate": np.full(len(evaluation), np.nan)}
    import xgboost as xgb

    checked_members = 0
    for record in models["models"]:
        arm, block = record["arm"], int(record["continuity_block"])
        population = legacy if arm == "legacy_control" else candidate
        eval_positions = np.flatnonzero(evaluation.continuity_block.eq(block).to_numpy())
        cutoff = grid.loc[grid.continuity_block.eq(block), "timestamp"].min()
        _require(pd.Timestamp(record["cutoff"]) == cutoff, "model cutoff is not full-grid block start")
        expected = expected_members(population, cutoff, block)
        _require(not expected.empty, "no mature members for model")
        wanted_members = expected[["population_row", "timestamp", "continuity_block", *LABEL_COLUMNS]].copy()
        wanted_members["legacy_remaining_hours"] = expected.legacy_remaining_hours.to_numpy() if arm == "legacy_control" else np.nan
        wanted_members["model_id"], wanted_members["arm"] = record["model_id"], arm
        wanted_members["evaluation_block"], wanted_members["cutoff"] = block, cutoff
        actual = members.loc[members.model_id.eq(record["model_id"])]
        compare_frame(wanted_members, actual, list(wanted_members), f"exact members {record['model_id']}")
        checked_members += len(expected)
        _require(record["training_member_path"] == "training_members.parquet", "undeclared training membership source")
        _require(record["train_rows"] == len(expected) and record["eval_rows"] == len(eval_positions), "model train/evaluation counts differ")
        compare_json(_label_counts(expected), record["label_counts"], "model mature label counts")
        _require(pd.Timestamp(record["training_last_timestamp"]) == expected.timestamp.max()
                 and pd.Timestamp(record["max_label_available_at"]) == expected.label_available_at.max(), "model training/maturity extrema differ")
        _require(record["training_features_sha256"] == matrix_digest(expected[names].to_numpy(dtype=float)), "training feature digest differs")
        timestamp_hash = hashlib.sha256(pd.DatetimeIndex(expected.timestamp).asi8.astype("<i8").tobytes()).hexdigest()
        _require(record["training_timestamps_sha256"] == timestamp_hash, "training origin digest differs")
        compare_json(names, record["feature_names"], "model input order")
        compare_json(names, record["booster_feature_names"], "actual fitted booster feature order")
        compare_json(report["configuration"], record["configuration"], "model fixed configuration")
        compare_json(_fixed_parameters(), record["aft_parameters"], "model fixed AFT parameters")
        _require(record["num_boosted_rounds"] == 500, "model fitted rounds differ")
        validate_trained_config(record["actual_trained_booster_config"], len(names))
        path = run_file(run, record["model_path"])
        _require(record["model_path"] in report["output_sha256"] and sha256(path) == record["model_sha256"], "model hash is not locked locally")
        _validate_native_trees(path, len(names))
        booster = xgb.Booster(params={"nthread": 1})
        booster.load_model(path)
        _require(booster.num_boosted_rounds() == 500 and booster.num_features() == 87, "loaded booster round/feature count mismatch")
        compare_json(names, booster.feature_names, "loaded booster feature order")
        objective = json.loads(booster.save_config())["learner"]["objective"]
        _require(objective["name"] == "survival:aft" and objective["aft_loss_param"]["aft_loss_distribution"] == "normal",
                 "loaded AFT objective/distribution mismatch")
        _require(np.isclose(float(objective["aft_loss_param"]["aft_loss_distribution_scale"]), 1.2, rtol=1e-6), "loaded AFT scale mismatch")
        values = legacy_features if arm == "legacy_control" else candidate_features
        matrix = pd.DataFrame(values[eval_positions], columns=names)
        inferred = booster.predict(xgb.DMatrix(matrix, missing=np.nan))
        _require(np.isfinite(inferred).all() and (inferred >= 0).all(), "loaded model produces invalid predictions")
        compare_array(inferred, saved_predictions.loc[eval_positions, arm].to_numpy(dtype=float), f"all predictions {record['model_id']}")
        predictions[arm][eval_positions] = inferred
    _require(checked_members == len(members), "unused or duplicate training members")
    row_expected = evaluation.copy()
    for arm in ARMS:
        prediction = predictions[arm]
        _validate_score_inputs(evaluation, prediction)
        row_expected[arm] = prediction
        row_expected[f"{arm}_error_lower"] = np.maximum(np.maximum(evaluation.lower_bound_hours - prediction, prediction - evaluation.upper_bound_hours), 0)
        row_expected[f"{arm}_error_upper"] = np.maximum(abs(evaluation.lower_bound_hours - prediction), abs(evaluation.upper_bound_hours - prediction))
        row_expected[f"{arm}_exact_error"] = abs(prediction - evaluation.exact_hours)
        row_expected[f"{arm}_signed_error"] = prediction - evaluation.exact_hours
    for reference in ARMS[:2]:
        lower, upper = paired_bounds(evaluation, predictions[reference], predictions["interval_candidate"])
        row_expected[f"candidate_vs_{reference}_delta_lower"] = lower
        row_expected[f"candidate_vs_{reference}_delta_upper"] = upper
    compare_frame(row_expected, saved_predictions, list(row_expected), "all per-origin predictions/error/paired bounds")
    score, process = compute_scores(evaluation, predictions)
    compare_json(score, _read_json(run / "scores.json"), "all independently recomputed scores")
    compare_json(score, report["metrics"], "report scores")
    compare_frame(process, pd.read_parquet(run / "process_scores.parquet"), list(process), "all process scores")
    verify_hashes(frozen, root)
    _require(sha256(report_path) == report_hash, "report changed during independent verification")
    for name, digest in report["output_sha256"].items():
        _require(sha256(run_file(run, name)) == digest.lower(), f"output changed during independent verification: {name}")
    result = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
              "run_dir": str(run), "report_sha256": report_hash, "verifier_sha256": sha256(__file__),
              "grid_rows_reconstructed": len(grid), "canonical_origins_reconstructed": len(candidate),
              "legacy_origins_reconstructed": len(legacy), "fixed_origins_verified": len(evaluation),
              "models_verified": len(models["models"]), "training_member_rows_verified": checked_members,
              "refitted_predictions_verified": len(evaluation) * 2, "feature_order": names,
              "metrics": score, "yearly_acceptance_passed": score["gates"]["passed"],
              "limits": ["Native model parameters and member digests form an auditable saved chain, not an independent retraining proof",
                         "Dependency versions, RECORD metadata and Python executable are locked; all installed binary bytes are not independently rehashed",
                         "2024/2025 are known historical years, not a new blind test", "No deployment or future real-time accuracy claim"]}
    write_exclusive_json(run / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing approved run under this experiment's runs directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = validate_run_directory(args.run, Path(__file__).resolve().parent)
    try:
        result = verify_run(run, root)
    except Exception as exc:
        if not (run / "verification.json").exists():
            write_exclusive_json(run / "verification.json", {"schema_version": 1, "status": "failed",
                "independent_verification_passed": False, "error": str(exc), "verifier_sha256": sha256(__file__)})
        raise
    print(json.dumps({"status": result["status"], "models_verified": result["models_verified"],
                      "fixed_origins_verified": result["fixed_origins_verified"], "yearly_acceptance_passed": result["yearly_acceptance_passed"]}))


if __name__ == "__main__":
    main()
