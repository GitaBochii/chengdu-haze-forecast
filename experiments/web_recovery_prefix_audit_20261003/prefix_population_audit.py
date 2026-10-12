"""Audit selected completed-event membership from measurement prefixes.

The legacy event selection below is an independent NumPy reconstruction. Only
canonical recovery bounds reuse the frozen independent bridge implementation.
No model or production label helper is loaded, fitted, or evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / "experiments/web_recovery_bridge_20261002/runs/20261002T153221834258Z"
PARENT_REPORT_SHA = "70217776f86326cd9a9d4436af12d2d352b4f11ea69b2f1fc68becb1951f01d0"
PARENT_VERIFY_SHA = "7003b164439cbf87e4146459ae2f219bf6f3b45af1cc70407daf7935eba1d703"
CANONICAL_MATH_SHA = "9477b5452830f82756219d317a6928e12708897a061cb442eeb65304fccd6b2b"
SOURCES = {
    "2014": ("data/interim/cnemc_chengdu_2014_weather.parquet", "data/processed/cnemc_chengdu_2014_training_features.parquet"),
    "2015_2021": ("data/processed/city_labeled.parquet", "data/processed/training_features.parquet"),
    "2022": ("data/interim/cnemc_chengdu_2022_weather.parquet", "data/processed/cnemc_chengdu_2022_training_features.parquet"),
    "2023": ("data/interim/cnemc_chengdu_2023_weather.parquet", "data/processed/cnemc_chengdu_2023_training_features.parquet"),
    "2024_2025": ("data/interim/cnemc_chengdu_2024_2025_weather.parquet", "data/processed/cnemc_chengdu_2024_2025_training_features.parquet"),
}
LABELS = ("lower_bound_hours", "upper_bound_hours", "exact_hours", "informative", "label_available_at", "reason")
HOUR_NS = 3600000000000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_locks(groups):
    result = {}
    for group in groups:
        for path, digest in group.items():
            digest = str(digest).lower()
            if path in result and result[path] != digest:
                raise ValueError(f"original/current hash lock conflict: {path}")
            result[path] = digest
    return result


def canonical_math():
    path = ROOT / "experiments/web_recovery_bridge_20261002/verify_bridge.py"
    if sha(path) != CANONICAL_MATH_SHA:
        raise ValueError("frozen canonical mathematics changed")
    spec = importlib.util.spec_from_file_location("frozen_prefix_canonical_bounds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hourly_times(values):
    times = pd.DatetimeIndex(values)
    if times.tz is None or times.hasnans or not times.is_unique or not times.equals(times.floor("h")):
        raise ValueError("unique timezone-aware exact hours required")
    return times


def _particle_iaqi(values, breaks):
    raw = pd.to_numeric(values, errors="raise").rolling(24, min_periods=24).mean().to_numpy(float)
    result = np.interp(raw, breaks, [0, 50, 100, 150, 200, 300, 400, 500])
    result[np.isnan(raw) | (raw < 0)] = np.nan
    return np.rint(result)


def completed_legacy_origins(frame):
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    times = hourly_times(frame.timestamp)
    aqi = pd.to_numeric(frame.aqi, errors="raise").to_numpy(float)
    measurements = frame[["aqi", "pm2_5", "pm10"]].to_numpy(float)
    if np.isinf(measurements).any():
        raise ValueError("infinite measurements")
    pm25 = _particle_iaqi(frame.pm2_5, [0, 35, 75, 115, 150, 250, 350, 500])
    pm10 = _particle_iaqi(frame.pm10, [0, 50, 150, 250, 350, 420, 500, 600])
    dirty = (aqi > 100) & ((pm25 > 100) | (pm10 > 100))
    clean = aqi <= 100
    selected = np.zeros(len(frame), dtype=bool)
    event_ids = np.zeros(len(frame), dtype=np.int64)
    confirmations = np.zeros(len(frame), dtype=np.int64)
    clock = times.asi8
    start, event = 0, 0
    while start < len(frame):
        if start + 3 > len(frame) or not dirty[start:start + 3].all() or not (np.diff(clock[start:start + 3]) == HOUR_NS).all():
            start += 1
            continue
        end = start + 3
        while end < len(frame):
            if end + 6 <= len(frame) and clean[end:end + 6].all() and (np.diff(clock[end:end + 6]) == HOUR_NS).all():
                event += 1
                selected[start:end] = dirty[start:end]
                event_ids[start:end] = event
                confirmations[start:end] = clock[end + 5]
                start = end + 6
                break
            end += 1
        else:
            break
    return pd.DataFrame({"timestamp": times[selected], "old_event_id": event_ids[selected],
                         "old_confirmation_at": pd.to_datetime(confirmations[selected], utc=True).tz_convert(times.tz)})


def canonical_grid(sources):
    parts = []
    for frame in sources.values():
        if frame.empty:
            continue
        part = pd.DataFrame({"timestamp": hourly_times(frame.timestamp), "aqi": pd.to_numeric(frame.aqi, errors="raise").to_numpy(float)})
        parts.append(part)
    if not parts:
        raise ValueError("nonempty measurement prefix required")
    observed = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    if observed.timestamp.duplicated().any() or np.isinf(observed.aqi).any():
        raise ValueError("overlapping or infinite measurement sources")
    grid = pd.date_range(observed.timestamp.min(), observed.timestamp.max(), freq="h")
    return observed.set_index("timestamp").reindex(grid.rename("timestamp")).reset_index()


def continuity_blocks(grid):
    block, clean, pending = 1, 0, False
    result = []
    for aqi in grid.aqi.to_numpy(float):
        if pending:
            block += 1
            pending = False
        clean = clean + 1 if np.isfinite(aqi) and aqi <= 100 else 0
        if clean == 6:
            pending = True
        result.append(block)
    return np.asarray(result, dtype=np.int64)


def compare_timestamps(expected, actual, name):
    left, right = hourly_times(expected).asi8, hourly_times(actual).asi8
    if not np.array_equal(np.sort(left), np.sort(right)):
        raise ValueError(f"{name}: timestamp population differs; missing={len(set(left)-set(right))}, extra={len(set(right)-set(left))}")


def audit_block(raws, population, parent_members, cutoff, block):
    cutoff = pd.Timestamp(cutoff)
    prefixes = {source: frame.loc[frame.timestamp.lt(cutoff)] for source, frame in raws.items()}
    selected = pd.concat([completed_legacy_origins(frame) for frame in prefixes.values()], ignore_index=True).sort_values("timestamp")
    grid = canonical_grid(prefixes)
    grid["continuity_block"] = continuity_blocks(grid)
    selected["continuity_block"] = selected.timestamp.map(grid.set_index("timestamp").continuity_block)
    selected = selected.loc[selected.continuity_block.ne(block)]
    expected = population.loc[population.timestamp.lt(cutoff) & population.label_available_at.lt(cutoff) & population.continuity_block.ne(block)]
    wanted = parent_members.loc[parent_members.model_id.eq(f"legacy_control_b{block}")]
    compare_timestamps(expected.timestamp, wanted.timestamp, "saved mature control members")
    compare_timestamps(wanted.timestamp, selected.timestamp, "independent raw-prefix completed population")
    if not selected.old_confirmation_at.lt(cutoff).all():
        raise ValueError("future old completion used for population selection")
    math = canonical_math()
    rebuilt = math.recover_bounds(grid, wanted.timestamp)
    math.compare_frame(wanted, rebuilt, ["timestamp", *LABELS], "canonical prefix bounds and availability")
    math.compare_frame(expected, wanted, ["timestamp", "population_row", "continuity_block", *LABELS, "legacy_remaining_hours"], "exact parent member metadata")
    return {"continuity_block": int(block), "cutoff": cutoff.isoformat(), "parent_training_rows": len(wanted),
            "prefix_selected_rows": len(selected), "missing_rows": 0, "extra_rows": 0,
            "future_old_completion_rows": 0, "canonical_bounds_and_maturity_match": True,
            "right_censored_training_rows": int(np.isposinf(wanted.upper_bound_hours).sum()),
            "selected_timestamps_sha256": hashlib.sha256(np.sort(hourly_times(selected.timestamp).asi8).astype("<i8").tobytes()).hexdigest()}


def write_result(path, result, allowed=HERE):
    path, allowed = Path(path).resolve(), Path(allowed).resolve()
    if not path.is_relative_to(allowed) or not path.parent.is_dir() or path.suffix.lower() != ".json":
        raise ValueError("output must be an explicit JSON path in the existing audit directory")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def run_audit():
    report = json.loads((PARENT / "report.json").read_text(encoding="utf-8"))
    verification = json.loads((PARENT / "verification.json").read_text(encoding="utf-8"))
    if sha(PARENT / "report.json") != PARENT_REPORT_SHA or sha(PARENT / "verification.json") != PARENT_VERIFY_SHA:
        raise ValueError("parent report/verification identity changed")
    if verification.get("status") != "passed" or verification.get("independent_verification_passed") is not True or verification.get("report_sha256") != PARENT_REPORT_SHA:
        raise ValueError("parent independently passed chain required")
    source_paths = [ROOT / path for pair in SOURCES.values() for path in pair]
    parent_paths = [PARENT / name for name in ("report.json", "verification.json", "evaluation.parquet", "legacy_population.parquet", "training_members.parquet")]
    code_paths = [HERE / name for name in ("PROTOCOL.md", "prefix_population_audit.py", "test_prefix_population_audit.py")]
    code_paths.append(ROOT / "experiments/web_recovery_bridge_20261002/verify_bridge.py")
    groups = {"source_sha256": {str(path.resolve()): sha(path) for path in source_paths},
              "parent_sha256": {str(path.resolve()): sha(path) for path in parent_paths},
              "protocol_and_code_sha256": {str(path.resolve()): sha(path) for path in code_paths}}
    locked = merge_locks([report["frozen_sha256"], *groups.values()])
    for path, digest in locked.items():
        if sha(path) != digest:
            raise ValueError(f"pre-audit lock drift: {path}")
    for path in parent_paths[2:]:
        if sha(path) != report["output_sha256"][path.relative_to(PARENT).as_posix()]:
            raise ValueError(f"parent population/member artifact changed: {path}")
    raws = {source: pd.read_parquet(ROOT / paths[0]) for source, paths in SOURCES.items()}
    saved = {source: pd.read_parquet(ROOT / paths[1]) for source, paths in SOURCES.items()}
    source_records, all_selected = {}, []
    for source in SOURCES:
        selected = completed_legacy_origins(raws[source])
        compare_timestamps(saved[source].timestamp, selected.timestamp, f"complete source {source}")
        source_records[source] = {"saved_rows": len(saved[source]), "reconstructed_rows": len(selected), "missing_rows": 0, "extra_rows": 0}
        all_selected.append(selected)
    population = pd.read_parquet(PARENT / "legacy_population.parquet")
    compare_timestamps(population.timestamp, pd.concat(all_selected).timestamp, "all legacy population")
    grid = canonical_grid(raws)
    grid["continuity_block"] = continuity_blocks(grid)
    if not np.array_equal(population.timestamp.map(grid.set_index("timestamp").continuity_block).to_numpy(), population.continuity_block.to_numpy()):
        raise ValueError("raw-rebuilt continuity block differs")
    evaluation = pd.read_parquet(PARENT / "evaluation.parquet")
    if len(population) != 8638 or len(evaluation) != 701 or evaluation.continuity_block.nunique() != 32:
        raise ValueError("fixed population/evaluation dimensions changed")
    members = pd.read_parquet(PARENT / "training_members.parquet")
    control = members.loc[members.arm.eq("legacy_control")]
    expected_ids = {f"legacy_control_b{block}" for block in evaluation.continuity_block.unique()}
    if set(control.model_id) != expected_ids:
        raise ValueError("parent control model membership inventory differs")
    checks = []
    for block, query in evaluation.groupby("continuity_block", sort=True):
        cutoff = query.cutoff.iloc[0]
        if not query.cutoff.eq(cutoff).all() or cutoff != grid.loc[grid.continuity_block.eq(block), "timestamp"].min():
            raise ValueError("cutoff is not unique full-grid block start")
        checks.append(audit_block(raws, population, control, cutoff, int(block)))
    for path, digest in locked.items():
        if sha(path) != digest:
            raise ValueError(f"file changed during audit: {path}")
    return {"schema_version": 1, "status": "passed", "scope": "Selected completed legacy population at the fixed 32 historical cutoffs",
            "parent_report_sha256": PARENT_REPORT_SHA, "parent_verification_sha256": PARENT_VERIFY_SHA,
            "canonical_mathematics_sha256": CANONICAL_MATH_SHA, **groups,
            "parent_original_frozen_sha256": report["frozen_sha256"], "all_sources_unchanged": True,
            "legacy_population_rows": len(population), "fixed_evaluation_origins": len(evaluation), "cutoffs_checked": len(checks),
            "parent_control_member_rows_checked": sum(item["parent_training_rows"] for item in checks),
            "source_population_checks": source_records, "cutoff_checks": checks,
            "model_fits": 0, "model_predictions": 0, "production_ready": False, "deployed": False,
            "limits": ["Membership selection is independently reconstructed; canonical label mathematics reuse the frozen independent verifier",
                       "This establishes selected completed legacy membership for these historical cutoffs, not all unfinished online supervision",
                       "Measurement timestamps and precomputed measurement values do not certify weather publication or upstream imputation availability",
                       "Legacy row-clock particle rolling and first-two-hour retrospective assignment are reproduced, not endorsed for prediction inputs",
                       "No new accuracy, model choice, deployment or future real-time claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Read-only audit, also the default")
    mode.add_argument("--output", type=Path, help="Explicit new JSON path within this audit directory; never overwrite")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    result = run_audit()
    if args.output is not None:
        write_result(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "legacy_population_rows", "cutoffs_checked", "parent_control_member_rows_checked", "model_fits", "model_predictions")}))


if __name__ == "__main__":
    main()
