"""Independently count frozen out-of-sample direction evidence, without models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


BRIDGE_SHA256 = "9477b5452830f82756219d317a6928e12708897a061cb442eeb65304fccd6b2b"
PARENT_REPORT_SHA256 = "21aa39d5a08a683e23904000eeeccbb3c31fda30201cf30e64b4cf28ab063998"
PARENT_VERIFICATION_SHA256 = "17f3a9e71d15e95c3f9f02149a5bd1c1f27abe784682979c050263e44e0f8566"
PARENT_VERIFIER_SHA256 = "1c3052b2f51023d8e08649ed47dff614f05065ba1dd161d9964b8151400c6c6e"
PARENT_RELATIVE = "experiments/web_recovery_factors_20261002/runs/20261002T161429405383Z"
EXPERIMENT = "experiments/web_recovery_direction_audit_20261003"
ARMS = ("archive", "interval_candidate", "grid_all_legacy_origins")
PRIMARY = "grid_all_legacy_origins"
BUCKETS = ("lt24", "ge24")
DIRECTIONS = ("insufficient", "none", "raise", "lower")
CURRENT_DIRECTIONS = ("raise", "lower", "unknown", "current_empty")
AGREEMENTS = ("consistent", "opposite", "unknown", "current_empty", "no_signal", "insufficient_history")
CONFIGURATION = {"prediction_bucket_threshold_hours": 24., "min_past_blocks": 6,
                 "certain_majority_threshold": .5, "required_majority_block_fraction": 2 / 3,
                 "strict_timestamp_before_cutoff": True, "strict_label_available_before_cutoff": True,
                 "exclude_current_continuity_block": True, "past_years": [2024, 2025]}
SCOPE = "Fixed saved 2024/2025 predictions; descriptive prefix direction screening only"
PROTECTED_DIRECTORIES = ("artifacts/cnemc_2025_deployment_candidate", "artifacts/strict_history_backtest",
                         "project-showcase", "sites-haze/dist", "releases/runtime_20261002")
RATE_NAMES = {"early_lower": "certain_early", "early_upper": "possible_early",
              "late_lower": "certain_late", "late_upper": "possible_late"}
KEYS = ["evaluation_block", "cutoff", "evaluation_year", "arm", "bucket"]
MEMBER_COLUMNS = [*KEYS, "origin_index", "timestamp", "year", "continuity_block", "label_available_at",
                  "lower_bound_hours", "upper_bound_hours", "exact_hours", "reason", "prediction"]


def bridge():
    path = Path(__file__).resolve().parent.parent / "web_recovery_bridge_20261002/verify_bridge.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError("frozen independent mathematics changed")
    spec = importlib.util.spec_from_file_location("direction_frozen_independent_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise ValueError(message)


def rates(frame, arm):
    """Use explicit row and block loops, not the new runner's mathematics."""
    require(arm in ARMS, "unknown direction arm")
    counts = dict.fromkeys(RATE_NAMES, 0)
    blocks = {}
    for row in frame.itertuples(index=False):
        prediction = float(getattr(row, arm))
        lower, upper = float(row.lower_bound_hours), float(row.upper_bound_hours)
        require(np.isfinite(prediction) and prediction >= 0 and np.isfinite(lower) and lower >= 0
                and not np.isnan(upper) and upper >= lower, "invalid prediction or duration bounds")
        block = row.continuity_block
        require(not pd.isna(block), "missing continuity block")
        values = (prediction < lower, prediction < upper, prediction > upper, prediction > lower)
        if block not in blocks:
            blocks[block] = {"rows": 0, **dict.fromkeys(RATE_NAMES, 0)}
        blocks[block]["rows"] += 1
        for name, value in zip(RATE_NAMES, values, strict=True):
            counts[name] += int(value)
            blocks[block][name] += int(value)
    result = {"rows": len(frame), "blocks": len(blocks)}
    for name in RATE_NAMES:
        result["row_" + name] = counts[name] / len(frame) if len(frame) else None
        total = 0.
        for record in blocks.values():
            total += record[name] / record["rows"]
        result["block_" + name] = total / len(blocks) if blocks else None
    for side in ("early", "late"):
        votes = 0
        for record in blocks.values():
            votes += int(record[side + "_lower"] / record["rows"] > .5)
        result[side + "_majority_blocks"] = votes
    return result


def signal(stats):
    if stats["blocks"] < 6:
        return "insufficient"
    for side, name in (("early", "raise"), ("late", "lower")):
        if stats["block_" + side + "_lower"] > .5 and stats[side + "_majority_blocks"] * 3 >= stats["blocks"] * 2:
            return name
    return "none"


def prefix(frame, cutoff, block, arm, bucket):
    require(arm in ARMS and bucket in BUCKETS, "unknown prefix arm/bucket")
    positions = []
    cutoff = pd.Timestamp(cutoff)
    for position, row in enumerate(frame.itertuples(index=False)):
        value = float(getattr(row, arm))
        in_bucket = value < 24 if bucket == "lt24" else value >= 24
        if row.timestamp < cutoff and row.label_available_at < cutoff and row.continuity_block != block and in_bucket:
            positions.append(position)
    return frame.iloc[positions].copy()


def current_direction(stats):
    if stats["rows"] == 0:
        return "current_empty"
    if stats["row_early_lower"] > .5:
        return "raise"
    if stats["row_late_lower"] > .5:
        return "lower"
    return "unknown"


def agreement(past, current):
    require(past in DIRECTIONS and current in CURRENT_DIRECTIONS, "unknown direction enum")
    if current == "current_empty":
        return "current_empty"
    if past == "insufficient":
        return "insufficient_history"
    if past == "none":
        return "no_signal"
    if current == "unknown":
        return "unknown"
    return "consistent" if past == current else "opposite"


def serialized_rates(stats, prefix_name):
    return {prefix_name + "_rows": stats["rows"], prefix_name + "_blocks": stats["blocks"],
            **{weight + "_" + public: stats[weight + "_" + internal]
               for internal, public in RATE_NAMES.items() for weight in ("row", "block")}}


def summary_leaf(rows):
    n = len(rows)
    direction_counts = dict.fromkeys(DIRECTIONS, 0)
    current_counts = dict.fromkeys(CURRENT_DIRECTIONS, 0)
    agreement_counts = dict.fromkeys(AGREEMENTS, 0)
    sufficient, empty = 0, 0
    for row in rows.itertuples(index=False):
        require(row.direction in DIRECTIONS and row.current_direction in CURRENT_DIRECTIONS and row.agreement in AGREEMENTS,
                "unknown summary enum")
        direction_counts[row.direction] += 1
        current_counts[row.current_direction] += 1
        agreement_counts[row.agreement] += 1
        sufficient += int(row.support_sufficient)
        empty += int(row.past_rows == 0)
    signals = direction_counts["raise"] + direction_counts["lower"]
    comparable = agreement_counts["consistent"] + agreement_counts["opposite"]
    fraction = lambda count: count / n if n else None
    return {"combinations": n, "direction_counts": direction_counts,
            "signal_combinations": signals, "signal_coverage": fraction(signals),
            "support_sufficient_combinations": sufficient, "empty_prefix_combinations": empty,
            "insufficient_history_combinations": direction_counts["insufficient"],
            "no_signal_combinations": direction_counts["none"],
            "current_empty_combinations": current_counts["current_empty"],
            "current_direction_counts": current_counts, "agreement_counts": agreement_counts,
            "comparable_combinations": comparable, "comparable_coverage": fraction(comparable),
            "consistent_fraction_all": fraction(agreement_counts["consistent"]),
            "opposite_fraction_all": fraction(agreement_counts["opposite"]),
            "unknown_fraction_all": fraction(agreement_counts["unknown"]),
            "consistent_fraction_comparable": agreement_counts["consistent"] / comparable if comparable else None}


def recount(frame):
    require(not frame.empty and frame.timestamp.is_unique and frame.origin_index.is_unique
            and frame.origin_index.tolist() == list(range(len(frame))), "unique ordered origin identities required")
    require(frame.timestamp.is_monotonic_increasing and not frame.cutoff.isna().any(), "origin/cutoff clock invalid")
    for arm in ARMS:
        rates(frame, arm)
    signals, current, members = [], [], []
    # Freeze every prefix result before examining any current-block outcomes.
    for block in sorted(frame.continuity_block.unique()):
        block_rows = frame.loc[frame.continuity_block.eq(block)]
        require(block_rows.cutoff.nunique() == 1, "one fixed cutoff required per block")
        cutoff = block_rows.cutoff.iloc[0]
        evaluation_year = int(block_rows.sort_values("timestamp").year.iloc[0])
        for arm in ARMS:
            for bucket in BUCKETS:
                key = dict(zip(KEYS, (int(block), cutoff, evaluation_year, arm, bucket), strict=True))
                past = prefix(frame, cutoff, block, arm, bucket)
                stats = rates(past, arm)
                signals.append({**key, **serialized_rates(stats, "past"),
                    "certain_early_majority_blocks": stats["early_majority_blocks"],
                    "certain_late_majority_blocks": stats["late_majority_blocks"],
                    "certain_early_majority_block_fraction": stats["early_majority_blocks"] / stats["blocks"] if stats["blocks"] else None,
                    "certain_late_majority_block_fraction": stats["late_majority_blocks"] / stats["blocks"] if stats["blocks"] else None,
                    "direction": signal(stats), "support_sufficient": stats["blocks"] >= 6})
                for row in past.itertuples(index=False):
                    members.append({**key, **{name: getattr(row, name) for name in MEMBER_COLUMNS
                                              if name not in KEYS and name != "prediction"},
                                    "prediction": float(getattr(row, arm))})
    for past in signals:
        arm, bucket, block = past["arm"], past["bucket"], past["evaluation_block"]
        positions = []
        for position, row in enumerate(frame.itertuples(index=False)):
            value = float(getattr(row, arm))
            if row.continuity_block == block and (value < 24 if bucket == "lt24" else value >= 24):
                positions.append(position)
        stats = rates(frame.iloc[positions], arm)
        direction = current_direction(stats)
        current.append({**{key: past[key] for key in KEYS}, **serialized_rates(stats, "current"),
                        "current_direction": direction, "agreement": agreement(past["direction"], direction)})
    signals, current, members = pd.DataFrame(signals), pd.DataFrame(current), pd.DataFrame(members, columns=MEMBER_COLUMNS)
    joined = signals.merge(current[[*KEYS, "current_direction", "agreement"]], on=KEYS, validate="one_to_one")
    summary = {"all": summary_leaf(joined),
               "by_arm": {arm: summary_leaf(joined.loc[joined.arm.eq(arm)]) for arm in ARMS},
               "by_arm_bucket": {arm: {bucket: summary_leaf(joined.loc[joined.arm.eq(arm) & joined.bucket.eq(bucket)])
                                       for bucket in BUCKETS} for arm in ARMS},
               "by_year": {str(year): summary_leaf(joined.loc[joined.evaluation_year.eq(year)]) for year in (2024, 2025)},
               "primary": summary_leaf(joined.loc[joined.arm.eq(PRIMARY)])}
    label_counts = {"exact": 0, "finite_interval": 0, "right_censored": 0}
    for row in frame.itertuples(index=False):
        key = "exact" if row.lower_bound_hours == row.upper_bound_hours else (
            "right_censored" if np.isposinf(row.upper_bound_hours) else "finite_interval")
        label_counts[key] += 1
    counts = {"origin_rows": len(frame), "evaluation_blocks": int(frame.continuity_block.nunique()),
              "arms": len(ARMS), "buckets": len(BUCKETS), "signal_rows": len(signals),
              "current_diagnostic_rows": len(current), "prefix_member_rows": len(members),
              "combinations_expected": int(frame.continuity_block.nunique()) * len(ARMS) * len(BUCKETS),
              "annual_origins": {str(year): int(frame.year.eq(year).sum()) for year in (2024, 2025)},
              "annual_blocks": {str(year): int(frame.loc[frame.year.eq(year), "continuity_block"].nunique()) for year in (2024, 2025)},
              "label_counts": label_counts}
    return signals, current, members, summary, counts


def artifact_locks(folder):
    old = bridge()
    paths = [path for path in Path(folder).rglob("*") if path.is_file()]
    return (sorted(path.relative_to(folder).as_posix() for path in paths),
            {str(path.resolve()): old.sha256(path) for path in paths})


def validate_artifact_locks(folder, inventory, hashes):
    actual_inventory, actual_hashes = artifact_locks(folder)
    old = bridge()
    old.compare_json(inventory, actual_inventory, "immutable parent inventory")
    old.compare_json(hashes, actual_hashes, "immutable parent file hashes")


def validate_output_inventory(run, expected, required):
    old = bridge()
    require(isinstance(expected, dict) and expected and set(required).issubset(expected), "required output locks missing")
    actual = {path.relative_to(run).as_posix() for path in Path(run).rglob("*")
              if path.is_file() and path != Path(run) / "report.json"}
    require(actual == set(expected), "output inventory differs")
    old.verify_hashes(expected, Path(run))


def validate_parent_evidence(evidence, report_hash, verification_hash):
    require(report_hash == PARENT_REPORT_SHA256 and verification_hash == PARENT_VERIFICATION_SHA256,
            "exact parent report/verification lock differs")
    wanted = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
              "report_sha256": PARENT_REPORT_SHA256, "verifier_sha256": PARENT_VERIFIER_SHA256,
              "reused_independent_verifier_sha256": BRIDGE_SHA256,
              "fixed_origins_verified": 701, "models_verified": 64, "refitted_predictions_verified": 1402,
              "primary_candidate": PRIMARY, "production_ready": False, "deployed": False}
    old = bridge()
    for key, value in wanted.items():
        if isinstance(value, bool):
            require(evidence.get(key) is value, "parent independent boolean " + key)
        else:
            old.compare_json(value, evidence.get(key), "parent independent evidence " + key)


def compare_table(expected, actual, name):
    require(set(expected) == set(actual), name + ": exact table schema required")
    bridge().compare_frame(expected, actual, list(expected), name)


def validate_code_locks(locks, root):
    paths = [root / EXPERIMENT / name for name in
             ("PROTOCOL.md", "direction_audit.py", "test_direction_audit.py", "verify_direction.py", "test_verify_direction.py")]
    require(all(str(path.resolve()) in locks for path in paths), "direction protocol/code lock coverage incomplete")


def validate_scope(report):
    wanted = {"schema_version": 1, "status": "pending_independent_verification", "years": [2024, 2025],
              "arms": list(ARMS), "buckets": list(BUCKETS), "primary_arm": PRIMARY,
              "configuration": CONFIGURATION, "scope": SCOPE,
              "fit_performed": False, "predict_performed": False, "calibration_performed": False,
              "deployed": False, "production_ready": False, "blind_test": False,
              "descriptive_screen_only": True, "historical_shadow_evaluation": True,
              "protected_unchanged": True, "parent_unchanged": True}
    old = bridge()
    for key, value in wanted.items():
        if isinstance(value, bool):
            require(report.get(key) is value, "declared boolean scope differs: " + key)
        else:
            old.compare_json(value, report.get(key), "declared scope " + key)
    require(not {"metrics", "gates", "yearly_acceptance_passed", "eligible_for_deployment"}.intersection(report),
            "direction counts cannot declare accuracy/deployment gates")
    for key, value in CONFIGURATION.items():
        if isinstance(value, bool):
            require(report["configuration"][key] is value, "strict configuration boolean required: " + key)


def validate_protected_inventory(locks, root, folders=PROTECTED_DIRECTORIES):
    for relative in folders:
        directory = root / relative
        paths = [str(path.resolve()) for path in directory.rglob("*") if path.is_file()] if directory.is_dir() else []
        require(paths and all(path in locks for path in paths), "protected directory missing/changed: " + relative)


def validate_parent_chain(parent, root):
    old = bridge()
    parent = Path(parent).resolve()
    require(parent == (root / PARENT_RELATIVE).resolve(), "wrong predeclared parent run")
    report = old._read_json(parent / "report.json")
    evidence = old._read_json(parent / "verification.json")
    validate_parent_evidence(evidence, old.sha256(parent / "report.json"), old.sha256(parent / "verification.json"))
    require(Path(evidence["run_dir"]).resolve() == parent and Path(report["run_dir"]).resolve() == parent,
            "parent run identity mismatch")
    old.compare_json(report["metrics"], evidence["metrics"], "passed parent metrics binding")
    inventory, hashes = artifact_locks(parent)
    require(set(inventory) == set(report["output_sha256"]) | {"report.json", "verification.json"},
            "parent artifacts missing or foreign")
    for relative, digest in report["output_sha256"].items():
        require(old.sha256(old.run_file(parent, relative)) == digest, "immutable parent output hash mismatch")
    union = old.combine_locks([report[key] for key in
                              ("original_scope_sha256", "input_sha256", "protected_sha256", "code_sha256", "environment_sha256", "parent_sha256")])
    old.compare_json(union, report["frozen_sha256"], "parent frozen union")
    require(len(union) == 528, "predeclared parent 528-file scope differs")
    old.verify_hashes(union, root)
    old._verify_environment(old._read_json(parent / "environment.json"), report["environment_sha256"])
    return report, inventory, hashes


def rebuild_origins(root, parent):
    old = bridge()
    grid = old.rebuild_grid({key: pd.read_parquet(root / path) for key, path in old.RAW_SOURCES.items()})
    archive = pd.read_parquet(root / "artifacts/strict_history_backtest/predictions.parquet")
    archive["timestamp"] = pd.to_datetime(archive.timestamp, utc=True).dt.tz_convert("Asia/Shanghai")
    archive = archive.loc[archive.timestamp.dt.year.isin((2024, 2025))].sort_values("timestamp").reset_index(drop=True)
    evaluation = old.rebuild_evaluation(archive, grid)
    require(len(grid) == 102024 and len(evaluation) == 701 and evaluation.continuity_block.nunique() == 32,
            "fixed original grid/origin/block population differs")
    old.compare_frame(evaluation, pd.read_parquet(parent / "evaluation.parquet"), list(evaluation), "raw reconstructed parent evaluation")
    predictions = pd.read_parquet(parent / "predictions.parquet")
    old.compare_frame(evaluation, predictions, list(evaluation), "raw reconstructed parent prediction labels/cutoffs")
    result = evaluation.rename(columns={"eval_row": "origin_index"}).copy()
    for arm in ARMS:
        result[arm] = predictions[arm].to_numpy(dtype=float)
    old.compare_array(result.archive.to_numpy(), evaluation.archive_prediction_hours.to_numpy(), "original archive values")
    return result, len(grid)


def validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes):
    old = bridge()
    groups = ("input_sha256", "protected_sha256", "code_sha256", "environment_sha256", "parent_sha256")
    require(all(isinstance(report.get(key), dict) and report[key] for key in (*groups, "frozen_sha256", "output_sha256")),
            "nonempty provenance groups required")
    old.compare_json(parent_inventory, report.get("parent_artifact_inventory"), "declared parent inventory")
    old.compare_json(parent_hashes, report["parent_sha256"], "declared parent complete file locks")
    old.compare_json(parent_report["frozen_sha256"], report.get("original_scope_sha256"), "inherited 528-file scope")
    for key in ("input_sha256", "protected_sha256", "environment_sha256", "historical_input_sha256", "historical_audit_drift"):
        old.compare_json(parent_report[key], report.get(key), "intact inherited " + key)
    require(all(report["code_sha256"].get(path) == digest for path, digest in parent_report["code_sha256"].items()),
            "inherited code hash mismatch")
    validate_code_locks(report["code_sha256"], root)
    new_code = {str((root / EXPERIMENT / name).resolve()) for name in
                ("PROTOCOL.md", "direction_audit.py", "test_direction_audit.py", "verify_direction.py", "test_verify_direction.py")}
    require(set(report["code_sha256"]) == set(parent_report["code_sha256"]) | new_code, "foreign/missing new code locks")
    require(Path(report["parent_run_dir"]).resolve() == parent and Path(report["run_dir"]).resolve() == run,
            "declared run identity differs")
    wanted = {"parent_report_sha256": PARENT_REPORT_SHA256, "parent_verification_sha256": PARENT_VERIFICATION_SHA256,
              "parent_verification_report_sha256": PARENT_REPORT_SHA256, "parent_verifier_sha256": PARENT_VERIFIER_SHA256}
    for key, value in wanted.items():
        old.compare_json(value, report.get(key), key)
    union = old.combine_locks([parent_report["frozen_sha256"], *[report[key] for key in groups]])
    old.compare_json(union, report["frozen_sha256"], "complete inherited/current/parent frozen union")
    old.verify_hashes(union, root)
    validate_protected_inventory(report["protected_sha256"], root)
    outputs = {"signals.parquet", "current_diagnostics.parquet", "prefix_members.parquet",
               "environment.json", "execution_record_at_start.md"}
    require(set(report["output_sha256"]) == outputs, "fixed direction outputs only; no new models or predictions")
    validate_output_inventory(run, report["output_sha256"], outputs)
    old.compare_json(old._read_json(parent / "environment.json"), old._read_json(run / "environment.json"), "unchanged environment metadata")
    old._verify_environment(old._read_json(run / "environment.json"), report["environment_sha256"])
    validate_scope(report)
    return union


def verify_run(run, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    old = bridge()
    run = old.validate_run_directory(run, root / EXPERIMENT)
    require(not (run / "verification.json").exists(), "verification.json already exists; no overwrite")
    parent = (root / PARENT_RELATIVE).resolve()
    parent_report, parent_inventory, parent_hashes = validate_parent_chain(parent, root)
    report_hash = old.sha256(run / "report.json")
    report = old._read_json(run / "report.json")
    frozen = validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes)
    frame, grid_rows = rebuild_origins(root, parent)
    signals, current, members, summary, counts = recount(frame)
    require(counts["signal_rows"] == counts["current_diagnostic_rows"] == counts["combinations_expected"] == 192,
            "all 192 fixed combinations required")
    compare_table(signals, pd.read_parquet(run / "signals.parquet"), "every independently counted prefix signal")
    compare_table(current, pd.read_parquet(run / "current_diagnostics.parquet"), "every independent current diagnostic")
    compare_table(members, pd.read_parquet(run / "prefix_members.parquet"), "all independently rebuilt mature prefix members")
    old.compare_json(summary, report.get("summary"), "complete explicit-denominator summaries")
    old.compare_json(counts, report.get("counts"), "complete fixed population/combination counts")
    require(isinstance(report.get("limits"), list) and report["limits"] and all(isinstance(item, str) for item in report["limits"]),
            "nonempty descriptive limits required")
    old.verify_hashes(frozen, root)
    validate_artifact_locks(parent, parent_inventory, parent_hashes)
    validate_protected_inventory(report["protected_sha256"], root)
    old._verify_environment(old._read_json(run / "environment.json"), report["environment_sha256"])
    require(old.sha256(run / "report.json") == report_hash, "direction report changed during verification")
    validate_output_inventory(run, report["output_sha256"], set(report["output_sha256"]))
    result = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
              "run_dir": str(run), "report_sha256": report_hash, "verifier_sha256": old.sha256(__file__),
              "reused_independent_verifier_sha256": BRIDGE_SHA256,
              "parent_report_sha256": PARENT_REPORT_SHA256, "parent_verification_sha256": PARENT_VERIFICATION_SHA256,
              "primary_arm": PRIMARY, "grid_rows_reconstructed": grid_rows,
              "fixed_origins_verified": len(frame), "signal_rows_verified": len(signals),
              "current_diagnostic_rows_verified": len(current), "prefix_member_rows_verified": len(members),
              "counts": counts, "summary": summary, "scope": SCOPE,
              "fit_performed": False, "predict_performed": False, "calibration_performed": False,
              "production_ready": False, "deployed": False, "descriptive_screen_only": True,
              "limits": ["Direction counts independently use explicit row and continuity-block loops, not the new runner",
                         "Raw canonical labels/cutoffs reuse the reviewed frozen independent bridge mathematics",
                         "Only already verified 2024/2025 out-of-sample saved predictions supply history",
                         "Interval direction bounds are observational compatibility bounds, not confidence intervals",
                         "Fixed descriptive signals do not prove correction magnitude or any MAE improvement",
                         "No training, new model prediction, calibration, acceptance gate or deployment approval"]}
    old.write_exclusive_json(run / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing approved direction-audit run")
    args = parser.parse_args()
    old = bridge()
    run = old.validate_run_directory(args.run, Path(__file__).resolve().parent)
    try:
        result = verify_run(run)
    except Exception as exc:
        if not (run / "verification.json").exists():
            old.write_exclusive_json(run / "verification.json", {"schema_version": 1, "status": "failed",
                "independent_verification_passed": False, "error": str(exc), "verifier_sha256": old.sha256(__file__)})
        raise
    print(json.dumps({key: result[key] for key in ("status", "fixed_origins_verified", "signal_rows_verified", "production_ready")}))


if __name__ == "__main__":
    main()
