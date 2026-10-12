"""Verify the all-calendar population with frozen independent bridge helpers.

Population and strict mature membership are constructed here without importing
any runner. Reused label/feature/scoring mathematics are identified by SHA256;
this is not a newly written second implementation of those mathematics.
"""

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
EXPERIMENT = "experiments/web_recovery_calendar_population_20261003"
INHERITED_ARMS = ("archive", "interval_candidate", "grid_all_legacy_origins")
PRIMARY = "calendar_all_hours"
ARMS = (*INHERITED_ARMS, PRIMARY)
PAIRS = {f"{PRIMARY}_vs_{reference}": (reference, PRIMARY) for reference in INHERITED_ARMS}
META_COLUMNS = ["population_row", "timestamp", "year", "continuity_block", "pollution_state",
                "input_hour_observed", "eligible", "aqi_category", "label_type"]
LABEL_COLUMNS = ["lower_bound_hours", "upper_bound_hours", "exact_hours", "informative", "label_available_at", "reason"]
CONFIGURATION = {"seed": 42, "num_boost_round": 500, "nthread": 1}
SCOPE = "Single full-calendar training population change; fixed 701 polluted evaluation origins"
WEIGHT_POLICY = "Unit weight per mature calendar hour; no resampling or extra weights"
RIGHT_CENSOR_POLICY = "global archive-end labels only; all mature rows retained"
REUSED_MAIN_HELPER_SHA256 = "3684ee50916a426f44ed9c6791a8a80fc1f017c82b587dc7d45cb728d081e6f3"
PROTECTED_DIRECTORIES = ("artifacts/cnemc_2025_deployment_candidate", "artifacts/strict_history_backtest",
                         "project-showcase", "sites-haze/dist", "releases/runtime_20261002")


def require(value, message):
    if not value:
        raise ValueError(message)


def bridge():
    path = Path(__file__).resolve().parent.parent / "web_recovery_bridge_20261002/verify_bridge.py"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == BRIDGE_SHA256, "frozen independent bridge changed")
    spec = importlib.util.spec_from_file_location("calendar_frozen_independent_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calendar_population(grid, names):
    times = pd.DatetimeIndex(grid.timestamp)
    require(len(times) > 0 and times.is_unique and times.equals(pd.date_range(times[0], times[-1], freq="h")),
            "all-calendar population requires complete unique ordered hour grid")
    require(len(names) == len(set(names)) and "aqi" in names, "unique feature names including AQI required")
    result = grid[["timestamp", "continuity_block", "pollution_state", "input_hour_observed", "eligible", *names]].copy().reset_index(drop=True)
    result["population_row"] = np.arange(len(grid), dtype=np.int64)
    result["year"] = result.timestamp.dt.year
    aqi = pd.to_numeric(result.aqi, errors="raise").to_numpy(dtype=float)
    require(not np.isinf(aqi).any(), "infinite AQI")
    result["aqi_category"] = np.where(np.isnan(aqi), "missing", np.where(aqi > 100, "polluted", "clean"))
    bounds = bridge().recover_bounds(grid, grid.timestamp)
    for name in LABEL_COLUMNS:
        result[name] = bounds[name].to_numpy() if name != "label_available_at" else bounds[name]
    result["label_type"] = np.where(result.lower_bound_hours.eq(result.upper_bound_hours), "exact",
                                    np.where(np.isposinf(result.upper_bound_hours), "right_censored", "interval"))
    require(np.array_equal(result.label_type.to_numpy(), result.reason.to_numpy()), "canonical label type/reason mismatch")
    return result[[*META_COLUMNS, *names, *LABEL_COLUMNS]]


def training_members(population, record):
    cutoff, block = pd.Timestamp(record["cutoff"]), int(record["continuity_block"])
    mask = population.timestamp.lt(cutoff) & population.label_available_at.lt(cutoff) & population.continuity_block.ne(block)
    result = population.loc[mask, [*META_COLUMNS, *LABEL_COLUMNS]].copy()
    result["model_id"], result["arm"] = record["model_id"], record["arm"]
    result["evaluation_block"], result["cutoff"] = block, cutoff
    return result


def category_counts(frame):
    result = dict.fromkeys(("clean", "polluted", "missing"), 0)
    for value in frame.aqi_category:
        require(value in result, "unknown AQI category")
        result[value] += 1
    return result


def label_counts(frame):
    result = dict.fromkeys(("exact", "interval", "right_censored"), 0)
    for value in frame.label_type:
        require(value in result, "unknown label type")
        result[value] += 1
    return result


def exact_one_stats(frame):
    count = int(frame.exact_hours.eq(1.).sum())
    return {"exact_t1_rows": count, "exact_t1_fraction": count / len(frame) if len(frame) else None}


def validate_common_population(population, parent_population, names):
    require(population.timestamp.is_unique and parent_population.timestamp.is_unique, "shared population identities must be unique")
    positions = pd.DatetimeIndex(population.timestamp).get_indexer(parent_population.timestamp)
    require((positions >= 0).all(), "old population is not contained in full calendar")
    bridge().compare_frame(parent_population, population.iloc[positions],
                           ["timestamp", "continuity_block", *names, *LABEL_COLUMNS], "all unchanged shared old X/bounds/maturity")


def validate_model_parameters(record, names):
    old = bridge()
    require(len(names) == len(set(names)) == 87, "all 87 unique ordered features required")
    for key in ("feature_names", "booster_feature_names"):
        old.compare_json(names, record[key], "fitted model " + key)
    old.compare_json(CONFIGURATION, record["configuration"], "fixed model configuration")
    old.compare_json(old._fixed_parameters(), record["aft_parameters"], "fixed AFT parameters")
    require(record["num_boosted_rounds"] == 500, "actual fitted rounds differ from 500")
    old.validate_trained_config(record["actual_trained_booster_config"], 87)


def observed_counts(frame):
    require(frame.input_hour_observed.isin((True, False)).all(), "binary observed-hour flags required")
    observed = int(frame.input_hour_observed.eq(True).sum())
    return {"observed": observed, "unobserved": len(frame) - observed}


def eligible_counts(frame):
    require(frame.eligible.isin((True, False)).all(), "binary eligibility metadata required")
    eligible = int(frame.eligible.eq(True).sum())
    return {"eligible": eligible, "ineligible": len(frame) - eligible}


def member_statistics(frame, names, parent_population):
    shared = int(frame.timestamp.isin(parent_population.timestamp).sum())
    return {"label_counts": label_counts(frame), "aqi_category_counts": category_counts(frame),
            "input_hour_observed_counts": observed_counts(frame), "causal_eligible_counts": eligible_counts(frame),
            **exact_one_stats(frame), "shared_legacy_member_rows": shared,
            "additional_calendar_member_rows": len(frame) - shared,
            "train_feature_missingness": dict(zip(names, np.isnan(frame[names].to_numpy(dtype=float)).mean(axis=0).tolist(), strict=True))}


def validate_common_members(members, parent_population, parent_members, record):
    shared = members.loc[members.timestamp.isin(parent_population.timestamp)].sort_values("timestamp")
    prior = parent_members.loc[parent_members.arm.eq("grid_all_legacy_origins")
                               & parent_members.evaluation_block.eq(int(record["continuity_block"]))].sort_values("timestamp")
    require(not shared.timestamp.duplicated().any() and not prior.timestamp.duplicated().any(), "duplicate shared train identities")
    require(prior.cutoff.eq(pd.Timestamp(record["cutoff"])).all(), "old shared training cutoff differs")
    bridge().compare_frame(prior, shared, ["timestamp", "continuity_block", *LABEL_COLUMNS], "all shared old mature training members")
    return len(shared)


def reload_predictions(run, record, features, names, output_hashes):
    old = bridge()
    validate_model_parameters(record, names)
    path = old.run_file(run, record["model_path"])
    require(output_hashes.get(record["model_path"]) == record["model_sha256"] == old.sha256(path), "trusted local native model hash differs")
    old._validate_native_trees(path, 87)
    import xgboost as xgb

    booster = xgb.Booster(params={"nthread": 1})
    booster.load_model(path)
    require(booster.num_boosted_rounds() == 500 and booster.num_features() == 87, "loaded native round/feature count differs")
    old.compare_json(names, booster.feature_names, "loaded ordered feature names")
    objective = json.loads(booster.save_config())["learner"]["objective"]
    require(objective["name"] == "survival:aft" and objective["aft_loss_param"]["aft_loss_distribution"] == "normal",
            "loaded AFT objective/distribution differs")
    require(np.isclose(float(objective["aft_loss_param"]["aft_loss_distribution_scale"]), 1.2, rtol=1e-6), "loaded AFT scale differs")
    query = pd.DataFrame(features, columns=names)
    result = booster.predict(xgb.DMatrix(query, missing=np.nan))
    require(np.asarray(result).shape == (len(query),) and np.isfinite(result).all() and (result >= 0).all(), "invalid reloaded predictions")
    return result


def validate_inventory(records, evaluation):
    wanted = {(PRIMARY, int(block)) for block in evaluation.continuity_block.unique()}
    got = [(record["arm"], int(record["continuity_block"])) for record in records]
    require(len(got) == len(wanted) and set(got) == wanted, "one calendar model per fixed evaluation block required")
    ids = [record["model_id"] for record in records]
    require(len(ids) == len(set(ids)), "duplicated calendar model identity")
    for record in records:
        require(record["model_id"] == f"{PRIMARY}_b{record['continuity_block']}", "calendar model identity mismatch")


def compute_scores(evaluation, predictions, years=(2024, 2025)):
    old = bridge()
    require(set(predictions) == set(ARMS), "all four fixed prediction arms required")

    def subset(frame, values):
        return {"arms": {arm: old.score_predictions(frame, values[arm]) for arm in ARMS},
                "paired": {key: old.paired_summary(frame, values[base], values[candidate])
                           for key, (base, candidate) in PAIRS.items()}}

    annual, gates = {}, {}
    for year in years:
        mask = evaluation.year.eq(year).to_numpy()
        metrics = subset(evaluation.loc[mask], {arm: np.asarray(values)[mask] for arm, values in predictions.items()})
        annual[str(year)] = metrics
        gate = old.acceptance_gate(metrics["arms"]["archive"], metrics["arms"][PRIMARY], metrics["paired"][f"{PRIMARY}_vs_archive"])
        gate["checks"]["coverage_100"] = all(item["prediction_coverage"] == 1. for item in metrics["arms"].values())
        gate["passed"] = all(gate["checks"].values())
        gate["failure_reasons"] = [key for key, passed in gate["checks"].items() if not passed]
        gates[str(year)] = gate
    scores = {"annual": annual, "pooled": subset(evaluation, predictions),
              "gates": {"annual": gates, "passed": all(item["passed"] for item in gates.values()),
                        "failure_reasons": [f"{year}:{key}" for year in years for key in gates[str(year)]["failure_reasons"]]}}
    processes = []
    for (year, block), frame in evaluation.groupby(["year", "continuity_block"], sort=True):
        for arm in ARMS:
            score = old.score_predictions(frame, np.asarray(predictions[arm])[frame.index.to_numpy()])
            record = {"year": int(year), "continuity_block": int(block), "arm": arm, "old_events": int(frame.event_key.nunique()),
                      **{key: value for key, value in score.items() if key != "subgroups"}}
            for name, subgroup in score["subgroups"].items():
                record.update({f"{name}_{key}": value for key, value in subgroup.items()})
            processes.append(record)
    return scores, pd.DataFrame(processes)


def prediction_rows(evaluation, predictions):
    old = bridge()
    require(set(predictions) == set(ARMS), "all four prediction arms required")
    result = evaluation.copy()
    for arm in ARMS:
        prediction = np.asarray(predictions[arm], dtype=float)
        old._validate_score_inputs(evaluation, prediction)
        result[arm] = prediction
        result[f"{arm}_error_lower"] = np.maximum(np.maximum(evaluation.lower_bound_hours - prediction, prediction - evaluation.upper_bound_hours), 0)
        result[f"{arm}_error_upper"] = np.maximum(abs(evaluation.lower_bound_hours - prediction), abs(evaluation.upper_bound_hours - prediction))
        result[f"{arm}_exact_error"] = abs(prediction - evaluation.exact_hours)
        result[f"{arm}_signed_error"] = prediction - evaluation.exact_hours
    for key, (base, candidate) in PAIRS.items():
        result[key + "_delta_lower"], result[key + "_delta_upper"] = old.paired_bounds(evaluation, predictions[base], predictions[candidate])
    return result


def artifact_locks(folder):
    old = bridge()
    paths = [path for path in Path(folder).rglob("*") if path.is_file()]
    return (sorted(path.relative_to(folder).as_posix() for path in paths),
            {str(path.resolve()): old.sha256(path) for path in paths})


def validate_artifact_locks(folder, inventory, hashes):
    actual_inventory, actual_hashes = artifact_locks(folder)
    old = bridge()
    old.compare_json(inventory, actual_inventory, "immutable parent inventory")
    old.compare_json(hashes, actual_hashes, "immutable parent hashes")


def validate_output_inventory(run, expected, required):
    old = bridge()
    require(isinstance(expected, dict) and expected and set(required).issubset(expected), "required output locks missing")
    actual = {path.relative_to(run).as_posix() for path in Path(run).rglob("*")
              if path.is_file() and path != Path(run) / "report.json"}
    require(actual == set(expected), "output inventory differs")
    for name, digest in expected.items():
        require(old.sha256(old.run_file(run, name)) == digest, "output hash mismatch: " + name)


def validate_protected_inventory(locks, root, folders=PROTECTED_DIRECTORIES):
    for relative in folders:
        directory = (root / relative).resolve()
        actual = {str(path.resolve()) for path in directory.rglob("*") if path.is_file()} if directory.is_dir() else set()
        expected = {name for name in locks if Path(name).resolve().is_relative_to(directory)}
        require(actual and actual == expected, "nonempty exact protected inventory differs: " + relative)


def validate_parent_evidence(evidence, report_hash, verification_hash):
    require(report_hash == PARENT_REPORT_SHA256 and verification_hash == PARENT_VERIFICATION_SHA256,
            "exact parent report/verification locks differ")
    expected = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
                "report_sha256": PARENT_REPORT_SHA256, "verifier_sha256": PARENT_VERIFIER_SHA256,
                "reused_independent_verifier_sha256": BRIDGE_SHA256,
                "fixed_origins_verified": 701, "models_verified": 64, "refitted_predictions_verified": 1402,
                "primary_candidate": "grid_all_legacy_origins", "production_ready": False, "deployed": False}
    old = bridge()
    for key, value in expected.items():
        if isinstance(value, bool):
            require(evidence.get(key) is value, "parent boolean evidence " + key)
        else:
            old.compare_json(value, evidence.get(key), "parent passed evidence " + key)


def validate_parent_chain(parent, root):
    old = bridge()
    parent = Path(parent).resolve()
    require(parent == (root / PARENT_RELATIVE).resolve(), "wrong fixed parent run")
    report, evidence = old._read_json(parent / "report.json"), old._read_json(parent / "verification.json")
    validate_parent_evidence(evidence, old.sha256(parent / "report.json"), old.sha256(parent / "verification.json"))
    require(Path(report["run_dir"]).resolve() == Path(evidence["run_dir"]).resolve() == parent, "parent run identity differs")
    old.compare_json(report["metrics"], evidence["metrics"], "parent passed metric binding")
    inventory, hashes = artifact_locks(parent)
    require(len(inventory) == 77 and set(inventory) == set(report["output_sha256"]) | {"report.json", "verification.json"},
            "fixed complete 77 parent artifacts differ")
    for relative, digest in report["output_sha256"].items():
        require(old.sha256(old.run_file(parent, relative)) == digest, "parent immutable output hash mismatch")
    union = old.combine_locks([report[key] for key in
                              ("original_scope_sha256", "input_sha256", "protected_sha256", "code_sha256", "environment_sha256", "parent_sha256")])
    require(len(union) == 528, "predeclared parent frozen 528 files differ")
    old.compare_json(union, report["frozen_sha256"], "parent complete frozen union")
    old.verify_hashes(union, root)
    validate_protected_inventory(report["protected_sha256"], root)
    old._verify_environment(old._read_json(parent / "environment.json"), report["environment_sha256"])
    return report, inventory, hashes


def validate_scope(report):
    expected = {"schema_version": 1, "status": "pending_independent_verification", "years": [2024, 2025],
                "configuration": CONFIGURATION, "aft_parameters": bridge()._fixed_parameters(),
                "primary_candidate": PRIMARY, "new_refitted_arms": [PRIMARY], "inherited_arms": list(INHERITED_ARMS),
                "arms": list(ARMS), "scope": SCOPE, "weight_policy": WEIGHT_POLICY,
                "historical_shadow_evaluation": True, "archive_is_primary_reference": True,
                "blind_test": False, "deployed": False, "production_ready": False,
                "protected_unchanged": True, "parent_unchanged": True,
                "shared_legacy_members_unchanged": True, "inherited_predictions_unchanged": True}
    old = bridge()
    for key, value in expected.items():
        if isinstance(value, bool):
            require(report.get(key) is value, "declared boolean scope differs: " + key)
        else:
            old.compare_json(value, report.get(key), "declared scope " + key)


def validate_inherited_predictions(parent, current):
    old = bridge()
    old.compare_frame(parent, current, ["timestamp", "year", "continuity_block", *INHERITED_ARMS], "all unchanged inherited predictions")


def validate_feature_archive(stored, names, timestamps, features, parent=False):
    old = bridge()
    key = "grid_all_legacy_origins" if parent else "candidate"
    expected = {"timestamps", "feature_names", key} | ({"labels_only"} if parent else set())
    require(set(stored.files) == expected, "exact parent/child evaluation NPZ schema differs")
    old.compare_array(np.asarray(names), stored["feature_names"], "NPZ ordered features")
    old.compare_array(np.asarray(timestamps, dtype=np.int64), stored["timestamps"], "NPZ exact UTCns origin clock")
    old.compare_array(features, stored[key], "all unchanged/rebuilt NPZ query vectors")
    if parent:
        require(stored["labels_only"].shape == features.shape, "parent saved legacy-vector shape differs")


def validate_code_locks(locks, root):
    paths = [root / EXPERIMENT / name for name in
             ("PROTOCOL.md", "calendar_runner.py", "test_calendar_runner.py", "verify_calendar.py", "test_verify_calendar.py")]
    require(all(str(path.resolve()) in locks for path in paths), "calendar protocol/code lock coverage incomplete")


def validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes):
    old = bridge()
    groups = ("input_sha256", "protected_sha256", "code_sha256", "environment_sha256", "parent_sha256")
    require(all(isinstance(report.get(key), dict) and report[key] for key in (*groups, "frozen_sha256", "output_sha256")),
            "nonempty provenance groups required")
    old.compare_json(parent_inventory, report.get("parent_artifact_inventory"), "declared exact parent inventory")
    old.compare_json(parent_hashes, report["parent_sha256"], "declared complete parent file locks")
    old.compare_json(parent_report["frozen_sha256"], report.get("original_scope_sha256"), "inherited exact 528-file scope")
    for key in ("input_sha256", "protected_sha256", "environment_sha256", "historical_input_sha256", "historical_audit_drift"):
        old.compare_json(parent_report[key], report.get(key), "unchanged inherited " + key)
    require(all(report["code_sha256"].get(path) == digest for path, digest in parent_report["code_sha256"].items()), "inherited code hash differs")
    validate_code_locks(report["code_sha256"], root)
    new_code = {str((root / EXPERIMENT / name).resolve()) for name in
                ("PROTOCOL.md", "calendar_runner.py", "test_calendar_runner.py", "verify_calendar.py", "test_verify_calendar.py")}
    require(set(report["code_sha256"]) == set(parent_report["code_sha256"]) | new_code, "foreign/missing new code locks")
    require(Path(report["parent_run_dir"]).resolve() == parent and Path(report["run_dir"]).resolve() == run, "declared run identity differs")
    wanted = {"parent_report_sha256": PARENT_REPORT_SHA256, "parent_verification_sha256": PARENT_VERIFICATION_SHA256,
              "parent_verification_report_sha256": PARENT_REPORT_SHA256, "parent_verifier_sha256": PARENT_VERIFIER_SHA256,
              "reused_main_helper_sha256": REUSED_MAIN_HELPER_SHA256}
    for key, value in wanted.items():
        old.compare_json(value, report.get(key), key)
    for key in ("target", "uncertainty_scope"):
        old.compare_json(parent_report[key], report.get(key), "unchanged target " + key)
    union = old.combine_locks([parent_report["frozen_sha256"], *[report[key] for key in groups]])
    old.compare_json(union, report["frozen_sha256"], "complete inherited/current/parent frozen union")
    old.verify_hashes(union, root)
    validate_protected_inventory(report["protected_sha256"], root)
    required = {"evaluation.parquet", "calendar_population.parquet", "features.npz", "training_members.parquet", "models.json",
                "predictions.parquet", "process_scores.parquet", "scores.json", "environment.json", "execution_record_at_start.md"}
    validate_output_inventory(run, report["output_sha256"], required)
    old.compare_json(old._read_json(parent / "environment.json"), old._read_json(run / "environment.json"), "unchanged environment metadata")
    old._verify_environment(old._read_json(run / "environment.json"), report["environment_sha256"])
    validate_scope(report)
    return union


def verify_models(run, report, models, members, evaluation, grid, population, parent_population, parent_members, features, names):
    old = bridge()
    validate_inventory(models["models"], evaluation)
    require(models.get("schema_version") == 1 and len(models["models"]) == report["fitted_models"] == 32, "all 32 calendar models required")
    old.compare_json(names, models["feature_names"], "all-model feature order")
    require(set(members.model_id) == {record["model_id"] for record in models["models"]}, "foreign/missing training member model IDs")
    require(set(members) == set(META_COLUMNS + LABEL_COLUMNS + ["model_id", "arm", "evaluation_block", "cutoff"]), "exact member metadata schema required")
    expected_outputs = {"evaluation.parquet", "calendar_population.parquet", "features.npz", "training_members.parquet", "models.json",
                        "predictions.parquet", "process_scores.parquet", "scores.json", "environment.json", "execution_record_at_start.md"}
    expected_outputs.update(f"models/{record['model_id']}.json" for record in models["models"])
    require(set(report["output_sha256"]) == expected_outputs, "exact new model/output inventory required")
    saved = pd.read_parquet(run / "predictions.parquet")
    prediction = np.full(len(evaluation), np.nan)
    checked, shared_checked = 0, 0
    maturity = []
    for record in models["models"]:
        block = int(record["continuity_block"])
        cutoff = grid.loc[grid.continuity_block.eq(block), "timestamp"].min()
        require(pd.Timestamp(record["cutoff"]) == cutoff, "new model cutoff is not full-grid block start")
        wanted = training_members(population, record)
        require(not wanted.empty, "empty mature calendar population")
        actual = members.loc[members.model_id.eq(record["model_id"])]
        old.compare_frame(wanted, actual, list(wanted), "every independent calendar train member " + record["model_id"])
        expected = population.iloc[wanted.population_row.to_numpy(dtype=np.int64)]
        positions = np.flatnonzero(evaluation.continuity_block.eq(block).to_numpy())
        checked += len(expected)
        require(record["train_rows"] == len(expected) and record["eval_rows"] == len(positions), "model member/evaluation count differs")
        require(record["training_member_path"] == "training_members.parquet" and record["model_path"] == f"models/{record['model_id']}.json",
                "model path/member source differs")
        require(pd.Timestamp(record["training_last_timestamp"]) == expected.timestamp.max()
                and pd.Timestamp(record["max_label_available_at"]) == expected.label_available_at.max(), "mature training extrema differ")
        stats = member_statistics(expected, names, parent_population)
        for key, value in stats.items():
            old.compare_json(value, record.get(key), "independent model member statistics " + key)
        shared = validate_common_members(wanted, parent_population, parent_members, record)
        require(shared == stats["shared_legacy_member_rows"], "shared member count differs")
        shared_checked += shared
        require(record["training_features_sha256"] == old.matrix_digest(expected[names].to_numpy(dtype=float)), "training 87-feature matrix digest differs")
        times_hash = hashlib.sha256(pd.DatetimeIndex(expected.timestamp).asi8.astype("<i8").tobytes()).hexdigest()
        require(record["training_timestamps_sha256"] == times_hash, "training exact timestamp digest differs")
        require(record.get("weight_policy") == WEIGHT_POLICY and record.get("right_censor_policy") == RIGHT_CENSOR_POLICY,
                "unit-weight or archive-end censor policy differs")
        require(record.get("reload_predictions_equal") is True, "original native reload equality was not established")
        values = reload_predictions(run, record, features[positions], names, report["output_sha256"])
        old.compare_array(values, saved.loc[positions, PRIMARY].to_numpy(dtype=float), "all independent native reloaded predictions " + record["model_id"])
        prediction[positions] = values
        summary_keys = ("model_id", "train_rows", "label_counts", "aqi_category_counts", "input_hour_observed_counts", "causal_eligible_counts",
                        "exact_t1_rows", "exact_t1_fraction", "shared_legacy_member_rows", "additional_calendar_member_rows",
                        "training_last_timestamp", "max_label_available_at")
        maturity.append({key: record[key] for key in summary_keys})
    require(checked == len(members) and np.isfinite(prediction).all(), "unused members or unpredicted original evaluation rows")
    old.compare_json(maturity, report["training_maturity_summary"], "every independently verified per-cutoff maturity summary")
    return prediction, checked, shared_checked


def verify_run(run, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    old = bridge()
    run = old.validate_run_directory(run, root / EXPERIMENT)
    require(not (run / "verification.json").exists(), "verification.json exists; exclusive output cannot overwrite")
    parent = (root / PARENT_RELATIVE).resolve()
    parent_report, parent_inventory, parent_hashes = validate_parent_chain(parent, root)
    report_hash = old.sha256(run / "report.json")
    report = old._read_json(run / "report.json")
    frozen = validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes)
    names = old._read_json(root / "artifacts/cnemc_2025_deployment_candidate/model_metadata.json")["feature_names"]
    require(len(names) == len(set(names)) == 87, "all original 87 ordered features required")
    old.compare_json(names, report["feature_names"], "report feature order")
    grid = old.rebuild_grid({key: pd.read_parquet(root / path) for key, path in old.RAW_SOURCES.items()})
    grid_columns = ["timestamp", *names, "continuity_block", "episode_id", "eligible", "pollution_state", "input_hour_observed"]
    old.compare_frame(grid, pd.read_parquet(root / "artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet"),
                      grid_columns, "complete independent five-measurement hourly grid")
    population = calendar_population(grid, names)
    require(len(grid) == len(population) == 102024, "no all-calendar origins may be removed")
    saved_population = pd.read_parquet(run / "calendar_population.parquet")
    require(set(population) == set(saved_population), "exact full population schema required")
    old.compare_frame(population, saved_population, list(population), "all 102024 original hour vectors/bounds/state metadata")
    parent_population = pd.read_parquet(parent / "grid_population.parquet")
    require(len(parent_population) == 8638, "all shared original legacy origins required")
    validate_common_population(population, parent_population, names)
    archive = pd.read_parquet(root / "artifacts/strict_history_backtest/predictions.parquet")
    archive["timestamp"] = pd.to_datetime(archive.timestamp, utc=True).dt.tz_convert("Asia/Shanghai")
    archive = archive.loc[archive.timestamp.dt.year.isin((2024, 2025))].sort_values("timestamp").reset_index(drop=True)
    evaluation = old.rebuild_evaluation(archive, grid)
    require(len(evaluation) == 701 and evaluation.continuity_block.nunique() == 32, "same 701 polluted evaluation origins/32 blocks required")
    for folder in (parent, run):
        old.compare_frame(evaluation, pd.read_parquet(folder / "evaluation.parquet"), list(evaluation), "same canonical original evaluation population")
    annual = {str(year): {"rows": int(evaluation.year.eq(year).sum()),
              "old_events": int(evaluation.loc[evaluation.year.eq(year), "event_key"].nunique()),
              "blocks": int(evaluation.loc[evaluation.year.eq(year), "continuity_block"].nunique()),
              "causal_eligible": int(evaluation.loc[evaluation.year.eq(year), "causal_eligible"].sum()),
              "labels": old._label_counts(evaluation.loc[evaluation.year.eq(year)])} for year in (2024, 2025)}
    old.compare_json(parent_report["counts"]["annual"], annual, "same fixed annual evaluation rows/labels")
    exact_t1 = exact_one_stats(population)
    counts = {"grid_rows": len(grid), "calendar_population": len(population), "evaluation_rows": len(evaluation),
              "evaluation_blocks": int(evaluation.continuity_block.nunique()), "planned_fits": 32, "annual": annual,
              "calendar_population_label_counts": label_counts(population), "calendar_population_aqi_category_counts": category_counts(population),
              "calendar_population_input_hour_observed_counts": observed_counts(population), "calendar_population_eligible_counts": eligible_counts(population),
              "calendar_population_exact_t1_rows": exact_t1["exact_t1_rows"], "calendar_population_exact_t1_fraction": exact_t1["exact_t1_fraction"]}
    old.compare_json(counts, report["counts"], "independent complete calendar/evaluation population counts")
    features = grid.set_index("timestamp").loc[evaluation.timestamp, names].to_numpy(dtype=float)
    with np.load(run / "features.npz", allow_pickle=False) as stored:
        validate_feature_archive(stored, names, pd.DatetimeIndex(evaluation.timestamp).asi8, features)
    with np.load(parent / "features.npz", allow_pickle=False) as stored:
        validate_feature_archive(stored, names, pd.DatetimeIndex(evaluation.timestamp).asi8, features, parent=True)
    inherited = pd.read_parquet(parent / "predictions.parquet")
    saved = pd.read_parquet(run / "predictions.parquet")
    old.compare_frame(evaluation, inherited, list(evaluation), "parent prediction canonical evaluation labels")
    old.compare_frame(evaluation, saved, list(evaluation), "child prediction canonical evaluation labels")
    validate_inherited_predictions(inherited, saved)
    old.compare_array(inherited.archive.to_numpy(dtype=float), evaluation.archive_prediction_hours, "immutable original archive values")
    predictions = {arm: inherited[arm].to_numpy(dtype=float) for arm in INHERITED_ARMS}
    models = old._read_json(run / "models.json")
    members = pd.read_parquet(run / "training_members.parquet")
    inferred, checked, shared_checked = verify_models(run, report, models, members, evaluation, grid, population, parent_population,
                                                    pd.read_parquet(parent / "training_members.parquet"), features, names)
    predictions[PRIMARY] = inferred
    old.compare_json({PRIMARY: dict(zip(names, np.isnan(features).mean(axis=0).tolist(), strict=True))}, report["feature_missingness"],
                     "new-arm evaluation missingness")
    rows = prediction_rows(evaluation, predictions)
    require(set(rows) == set(saved), "exact four-arm/three-pair prediction schema required")
    old.compare_frame(rows, saved, list(rows), "all four unchanged/new predictions and row error/paired bounds")
    scores, processes = compute_scores(evaluation, predictions)
    old.compare_json(scores, old._read_json(run / "scores.json"), "all independent four-arm annual/pooled scores and primary gates")
    old.compare_json(scores, report["metrics"], "independent report scores")
    old.compare_frame(processes, pd.read_parquet(run / "process_scores.parquet"), list(processes), "all fixed four-arm process scores")
    old.verify_hashes(frozen, root)
    validate_artifact_locks(parent, parent_inventory, parent_hashes)
    validate_protected_inventory(report["protected_sha256"], root)
    old._verify_environment(old._read_json(run / "environment.json"), report["environment_sha256"])
    require(old.sha256(run / "report.json") == report_hash, "child report changed during verification")
    validate_output_inventory(run, report["output_sha256"], set(report["output_sha256"]))
    result = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
              "run_dir": str(run), "report_sha256": report_hash, "verifier_sha256": old.sha256(__file__),
              "reused_independent_verifier_sha256": BRIDGE_SHA256, "parent_report_sha256": PARENT_REPORT_SHA256,
              "parent_verification_sha256": PARENT_VERIFICATION_SHA256, "grid_rows_reconstructed": len(grid),
              "calendar_origins_reconstructed": len(population), "shared_legacy_origins_verified": len(parent_population),
              "fixed_origins_verified": len(evaluation), "models_verified": len(models["models"]),
              "training_member_rows_verified": checked, "shared_legacy_training_member_rows_verified": shared_checked,
              "refitted_predictions_verified": len(evaluation), "inherited_predictions_verified": len(evaluation) * len(INHERITED_ARMS),
              "feature_order": names, "counts": counts, "metrics": scores, "yearly_acceptance_passed": scores["gates"]["passed"],
              "primary_candidate": PRIMARY, "production_ready": False, "deployed": False,
              "limits": ["All-calendar population/members are constructed independently, without importing any runner",
                         "Label/feature/scoring mathematics reuse the reviewed frozen independent bridge verifier",
                         "Per-cutoff exact T=1 exposure, shared old vectors/labels/maturity and three fixed reference predictions are verified",
                         "Population expansion changes target frequencies, histogram bins and effective regularization; it is not unique clean-supervision causation",
                         "Native model/member hashes and reloaded predictions are an auditable saved chain, not independent retraining proof",
                         "Archive-end censor maturity does not establish complete ongoing online supervision or weather issue time",
                         "Known historical years are not blind tests; an accuracy gate pass is not deployment or real-time accuracy approval"]}
    old.write_exclusive_json(run / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing approved all-calendar run")
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
    print(json.dumps({key: result[key] for key in ("status", "models_verified", "fixed_origins_verified", "yearly_acceptance_passed", "production_ready")}))


if __name__ == "__main__":
    main()
