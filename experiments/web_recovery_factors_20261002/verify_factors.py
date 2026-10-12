"""Verify fixed factors using the frozen, independent bridge mathematics.

This reuses the reviewed bridge reconstruction rather than claiming a newly
written second algorithm. The new runner is never imported as evidence.
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
PARENT_REPORT_SHA256 = "70217776f86326cd9a9d4436af12d2d352b4f11ea69b2f1fc68becb1951f01d0"
PARENT_VERIFICATION_SHA256 = "7003b164439cbf87e4146459ae2f219bf6f3b45af1cc70407daf7935eba1d703"
PARENT_RELATIVE = "experiments/web_recovery_bridge_20261002/runs/20261002T153221834258Z"
INHERITED_ARMS = ("archive", "legacy_control", "interval_candidate")
NEW_ARMS = ("labels_only", "grid_all_legacy_origins")
ARMS = (*INHERITED_ARMS, *NEW_ARMS)
PRIMARY = "grid_all_legacy_origins"
PAIRS = {**{f"{arm}_vs_archive": ("archive", arm) for arm in ARMS[1:]},
         "labels_only_vs_legacy_control": ("legacy_control", "labels_only"),
         "grid_all_legacy_origins_vs_labels_only": ("labels_only", "grid_all_legacy_origins"),
         "interval_candidate_vs_grid_all_legacy_origins": ("grid_all_legacy_origins", "interval_candidate")}


def bridge():
    path = Path(__file__).resolve().parent.parent / "web_recovery_bridge_20261002/verify_bridge.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError("frozen independent verifier changed")
    spec = importlib.util.spec_from_file_location("frozen_independent_bridge_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid_population(legacy, grid, names):
    times = pd.DatetimeIndex(grid.timestamp)
    positions = times.get_indexer(legacy.timestamp)
    if not times.is_unique or not legacy.timestamp.is_unique or (positions < 0).any():
        raise ValueError("legacy population outside unique grid")
    result = legacy.copy()
    result.loc[:, names] = grid.iloc[positions][names].to_numpy(dtype=float)
    return result


def validate_inventory(records, evaluation):
    old = bridge()
    wanted = {(arm, int(block)) for block in evaluation.continuity_block.unique() for arm in NEW_ARMS}
    got = [(record["arm"], int(record["continuity_block"])) for record in records]
    old._require(len(got) == len(wanted) and set(got) == wanted, "new model inventory mismatch")
    ids = [record["model_id"] for record in records]
    old._require(len(ids) == len(set(ids)), "duplicated model identity")
    for record in records:
        old._require(record["model_id"] == f"{record['arm']}_b{record['continuity_block']}", "model identity mismatch")


def validate_same_members(members):
    old = bridge()
    columns = [name for name in members.columns if name not in ("model_id", "arm")]
    for block, rows in members.groupby("evaluation_block", sort=True):
        old._require(set(rows.arm) == set(NEW_ARMS), "two member arms required")
        left = rows.loc[rows.arm.eq(NEW_ARMS[0])].sort_values("population_row")
        right = rows.loc[rows.arm.eq(NEW_ARMS[1])].sort_values("population_row")
        old._require(not left.population_row.duplicated().any() and not right.population_row.duplicated().any(), "duplicate training members")
        old.compare_frame(left, right, columns, f"same mature members block {block}")


def validate_parent_members(members, parent_members):
    old = bridge()
    columns = [name for name in parent_members if name not in ("model_id", "arm")]
    control = parent_members.loc[parent_members.arm.eq("legacy_control")]
    for arm in NEW_ARMS:
        current = members.loc[members.arm.eq(arm)]
        old.compare_frame(control.sort_values(["evaluation_block", "population_row"]),
                          current.sort_values(["evaluation_block", "population_row"]), columns, f"unchanged parent control members {arm}")


def compute_scores(evaluation, predictions, years=(2024, 2025)):
    old = bridge()
    old._require(set(predictions) == set(ARMS), "all five prediction arms required")

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
        gate["checks"]["coverage_100"] = all(score["prediction_coverage"] == 1. for score in metrics["arms"].values())
        gate["passed"] = all(gate["checks"].values())
        gate["failure_reasons"] = [key for key, passed in gate["checks"].items() if not passed]
        gates[str(year)] = gate
    scores = {"annual": annual, "pooled": subset(evaluation, predictions),
              "gates": {"annual": gates, "passed": all(item["passed"] for item in gates.values()),
                        "failure_reasons": [f"{year}:{key}" for year in years for key in gates[str(year)]["failure_reasons"]]}}
    rows = []
    for (year, block), frame in evaluation.groupby(["year", "continuity_block"], sort=True):
        for arm in ARMS:
            score = old.score_predictions(frame, np.asarray(predictions[arm])[frame.index.to_numpy()])
            record = {"year": int(year), "continuity_block": int(block), "arm": arm,
                      "old_events": int(frame.event_key.nunique()), **{key: value for key, value in score.items() if key != "subgroups"}}
            for name, subgroup in score["subgroups"].items():
                record.update({f"{name}_{key}": value for key, value in subgroup.items()})
            rows.append(record)
    return scores, pd.DataFrame(rows)


def prediction_rows(evaluation, predictions):
    old = bridge()
    old._require(set(predictions) == set(ARMS), "all five prediction arms required")
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


def validate_scope(report):
    expected = {"primary_candidate": PRIMARY, "diagnostic_only_arms": ["labels_only"],
                "production_ready": False, "deployed": False, "blind_test": False,
                "historical_shadow_evaluation": True, "archive_is_primary_reference": True}
    old = bridge()
    for key, value in expected.items():
        if isinstance(value, bool):
            old._require(report.get(key) is value, f"declared scope {key} is not the required boolean")
        else:
            old.compare_json(value, report.get(key), f"declared scope {key}")


def validate_output_inventory(run, expected, required):
    old = bridge()
    old._require(isinstance(expected, dict) and expected and set(required).issubset(expected), "required output locks missing")
    actual = {path.relative_to(run).as_posix() for path in Path(run).rglob("*")
              if path.is_file() and path != Path(run) / "report.json"}
    old._require(actual == set(expected), "output inventory differs")
    for name, digest in expected.items():
        old._require(old.sha256(old.run_file(run, name)) == digest, f"output hash mismatch: {name}")


def artifact_locks(folder):
    old = bridge()
    paths = sorted(path for path in Path(folder).rglob("*") if path.is_file())
    return (sorted(path.relative_to(folder).as_posix() for path in paths),
            {str(path.resolve()): old.sha256(path) for path in paths})


def validate_artifact_locks(folder, inventory, hashes):
    names, current = artifact_locks(folder)
    old = bridge()
    old.compare_json(names, inventory, "parent artifact inventory")
    old.compare_json(current, hashes, "parent artifact locks")


def validate_parent_evidence(verification, report_hash, verification_hash):
    old = bridge()
    old._require(report_hash == PARENT_REPORT_SHA256 and verification_hash == PARENT_VERIFICATION_SHA256,
                 "parent exact report/verification hash differs")
    expected = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
                "report_sha256": report_hash, "verifier_sha256": BRIDGE_SHA256,
                "fixed_origins_verified": 701, "models_verified": 64, "refitted_predictions_verified": 1402}
    for key, value in expected.items():
        old.compare_json(value, verification.get(key), f"parent passed evidence {key}")


def validate_parent_chain(parent, root):
    old = bridge()
    parent = Path(parent).resolve()
    old._require(parent == (root / PARENT_RELATIVE).resolve(), "parent is not the predeclared run")
    report = old._read_json(parent / "report.json")
    verification = old._read_json(parent / "verification.json")
    validate_parent_evidence(verification, old.sha256(parent / "report.json"), old.sha256(parent / "verification.json"))
    old._require(Path(verification["run_dir"]).resolve() == parent, "parent verification run identity differs")
    old.compare_json(report["metrics"], verification["metrics"], "parent independently verified metrics")
    outputs = report["output_sha256"]
    inventory, hashes = artifact_locks(parent)
    old._require(set(inventory) == set(outputs) | {"report.json", "verification.json"}, "parent has foreign or missing artifacts")
    old.verify_hashes(outputs, parent)
    union = old.combine_locks([report[name] for name in ("input_sha256", "protected_sha256", "code_sha256", "environment_sha256")])
    old.compare_json(union, report["frozen_sha256"], "parent original frozen union")
    old.verify_hashes(union, root)
    old._verify_environment(old._read_json(parent / "environment.json"), report["environment_sha256"])
    return report, inventory, hashes


def training_members(population, record):
    old = bridge()
    expected = old.expected_members(population, record["cutoff"], int(record["continuity_block"]))
    result = expected[["population_row", "timestamp", "continuity_block", *old.LABEL_COLUMNS]].copy()
    result["legacy_remaining_hours"] = expected.legacy_remaining_hours.to_numpy()
    result["model_id"], result["arm"] = record["model_id"], record["arm"]
    result["evaluation_block"], result["cutoff"] = int(record["continuity_block"]), pd.Timestamp(record["cutoff"])
    return result


def validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes):
    old = bridge()
    groups = ("input_sha256", "protected_sha256", "code_sha256", "environment_sha256", "parent_sha256")
    old._require(all(isinstance(report.get(name), dict) and report[name] for name in (*groups, "frozen_sha256", "output_sha256")),
                 "nonempty provenance groups required")
    old.compare_json(parent_inventory, report.get("parent_artifact_inventory"), "declared parent inventory")
    old.compare_json(parent_hashes, report["parent_sha256"], "declared parent all-file locks")
    old.compare_json(parent_report["frozen_sha256"], report.get("original_scope_sha256"), "original protected scope")
    old.compare_json(BRIDGE_SHA256, report.get("parent_verifier_sha256"), "parent verifier source identity")
    for name in ("historical_input_sha256", "historical_audit_drift"):
        old.compare_json(parent_report[name], report.get(name), f"unchanged parent {name}")
    old._require(Path(report["parent_run_dir"]).resolve() == parent, "declared parent run differs")
    for name, digest in (("parent_report_sha256", PARENT_REPORT_SHA256), ("parent_verification_sha256", PARENT_VERIFICATION_SHA256),
                         ("parent_verification_report_sha256", PARENT_REPORT_SHA256)):
        old.compare_json(digest, report.get(name), name)
    union = old.combine_locks([parent_report["frozen_sha256"], *[report[name] for name in groups]])
    old.compare_json(union, report["frozen_sha256"], "child complete frozen union")
    for group in ("input_sha256", "protected_sha256", "code_sha256", "environment_sha256"):
        old._require(all(report[group].get(path) == digest for path, digest in parent_report[group].items()), f"parent {group} not inherited intact")
    required_code = [root / "experiments/web_recovery_factors_20261002" / name for name in
                     ("PROTOCOL.md", "factor_runner.py", "test_factor_runner.py", "verify_factors.py", "test_verify_factors.py")]
    old._require(all(str(path.resolve()) in report["code_sha256"] for path in required_code), "child protocol/code lock coverage incomplete")
    for name in ("artifacts/cnemc_2025_deployment_candidate", "artifacts/strict_history_backtest", "project-showcase", "sites-haze/dist", "releases/runtime_20261002"):
        folder = root / name
        paths = [str(path.resolve()) for path in folder.rglob("*") if path.is_file()] if folder.is_dir() else []
        old._require(paths and all(path in report["protected_sha256"] for path in paths), f"protected inventory missing: {name}")
    old.verify_hashes(union, root)
    required = {"evaluation.parquet", "legacy_population.parquet", "grid_population.parquet", "features.npz", "environment.json",
                "execution_record_at_start.md", "training_members.parquet", "models.json", "predictions.parquet", "process_scores.parquet", "scores.json"}
    validate_output_inventory(run, report["output_sha256"], required)
    old._require(report.get("schema_version") == 1 and report.get("status") == "pending_independent_verification", "child run is not completed pending verification")
    validate_scope(report)
    old.compare_json(list(NEW_ARMS), report.get("new_refitted_arms"), "declared new arms")
    old.compare_json(list(INHERITED_ARMS), report.get("inherited_arms"), "declared inherited arms")
    old.compare_json("sequential conditional decomposition, not a full factorial or unique causal attribution", report.get("scope"), "factor scope")
    old._require(report.get("years") == [2024, 2025] and report.get("protected_unchanged") is True, "historical years/protection scope differs")
    old._require(report.get("parent_unchanged") is True, "parent protection status differs")
    for key in ("target", "uncertainty_scope"):
        old.compare_json(parent_report[key], report.get(key), f"unchanged target {key}")
    old.compare_json(parent_report["configuration"], report.get("configuration"), "fixed configuration")
    old.compare_json(old._fixed_parameters(), report.get("aft_parameters"), "fixed AFT parameters")
    return union


def verify_models(run, report, models, members, evaluation, grid, populations, features, names):
    old = bridge()
    import xgboost as xgb

    validate_inventory(models["models"], evaluation)
    old._require(models.get("schema_version") == 1 and len(models["models"]) == report["fitted_models"] == 64, "all 64 new models required")
    old.compare_json(names, models["feature_names"], "model feature names")
    old._require(set(members.model_id) == {record["model_id"] for record in models["models"]}, "foreign or missing model members")
    validate_same_members(members)
    saved = pd.read_parquet(run / "predictions.parquet")
    result = {arm: np.full(len(evaluation), np.nan) for arm in NEW_ARMS}
    checked = 0
    for record in models["models"]:
        arm, block = record["arm"], int(record["continuity_block"])
        cutoff = grid.loc[grid.continuity_block.eq(block), "timestamp"].min()
        old._require(pd.Timestamp(record["cutoff"]) == cutoff, "model cutoff is not full-grid block start")
        population = populations[arm]
        wanted = training_members(population, record)
        actual = members.loc[members.model_id.eq(record["model_id"])]
        old._require(not wanted.empty, "empty mature training population")
        old.compare_frame(wanted, actual, list(wanted), f"all members {record['model_id']}")
        expected = old.expected_members(population, cutoff, block)
        positions = np.flatnonzero(evaluation.continuity_block.eq(block).to_numpy())
        checked += len(expected)
        old._require(record["training_member_path"] == "training_members.parquet", "member source differs")
        old._require(record["parent_member_model_id"] == f"legacy_control_b{block}", "parent member model identity differs")
        old._require(record["train_rows"] == len(expected) and record["eval_rows"] == len(positions), "model member/query count differs")
        old.compare_json(old._label_counts(expected), record["label_counts"], "model label counts")
        old._require(pd.Timestamp(record["training_last_timestamp"]) == expected.timestamp.max()
                     and pd.Timestamp(record["max_label_available_at"]) == expected.label_available_at.max(), "model mature cutoff extrema differ")
        old._require(record["training_features_sha256"] == old.matrix_digest(expected[names].to_numpy(dtype=float)), "model training matrix digest differs")
        timestamps_hash = hashlib.sha256(pd.DatetimeIndex(expected.timestamp).asi8.astype("<i8").tobytes()).hexdigest()
        old._require(record["training_timestamps_sha256"] == timestamps_hash, "model training timestamp digest differs")
        for key in ("feature_names", "booster_feature_names"):
            old.compare_json(names, record[key], key)
        old.compare_json(report["configuration"], record["configuration"], "model fixed configuration")
        old.compare_json(old._fixed_parameters(), record["aft_parameters"], "model fixed AFT parameters")
        old._require(record["num_boosted_rounds"] == 500, "model fitted rounds differ")
        old.validate_trained_config(record["actual_trained_booster_config"], 87)
        old._require(record["model_path"] == f"models/{record['model_id']}.json", "model path/identity differs")
        path = old.run_file(run, record["model_path"])
        old._require(report["output_sha256"].get(record["model_path"]) == record["model_sha256"] == old.sha256(path), "model is not locked locally")
        old._validate_native_trees(path, 87)
        booster = xgb.Booster(params={"nthread": 1})
        booster.load_model(path)
        old._require(booster.num_boosted_rounds() == 500 and booster.num_features() == 87, "loaded round/feature count differs")
        old.compare_json(names, booster.feature_names, "loaded feature order")
        objective = json.loads(booster.save_config())["learner"]["objective"]
        old._require(objective["name"] == "survival:aft" and objective["aft_loss_param"]["aft_loss_distribution"] == "normal",
                     "loaded AFT objective/distribution differs")
        old._require(np.isclose(float(objective["aft_loss_param"]["aft_loss_distribution_scale"]), 1.2, rtol=1e-6), "loaded AFT scale differs")
        query = pd.DataFrame(features[arm][positions], columns=names)
        prediction = booster.predict(xgb.DMatrix(query, missing=np.nan))
        old._require(np.isfinite(prediction).all() and (prediction >= 0).all(), "invalid inferred predictions")
        old.compare_array(prediction, saved.loc[positions, arm].to_numpy(dtype=float), f"all inferred predictions {record['model_id']}")
        result[arm][positions] = prediction
    old._require(checked == len(members), "unused or duplicated model members")
    return result, checked


def verify_run(run, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    old = bridge()
    run = old.validate_run_directory(run, root / "experiments/web_recovery_factors_20261002")
    old._require(not (run / "verification.json").exists(), "verification.json already exists; no overwrite")
    parent = (root / PARENT_RELATIVE).resolve()
    parent_report, parent_inventory, parent_hashes = validate_parent_chain(parent, root)
    report_hash = old.sha256(run / "report.json")
    report = old._read_json(run / "report.json")
    frozen = validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes)
    old._verify_environment(old._read_json(run / "environment.json"), report["environment_sha256"])
    names = old._read_json(root / "artifacts/cnemc_2025_deployment_candidate/model_metadata.json")["feature_names"]
    old._require(len(names) == len(set(names)) == 87, "87 unique ordered features required")
    old.compare_json(names, report["feature_names"], "report feature order")
    grid = old.rebuild_grid({source: pd.read_parquet(root / path) for source, path in old.RAW_SOURCES.items()})
    grid_columns = ["timestamp", *names, "continuity_block", "episode_id", "eligible", "pollution_state", "input_hour_observed"]
    old.compare_frame(grid, pd.read_parquet(root / "artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet"),
                      grid_columns, "complete raw-rebuilt grid")
    candidate = old._add_bounds(grid.loc[grid.eligible].copy(), grid)
    candidate["population_row"] = np.arange(len(candidate), dtype=np.int64)
    legacy = old.rebuild_legacy({source: pd.read_parquet(root / path) for source, path in old.LEGACY_SOURCES.items()}, grid, names)
    grid_all = grid_population(legacy, grid, names)
    old.compare_frame(candidate, pd.read_parquet(parent / "candidate_population.parquet"),
                      ["timestamp", *names, "population_row", "continuity_block", *old.LABEL_COLUMNS], "parent canonical population")
    for folder in (parent, run):
        old.compare_frame(legacy, pd.read_parquet(folder / "legacy_population.parquet"), list(legacy), "all old origins and canonical bounds")
    old.compare_frame(grid_all, pd.read_parquet(run / "grid_population.parquet"), list(grid_all), "all grid legacy origins and canonical bounds")
    archive = pd.read_parquet(root / "artifacts/strict_history_backtest/predictions.parquet")
    archive["timestamp"] = pd.to_datetime(archive.timestamp, utc=True).dt.tz_convert("Asia/Shanghai")
    archive = archive.loc[archive.timestamp.dt.year.isin((2024, 2025))].sort_values("timestamp").reset_index(drop=True)
    evaluation = old.rebuild_evaluation(archive, grid)
    for folder in (parent, run):
        old.compare_frame(evaluation, pd.read_parquet(folder / "evaluation.parquet"), list(evaluation), "all fixed 701 origins and labels")
    old._require((len(grid), len(candidate), len(legacy), len(evaluation), evaluation.continuity_block.nunique()) == (102024, 7890, 8638, 701, 32),
                 "full population/block counts differ")
    counts = {"grid_rows": len(grid), "candidate_population": len(candidate), "legacy_population": len(legacy),
        "evaluation_rows": len(evaluation), "evaluation_blocks": int(evaluation.continuity_block.nunique()),
        "planned_fits": int(evaluation.continuity_block.nunique() * 2),
        "annual": {str(year): {"rows": int(evaluation.year.eq(year).sum()),
            "old_events": int(evaluation.loc[evaluation.year.eq(year), "event_key"].nunique()),
            "blocks": int(evaluation.loc[evaluation.year.eq(year), "continuity_block"].nunique()),
            "causal_eligible": int(evaluation.loc[evaluation.year.eq(year), "causal_eligible"].sum()),
            "labels": old._label_counts(evaluation.loc[evaluation.year.eq(year)])} for year in (2024, 2025)}}
    old.compare_json(counts, parent_report["counts"], "raw-reconstructed parent counts")
    counts.update({"new_training_population": len(legacy), "new_training_population_label_counts": old._label_counts(legacy)})
    old.compare_json(counts, report["counts"], "factor complete population counts")
    features = {"labels_only": legacy.set_index("timestamp").loc[evaluation.timestamp, names].to_numpy(dtype=float),
                PRIMARY: grid.set_index("timestamp").loc[evaluation.timestamp, names].to_numpy(dtype=float)}
    old.compare_array(features["labels_only"], archive[names].to_numpy(dtype=float), "legacy archive feature vectors")
    for folder, keys in ((parent, {"labels_only": "legacy", PRIMARY: "candidate"}), (run, {arm: arm for arm in NEW_ARMS})):
        with np.load(folder / "features.npz", allow_pickle=False) as stored:
            old._require(set(stored.files) == {"timestamps", "feature_names", *keys.values()}, "evaluation feature archive schema differs")
            old.compare_array(np.asarray(names), stored["feature_names"], "ordered NPZ features")
            old.compare_array(pd.DatetimeIndex(evaluation.timestamp).asi8, stored["timestamps"], "exact UTCns origin clock")
            for arm, key in keys.items():
                old.compare_array(features[arm], stored[key], f"all raw-rebuilt {arm} features")
    inherited = pd.read_parquet(parent / "predictions.parquet")
    old.compare_frame(evaluation, inherited, list(evaluation), "parent prediction cohort")
    predictions = {arm: inherited[arm].to_numpy(dtype=float) for arm in INHERITED_ARMS}
    old.compare_array(predictions["archive"], evaluation.archive_prediction_hours, "parent archive predictions")
    parent_scores, parent_process = old.compute_scores(evaluation, predictions)
    old.compare_json(parent_scores, parent_report["metrics"], "parent metrics from immutable predictions")
    old.compare_frame(parent_process, pd.read_parquet(parent / "process_scores.parquet"), list(parent_process), "parent immutable process scores")
    models = old._read_json(run / "models.json")
    members = pd.read_parquet(run / "training_members.parquet")
    validate_parent_members(members, pd.read_parquet(parent / "training_members.parquet"))
    inferred, member_count = verify_models(run, report, models, members, evaluation, grid,
                                         {"labels_only": legacy, PRIMARY: grid_all}, features, names)
    old.compare_json({arm: dict(zip(names, np.isnan(features[arm]).mean(axis=0).tolist())) for arm in NEW_ARMS},
                     report["feature_missingness"], "evaluation feature missingness")
    maturity = [{key: record[key] for key in ("model_id", "train_rows", "label_counts", "training_last_timestamp", "max_label_available_at")}
                for record in models["models"]]
    old.compare_json(maturity, report["training_maturity_summary"], "all model maturity summary")
    predictions.update(inferred)
    rows = prediction_rows(evaluation, predictions)
    old.compare_frame(rows, pd.read_parquet(run / "predictions.parquet"), list(rows), "all five arms and seven paired row bounds")
    scores, process = compute_scores(evaluation, predictions)
    old.compare_json(scores, old._read_json(run / "scores.json"), "all independently aggregated five-arm scores")
    old.compare_json(scores, report["metrics"], "report scores")
    old.compare_frame(process, pd.read_parquet(run / "process_scores.parquet"), list(process), "all five-arm process scores")
    old.verify_hashes(frozen, root)
    validate_artifact_locks(parent, parent_inventory, parent_hashes)
    old._require(old.sha256(run / "report.json") == report_hash, "child report changed during verification")
    validate_output_inventory(run, report["output_sha256"], set(report["output_sha256"]))
    result = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
              "run_dir": str(run), "report_sha256": report_hash, "verifier_sha256": old.sha256(__file__),
              "reused_independent_verifier_sha256": BRIDGE_SHA256, "parent_report_sha256": PARENT_REPORT_SHA256,
              "parent_verification_sha256": PARENT_VERIFICATION_SHA256, "grid_rows_reconstructed": len(grid),
              "canonical_origins_reconstructed": len(candidate), "legacy_origins_reconstructed": len(legacy),
              "fixed_origins_verified": len(evaluation), "models_verified": len(models["models"]),
              "training_member_rows_verified": member_count, "refitted_predictions_verified": len(evaluation) * 2,
              "feature_order": names, "metrics": scores, "yearly_acceptance_passed": scores["gates"]["passed"],
              "primary_candidate": PRIMARY, "diagnostic_only_arms": ["labels_only"], "production_ready": False, "deployed": False,
              "limits": ["Reuses the reviewed frozen independent bridge mathematics, not a newly written second algorithm",
                         "Label maturity does not prove cutoff-prefix reconstruction of the historically selected legacy population or weather issue time",
                         "Passing the retrospective score gate permits further prefix auditing, not deployment",
                         "Native model and member digests are an auditable saved chain, not independent retraining proof",
                         "Sequential conditional decomposition is not a full factorial or unique physical causal attribution",
                         "Known 2024/2025 years are not a new blind test; no future real-time accuracy claim"]}
    old.write_exclusive_json(run / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing approved factor run")
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
