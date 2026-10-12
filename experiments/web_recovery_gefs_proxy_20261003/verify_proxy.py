"""Independent GEFS proxy reconstruction and native-model replay, never fitting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BRIDGE_SHA256 = "9477b5452830f82756219d317a6928e12708897a061cb442eeb65304fccd6b2b"
FACTORS_SHA256 = "1c3052b2f51023d8e08649ed47dff614f05065ba1dd161d9964b8151400c6c6e"
PARENT_REPORT_SHA256 = "21aa39d5a08a683e23904000eeeccbb3c31fda30201cf30e64b4cf28ab063998"
PARENT_VERIFICATION_SHA256 = "17f3a9e71d15e95c3f9f02149a5bd1c1f27abe784682979c050263e44e0f8566"
PARENT_RELATIVE = "experiments/web_recovery_factors_20261002/runs/20261002T161429405383Z"
SNAPSHOT_RELATIVE = "artifacts/task01_recovery_head/reusable_gefs_snapshots_v1/snapshots.parquet"
SNAPSHOT_SHA256 = "a686840d341e3e838c3e0c7021da9adf34200303a8d0733ee5b9e390b1f53ca4"
SNAPSHOT_REPORT_SHA256 = "08ee7cb59d6bae4dd3a624a9f1f526e27bd1ed8216929af5d9e9466b6643dba6"
TIME_COLUMNS = ("known_at", "run_at", "valid_at")
AGE_COLUMNS = ("forecast_known_age_hours", "forecast_run_age_hours", "forecast_remaining_lead_hours")
INHERITED_ARMS = ("archive", "grid_all_legacy_origins")
NEW_ARMS = ("metadata_control", "weather_values")
ARMS = (*INHERITED_ARMS, *NEW_ARMS)
PRIMARY = "weather_values"
PAIRS = {**{f"{arm}_vs_archive": ("archive", arm) for arm in ARMS[1:]},
         "weather_values_vs_metadata_control": ("metadata_control", "weather_values")}


def _load(relative, digest, name):
    path = Path(__file__).resolve().parents[1] / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("frozen independent verifier changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def bridge():
    return _load("web_recovery_bridge_20261002/verify_bridge.py", BRIDGE_SHA256, "frozen_proxy_bridge")


@lru_cache(maxsize=1)
def factors():
    return _load("web_recovery_factors_20261002/verify_factors.py", FACTORS_SHA256, "frozen_proxy_factors")


def _utc_times(values, name):
    try:
        times = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: invalid aware datetime vector") from exc
    if times.tz is None or times.hasnans:
        raise ValueError(f"{name}: timezone-aware nonmissing timestamps required")
    return times.tz_convert("UTC")


def match_snapshots(origins, snapshots):
    """Select latest valid known snapshot by independent UTC-ns NumPy comparisons."""
    old = bridge()
    origins = _utc_times(origins, "origins")
    old._require(set(TIME_COLUMNS).issubset(snapshots), "snapshot clocks missing")
    names = [name for name in snapshots if name not in TIME_COLUMNS]
    old._require(len(names) == 9 and len(set(names)) == 9, "nine ordered weather features required")
    clocks = {name: _utc_times(snapshots[name], name) for name in TIME_COLUMNS}
    known, run, valid = (clocks[name].asi8 for name in TIME_COLUMNS)
    old._require(len(np.unique(known)) == len(known), "duplicated known time")
    old._require(np.all(run <= known) and np.all(known < valid), "invalid forecast time relationship")
    old._require(np.all(valid - run == 24 * 3600 * 10**9), "snapshot is not f024")
    values = snapshots[names].to_numpy(dtype=float)
    old._require(np.isfinite(values).all(), "complete finite snapshot weather required")
    order = np.argsort(known, kind="stable")
    known, run, valid, values = known[order], run[order], valid[order], values[order]
    chosen = np.full(len(origins), -1, dtype=np.int64)
    for index, origin in enumerate(origins.asi8):
        legal = np.flatnonzero((known <= origin) & (origin < valid))
        if len(legal):
            chosen[index] = legal[-1]
    present = chosen >= 0
    output = pd.DataFrame(np.full((len(origins), 9), np.nan), columns=names)
    output.loc[present, names] = values[chosen[present]]
    for name, clock in zip(TIME_COLUMNS, (known, run, valid), strict=True):
        result = np.full(len(origins), np.iinfo(np.int64).min, dtype=np.int64)
        result[present] = clock[chosen[present]]
        output[name] = pd.to_datetime(result, utc=True)
    for name, clock, sign in zip(AGE_COLUMNS, (known, run, valid), (1, 1, -1), strict=True):
        age = np.full(len(origins), np.nan)
        age[present] = sign * (origins.asi8[present] - clock[chosen[present]]) / (3600 * 10**9)
        output[name] = age
    return output, names


def rebuild_population(base, snapshots, names):
    matched, weather_names = match_snapshots(base.timestamp, snapshots)
    all_names = [*names, *weather_names, *AGE_COLUMNS]
    bridge()._require(len(all_names) == len(set(all_names)), "feature name collision")
    true = base.copy().reset_index(drop=True)
    for name in matched:
        true[name] = matched[name]
    control = true.copy()
    values = true[weather_names].to_numpy(dtype=float)
    control.loc[:, weather_names] = np.where(np.isfinite(values), 0., np.nan)
    bridge().compare_array(np.isnan(true[all_names]), np.isnan(control[all_names]), "same two-arm missingness")
    return true, control, all_names


def validate_population(expected, actual, base):
    old = bridge()
    old._require(set(actual) == set(expected), "population exact schema differs")
    old.compare_frame(base, actual, list(base), "unchanged original realhour features/bounds/population")
    old.compare_frame(expected, actual, list(expected), "all independent proxy clocks/features")


def validate_inventory(records, evaluation):
    old = bridge()
    wanted = {(arm, int(block)) for block in evaluation.continuity_block.unique() for arm in NEW_ARMS}
    actual = [(record["arm"], int(record["continuity_block"])) for record in records]
    old._require(len(actual) == len(wanted) and set(actual) == wanted, "proxy model inventory mismatch")
    ids = [record["model_id"] for record in records]
    old._require(len(ids) == len(set(ids)), "duplicate model identity")
    for record in records:
        old._require(record["model_id"] == f"{record['arm']}_b{record['continuity_block']}", "model identity mismatch")


def information_gate(metadata, weather, paired):
    def compare(name, strict=False):
        left, right = metadata.get(name), weather.get(name)
        return left is not None and right is not None and (right < left if strict else right <= left)

    checks = {"exact_row_strict_improvement": compare("exact_row_mae", True),
              "exact_process_not_worse": compare("exact_block_mae"),
              "tail_exact_block_not_worse": compare("tail24_block_mae"),
              "paired_row_upper_nonpositive": paired.get("all_row_delta_upper") is not None and paired["all_row_delta_upper"] <= 0,
              "paired_block_upper_nonpositive": paired.get("all_block_delta_upper") is not None and paired["all_block_delta_upper"] <= 0,
              "coverage_100": metadata.get("prediction_coverage") == weather.get("prediction_coverage") == 1.}
    return {"checks": checks, "passed": bool(all(checks.values())),
            "failure_reasons": [key for key, value in checks.items() if not value]}


def compute_scores(evaluation, predictions, years=(2024, 2025)):
    old = bridge()
    old._require(set(predictions) == set(ARMS), "all four prediction arms required")

    def subset(frame, values):
        return {"arms": {arm: old.score_predictions(frame, values[arm]) for arm in ARMS},
                "paired": {key: old.paired_summary(frame, values[base], values[candidate])
                           for key, (base, candidate) in PAIRS.items()}}

    annual, gates = {}, {}
    for year in years:
        mask = evaluation.year.eq(year).to_numpy()
        metrics = subset(evaluation.loc[mask], {arm: np.asarray(values)[mask] for arm, values in predictions.items()})
        annual[str(year)] = metrics
        primary = old.acceptance_gate(metrics["arms"]["archive"], metrics["arms"][PRIMARY], metrics["paired"][f"{PRIMARY}_vs_archive"])
        primary["checks"]["coverage_100"] = all(score["prediction_coverage"] == 1. for score in metrics["arms"].values())
        primary["passed"] = all(primary["checks"].values())
        primary["failure_reasons"] = [key for key, passed in primary["checks"].items() if not passed]
        information = information_gate(metrics["arms"]["metadata_control"], metrics["arms"][PRIMARY], metrics["paired"]["weather_values_vs_metadata_control"])
        gates[str(year)] = {"primary": primary, "weather_information": information,
                           "passed": primary["passed"] and information["passed"],
                           "failure_reasons": [f"{name}:{key}" for name, gate in (("primary", primary), ("weather_information", information)) for key in gate["failure_reasons"]]}
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
    old._require(set(predictions) == set(ARMS), "all four prediction arms required")
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


def validate_output_inventory(run, expected, required):
    return factors().validate_output_inventory(run, expected, required)


def validate_parent_chain(parent, root):
    old, factor = bridge(), factors()
    parent = Path(parent).resolve()
    old._require(parent == (root / PARENT_RELATIVE).resolve(), "parent is not predeclared factor run")
    report, verification = (old._read_json(parent / name) for name in ("report.json", "verification.json"))
    old._require(old.sha256(parent / "report.json") == PARENT_REPORT_SHA256, "parent report hash differs")
    old._require(old.sha256(parent / "verification.json") == PARENT_VERIFICATION_SHA256, "parent verification hash differs")
    expected = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
                "report_sha256": PARENT_REPORT_SHA256, "verifier_sha256": FACTORS_SHA256,
                "fixed_origins_verified": 701, "models_verified": 64, "training_member_rows_verified": 533044,
                "refitted_predictions_verified": 1402, "production_ready": False, "deployed": False}
    for key, value in expected.items():
        old.compare_json(value, verification.get(key), f"parent passed evidence {key}")
    old._require(Path(verification["run_dir"]).resolve() == parent, "parent verified run identity differs")
    old.compare_json(report["metrics"], verification["metrics"], "parent verified metrics")
    inventory, hashes = factor.artifact_locks(parent)
    old._require(len(inventory) == 77 and set(inventory) == set(report["output_sha256"]) | {"report.json", "verification.json"}, "parent 77 artifact inventory differs")
    old.verify_hashes(report["output_sha256"], parent)
    old.verify_hashes(report["frozen_sha256"], root)
    old._verify_environment(old._read_json(parent / "environment.json"), report["environment_sha256"])
    return report, inventory, hashes


def verify_run(run, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    run = bridge().validate_run_directory(run, root / "experiments/web_recovery_gefs_proxy_20261003")
    bridge()._require(not (run / "verification.json").exists(), "verification.json already exists; no overwrite")
    return _verify_complete_run(run, root)


def validate_same_members(members, parent_members):
    old = bridge()
    old._require(set(members.arm) == set(NEW_ARMS), "exact two training member arms required")
    parent = parent_members.loc[parent_members.arm.eq("grid_all_legacy_origins")]
    columns = [name for name in parent if name not in ("model_id", "arm")]
    keys = ["evaluation_block", "population_row"]
    old._require(not parent.duplicated(keys).any(), "duplicate parent members")
    for arm in NEW_ARMS:
        actual = members.loc[members.arm.eq(arm)]
        old._require(not actual.duplicated(keys).any(), "duplicate proxy members")
        old.compare_frame(parent.sort_values(keys), actual.sort_values(keys), columns, f"parent identical mature members {arm}")


def validate_feature_archive(path, evaluation, names, features):
    old = bridge()
    with np.load(path, allow_pickle=False) as stored:
        old._require(set(stored.files) == {"timestamps", "feature_names", *NEW_ARMS}, "evaluation feature schema differs")
        old.compare_array(pd.DatetimeIndex(evaluation.timestamp).asi8, stored["timestamps"], "exact UTCns evaluation origins")
        old.compare_array(np.asarray(names), stored["feature_names"], "all 99 feature order")
        for arm in NEW_ARMS:
            old.compare_array(features[arm], stored[arm], f"all independent {arm} evaluation features")
        old.compare_array(np.isnan(stored[NEW_ARMS[0]]), np.isnan(stored[NEW_ARMS[1]]), "metadata/value missingness")


def validate_protected_inventory(root, hashes):
    old = bridge()
    for name in ("artifacts/cnemc_2025_deployment_candidate", "artifacts/strict_history_backtest", "project-showcase", "sites-haze/dist", "releases/runtime_20261002"):
        folder = (root / name).resolve()
        expected = {str(Path(path).resolve()) for path in hashes if Path(path).resolve().is_relative_to(folder)}
        actual = {str(path.resolve()) for path in folder.rglob("*") if path.is_file()}
        old._require(expected and actual == expected, f"protected exact inventory differs: {name}")


def validate_native_file(path, names):
    old = bridge()
    native = old._read_json(path)["learner"]
    old._require(int(native["learner_model_param"]["num_feature"]) == 99, "native feature count differs")
    old.compare_json(names, native["feature_names"], "native feature order")
    old._validate_native_trees(path, 99)


def reload_native_prediction(path, names, query):
    import xgboost as xgb

    old = bridge()
    old._require(len(names) == len(set(names)) == 99, "99 unique ordered model features required")
    old._require(np.asarray(query).ndim == 2 and np.asarray(query).shape[1] == 99, "99-column query required")
    validate_native_file(path, names)
    booster = xgb.Booster(params={"nthread": 1})
    booster.load_model(path)
    old._require(booster.num_boosted_rounds() == 500 and booster.num_features() == 99, "loaded model size differs")
    old.compare_json(names, booster.feature_names, "loaded model feature order")
    objective = json.loads(booster.save_config())["learner"]["objective"]
    old._require(objective["name"] == "survival:aft" and objective["aft_loss_param"]["aft_loss_distribution"] == "normal", "loaded objective differs")
    old._require(np.isclose(float(objective["aft_loss_param"]["aft_loss_distribution_scale"]), 1.2, rtol=1e-6), "loaded AFT scale differs")
    prediction = booster.predict(xgb.DMatrix(pd.DataFrame(query, columns=names), missing=np.nan))
    old._require(np.asarray(prediction).shape == (len(query),) and np.isfinite(prediction).all() and (prediction >= 0).all(), "invalid native predictions")
    return prediction


def validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes):
    old = bridge()
    groups = ("input_sha256", "protected_sha256", "code_sha256", "environment_sha256", "parent_sha256")
    old._require(all(isinstance(report.get(name), dict) and report[name] for name in (*groups, "frozen_sha256", "output_sha256")), "nonempty provenance groups required")
    old.compare_json(parent_inventory, report.get("parent_artifact_inventory"), "all 77 parent artifacts")
    old.compare_json(parent_hashes, report["parent_sha256"], "all parent artifact locks")
    old.compare_json(parent_report["frozen_sha256"], report.get("original_scope_sha256"), "parent frozen scope")
    for key, value in (("parent_report_sha256", PARENT_REPORT_SHA256), ("parent_verification_sha256", PARENT_VERIFICATION_SHA256),
                       ("parent_verification_report_sha256", PARENT_REPORT_SHA256), ("parent_verifier_sha256", FACTORS_SHA256)):
        old.compare_json(value, report.get(key), key)
    for key in ("historical_input_sha256", "historical_audit_drift", "configuration", "target", "uncertainty_scope"):
        old.compare_json(parent_report[key], report.get(key), f"unchanged {key}")
    old._require(Path(report["parent_run_dir"]).resolve() == parent and Path(report["run_dir"]).resolve() == run, "run identity differs")
    union = old.combine_locks([parent_report["frozen_sha256"], *[report[key] for key in groups]])
    old.compare_json(union, report["frozen_sha256"], "complete parent/child frozen union")
    for group in groups[:-1]:
        old._require(all(report[group].get(path) == digest for path, digest in parent_report[group].items()), f"parent {group} lost")
    folder = root / "experiments/web_recovery_gefs_proxy_20261003"
    required_code = [folder / name for name in ("PROTOCOL.md", "proxy_runner.py", "test_proxy_runner.py", "verify_proxy.py", "test_verify_proxy.py")]
    old._require(all(str(path.resolve()) in report["code_sha256"] for path in required_code), "five proxy source locks required")
    old._require(report["input_sha256"].get(str((root / SNAPSHOT_RELATIVE).resolve())) == SNAPSHOT_SHA256, "unique fixed snapshot not locked")
    snapshot_report_path = (root / SNAPSHOT_RELATIVE).with_name("report.json")
    old._require(old.sha256(snapshot_report_path) == SNAPSHOT_REPORT_SHA256, "snapshot source report changed")
    snapshot_report = old._read_json(snapshot_report_path)
    old._require(snapshot_report["output_sha256"]["snapshots.parquet"] == SNAPSHOT_SHA256, "snapshot output lock changed")
    sources = {path: digest for path, digest in snapshot_report["input_sha256"].items()
               if Path(path).resolve() != (root / "src/haze_forecast/app.py").resolve()}
    availability = root / "src/haze_forecast/operational_gefs_availability.py"
    sources.update({str(snapshot_report_path.resolve()): SNAPSHOT_REPORT_SHA256,
                    str(availability.resolve()): old.sha256(availability), str((root / SNAPSHOT_RELATIVE).resolve()): SNAPSHOT_SHA256})
    old.compare_json(SNAPSHOT_REPORT_SHA256, report.get("snapshot_report_sha256"), "declared snapshot source report")
    old.compare_json(sources, report.get("snapshot_source_sha256"), "complete snapshot source provenance except old app")
    for path, digest in sources.items():
        old._require(report["input_sha256"].get(path) == digest, "snapshot source omitted from input locks")
        if Path(path).suffix == ".py":
            old._require(report["code_sha256"].get(path) == digest, "snapshot source omitted from code locks")
    old.verify_hashes(sources, root)
    validate_protected_inventory(root, report["protected_sha256"])
    old.verify_hashes(union, root)
    required = {"population.parquet", "evaluation.parquet", "features.npz", "environment.json", "training_members.parquet",
                "models.json", "predictions.parquet", "process_scores.parquet", "scores.json"}
    validate_output_inventory(run, report["output_sha256"], required)
    old._require(report.get("schema_version") == 1 and report.get("status") == "pending_independent_verification", "run not pending independent verification")
    for key in ("production_ready", "deployed", "blind_test"):
        old._require(report.get(key) is False, f"{key} must remain false")
    for key in ("historical_shadow_evaluation", "archive_is_primary_reference", "protected_unchanged", "parent_unchanged"):
        old._require(report.get(key) is True, f"{key} must be true")
    old.compare_json(PRIMARY, report.get("primary_candidate"), "fixed primary weather candidate")
    old.compare_json(["metadata_control"], report.get("diagnostic_only_arms"), "diagnostic metadata arm")
    old.compare_json(list(NEW_ARMS), report.get("new_refitted_arms"), "two new arms")
    old.compare_json(list(INHERITED_ARMS), report.get("inherited_arms"), "inherited arms")
    old.compare_json([2024, 2025], report.get("years"), "historical years")
    old.compare_json("archive_timestamp_proxy", report.get("forecast_time_semantics"), "archive proxy time semantics")
    old.compare_json("offline archive-proxy GEFS values against metadata; no live availability claim", report.get("scope"), "offline scope")
    old.compare_json(old._fixed_parameters(), report.get("aft_parameters"), "fixed AFT parameters")
    return union


def verify_models(run, report, models, members, evaluation, grid, populations, features, names):
    old = bridge()
    validate_inventory(models["models"], evaluation)
    old._require(models.get("schema_version") == 1 and len(models["models"]) == report["fitted_models"] == 64, "all 64 models required")
    old.compare_json(names, models["feature_names"], "model feature order")
    old._require(set(members.model_id) == {record["model_id"] for record in models["models"]}, "foreign or missing member model")
    saved = pd.read_parquet(run / "predictions.parquet")
    old.compare_frame(evaluation, saved, list(evaluation), "saved prediction order")
    result = {arm: np.full(len(evaluation), np.nan) for arm in NEW_ARMS}
    checked = 0
    for record in models["models"]:
        arm, block = record["arm"], int(record["continuity_block"])
        cutoff = grid.loc[grid.continuity_block.eq(block), "timestamp"].min()
        old._require(pd.Timestamp(record["cutoff"]) == cutoff, "cutoff is not grid block start")
        population = populations[arm]
        expected = old.expected_members(population, cutoff, block)
        old._require(not expected.empty, "empty mature members")
        known = expected.known_at.dropna()
        old._require((known < cutoff).all(), "training weather not known before cutoff")
        wanted = factors().training_members(population, record)
        actual = members.loc[members.model_id.eq(record["model_id"])]
        old.compare_frame(wanted, actual, list(wanted), f"all mature members {record['model_id']}")
        positions = np.flatnonzero(evaluation.continuity_block.eq(block).to_numpy())
        checked += len(expected)
        old._require(record["training_member_path"] == "training_members.parquet", "wrong member artifact")
        old._require(record["parent_member_model_id"] == f"grid_all_legacy_origins_b{block}", "parent member identity differs")
        old._require(record["train_rows"] == len(expected) and record["eval_rows"] == len(positions), "model/query counts differ")
        old.compare_json(old._label_counts(expected), record["label_counts"], "model canonical label counts")
        old._require(pd.Timestamp(record["training_last_timestamp"]) == expected.timestamp.max() and pd.Timestamp(record["max_label_available_at"]) == expected.label_available_at.max(), "maturity extrema differ")
        old._require(record["training_features_sha256"] == old.matrix_digest(expected[names].to_numpy(dtype=float)), "all 99 training matrix digest differs")
        timestamps_hash = hashlib.sha256(pd.DatetimeIndex(expected.timestamp).asi8.astype("<i8").tobytes()).hexdigest()
        old._require(record["training_timestamps_sha256"] == timestamps_hash, "training timestamp digest differs")
        for key in ("feature_names", "booster_feature_names"):
            old.compare_json(names, record[key], key)
        old.compare_json(report["configuration"], record["configuration"], "model fixed configuration")
        old.compare_json(old._fixed_parameters(), record["aft_parameters"], "model AFT parameters")
        old._require(record["num_boosted_rounds"] == 500, "model rounds differ")
        old.compare_json("global archive-end labels only; all mature rows retained", record.get("right_censor_policy"), "canonical right-censor policy")
        old.validate_trained_config(record["actual_trained_booster_config"], 99)
        old._require(record["model_path"] == f"models/{record['model_id']}.json", "model path/identity differs")
        path = old.run_file(run, record["model_path"])
        old._require(report["output_sha256"].get(record["model_path"]) == record["model_sha256"] == old.sha256(path), "native model lock differs")
        prediction = reload_native_prediction(path, names, features[arm][positions])
        old.compare_array(prediction, saved.loc[positions, arm].to_numpy(dtype=float), f"all native predictions {record['model_id']}")
        result[arm][positions] = prediction
    old._require(checked == len(members) == 533044, "total member count differs")
    return result, checked


def _verify_complete_run(run, root):
    old, factor = bridge(), factors()
    parent = (root / PARENT_RELATIVE).resolve()
    parent_report, parent_inventory, parent_hashes = validate_parent_chain(parent, root)
    report_hash = old.sha256(run / "report.json")
    report = old._read_json(run / "report.json")
    frozen = validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes)
    old._verify_environment(old._read_json(run / "environment.json"), report["environment_sha256"])
    names = old._read_json(root / "artifacts/cnemc_2025_deployment_candidate/model_metadata.json")["feature_names"]
    old._require(len(names) == len(set(names)) == 87, "87 original ordered features required")
    grid = old.rebuild_grid({source: pd.read_parquet(root / path) for source, path in old.RAW_SOURCES.items()})
    grid_columns = ["timestamp", *names, "continuity_block", "episode_id", "eligible", "pollution_state", "input_hour_observed"]
    old.compare_frame(grid, pd.read_parquet(root / "artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet"), grid_columns, "complete raw rebuilt grid")
    legacy = old.rebuild_legacy({source: pd.read_parquet(root / path) for source, path in old.LEGACY_SOURCES.items()}, grid, names)
    base = factor.grid_population(legacy, grid, names)
    old.compare_frame(base, pd.read_parquet(parent / "grid_population.parquet"), list(base), "parent all origins realhour features canonical bounds")
    old.compare_frame(legacy, pd.read_parquet(parent / "legacy_population.parquet"), list(legacy), "parent legacy identity")
    source = root / SNAPSHOT_RELATIVE
    old._require(old.sha256(source) == SNAPSHOT_SHA256, "fixed snapshot SHA differs")
    snapshots = pd.read_parquet(source)
    old._require(len(snapshots) == 548, "fixed snapshot row count differs")
    true, control, all_names = rebuild_population(base, snapshots, names)
    old._require(len(all_names) == 99, "99 columns required")
    old.compare_json(all_names, report["feature_names"], "report feature order")
    validate_population(true, pd.read_parquet(run / "population.parquet"), base)
    archive = pd.read_parquet(root / "artifacts/strict_history_backtest/predictions.parquet")
    archive["timestamp"] = pd.to_datetime(archive.timestamp, utc=True).dt.tz_convert("Asia/Shanghai")
    archive = archive.loc[archive.timestamp.dt.year.isin((2024, 2025))].sort_values("timestamp").reset_index(drop=True)
    evaluation = old.rebuild_evaluation(archive, grid)
    for folder in (parent, run):
        old.compare_frame(evaluation, pd.read_parquet(folder / "evaluation.parquet"), list(evaluation), "all fixed 701 evaluation origins and bounds")
    old._require((len(grid), len(base), len(evaluation), evaluation.continuity_block.nunique()) == (102024, 8638, 701, 32), "fixed population/block counts differ")
    old._require(old.sha256(run / "evaluation.parquet") == old.sha256(parent / "evaluation.parquet"), "parent evaluation must be copied unchanged")
    for key, value in parent_report["counts"].items():
        old.compare_json(value, report["counts"].get(key), f"unchanged count {key}")
    features = {arm: population.set_index("timestamp").loc[evaluation.timestamp, all_names].to_numpy(dtype=float)
                for arm, population in (("metadata_control", control), ("weather_values", true))}
    validate_feature_archive(run / "features.npz", evaluation, all_names, features)
    inherited = pd.read_parquet(parent / "predictions.parquet")
    old.compare_frame(evaluation, inherited, list(evaluation), "parent predicted cohort identity")
    parent_predictions = {arm: inherited[arm].to_numpy(dtype=float) for arm in factor.ARMS}
    parent_scores, parent_process = factor.compute_scores(evaluation, parent_predictions)
    old.compare_json(parent_scores, parent_report["metrics"], "independently recomputed parent scores")
    old.compare_frame(parent_process, pd.read_parquet(parent / "process_scores.parquet"), list(parent_process), "parent process scores")
    predictions = {arm: parent_predictions[arm] for arm in INHERITED_ARMS}
    old.compare_array(predictions["archive"], evaluation.archive_prediction_hours, "unchanged archive values")
    models, members = old._read_json(run / "models.json"), pd.read_parquet(run / "training_members.parquet")
    validate_same_members(members, pd.read_parquet(parent / "training_members.parquet"))
    inferred, member_count = verify_models(run, report, models, members, evaluation, grid,
                                          {"metadata_control": control, "weather_values": true}, features, all_names)
    old.compare_json({arm: dict(zip(all_names, np.isnan(features[arm]).mean(axis=0).tolist())) for arm in NEW_ARMS}, report["feature_missingness"], "evaluation missingness")
    maturity = [{key: record[key] for key in ("model_id", "train_rows", "label_counts", "training_last_timestamp", "max_label_available_at")}
                for record in models["models"]]
    old.compare_json(maturity, report["training_maturity_summary"], "all maturity summary")
    predictions.update(inferred)
    rows = prediction_rows(evaluation, predictions)
    old.compare_frame(rows, pd.read_parquet(run / "predictions.parquet"), list(rows), "all predictions and uncapped paired bounds")
    scores, process = compute_scores(evaluation, predictions)
    old.compare_json(scores, old._read_json(run / "scores.json"), "independent four-arm metrics")
    old.compare_json(scores, report["metrics"], "report independent metrics")
    old.compare_frame(process, pd.read_parquet(run / "process_scores.parquet"), list(process), "all independent process scores")
    old._require(report.get("accuracy_passed") is scores["gates"]["passed"], "accuracy status is not combined independent gates")
    old.verify_hashes(frozen, root)
    validate_protected_inventory(root, report["protected_sha256"])
    factor.validate_artifact_locks(parent, parent_inventory, parent_hashes)
    old._require(old.sha256(run / "report.json") == report_hash, "report changed during verification")
    validate_output_inventory(run, report["output_sha256"], set(report["output_sha256"]))
    result = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
              "run_dir": str(run), "report_sha256": report_hash, "verifier_sha256": old.sha256(__file__),
              "reused_independent_verifier_sha256": BRIDGE_SHA256, "parent_report_sha256": PARENT_REPORT_SHA256,
              "parent_verification_sha256": PARENT_VERIFICATION_SHA256, "snapshot_sha256": SNAPSHOT_SHA256,
              "grid_rows_reconstructed": len(grid), "population_origins_reconstructed": len(true),
              "fixed_origins_verified": len(evaluation), "models_verified": len(models["models"]),
              "training_member_rows_verified": member_count, "refitted_predictions_verified": 2 * len(evaluation),
              "feature_order": all_names, "metrics": scores, "accuracy_passed": scores["gates"]["passed"],
              "primary_candidate": PRIMARY, "production_ready": False, "deployed": False, "blind_test": False,
              "limits": ["Offline historical archive timestamp proxy; matching does not authenticate first public availability",
                         "Known historical years are not a blind test or a real-time accuracy claim",
                         "Shared-truth error bounds are not statistical confidence intervals",
                         "Native replay and provenance checks are not independent retraining",
                         "Weather-value contrast is conditional on the same metadata/missingness, not unique causal attribution"]}
    old.write_exclusive_json(run / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing completed offline proxy run")
    args = parser.parse_args()
    old = bridge()
    run = old.validate_run_directory(args.run_dir, Path(__file__).resolve().parent)
    old._require(not (run / "verification.json").exists(), "verification.json already exists; no overwrite")
    try:
        result = verify_run(run)
    except Exception as exc:
        if not (run / "verification.json").exists():
            old.write_exclusive_json(run / "verification.json", {"schema_version": 1, "status": "failed",
                "independent_verification_passed": False, "error": str(exc), "verifier_sha256": old.sha256(__file__)})
        raise
    print(json.dumps({key: result[key] for key in ("status", "models_verified", "fixed_origins_verified", "accuracy_passed", "production_ready")}))


if __name__ == "__main__":
    main()
