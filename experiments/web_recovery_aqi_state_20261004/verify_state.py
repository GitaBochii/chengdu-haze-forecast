"""Independent full-grid AQI-state reconstruction and native replay, never fitting.

Weather, canonical bounds and scoring reuse the hash-frozen independent proxy,
bridge and factors verifiers. AQI state uses a separate vectorized segment
algorithm, not the new runner or any production state helper.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from functools import lru_cache
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

PROXY_SHA256 = "840b2e8308eb5084ff8d2ac64e5ae9150af3c57621e00fe591b443d065ff1457"
PARENT_REPORT_SHA256 = "d553ab51dbb596c0c01960c340701386ac87635b21d611e4724ab662f3b7555b"
PARENT_VERIFICATION_SHA256 = "b11f7163cb14a8225995dc1ebe8c46761d9f7e053a876d8f99c3cee9bd316807"
PARENT_RELATIVE = "experiments/web_recovery_gefs_proxy_20261003/runs/20261003T145140578727Z"
EXPERIMENT = "experiments/web_recovery_aqi_state_20261004"
GRID_RELATIVE = "artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet"
GRID_SHA256 = "52efbd1bfef81643c317c193895104baff61a3eee1b0ca465066e3e39cec473e"
STATE_FIELDS = ("aqi_observed_high_run_hours", "aqi_high_run_start_observed")
INHERITED_ARMS = ("archive", "weather_values")
NEW_ARMS = ("state_control", "state_values")
ARMS = (*INHERITED_ARMS, *NEW_ARMS)
PRIMARY = "state_values"
PAIRS = {**{f"{arm}_vs_archive": ("archive", arm) for arm in ARMS[1:]},
         "state_values_vs_state_control": ("state_control", "state_values"),
         "state_values_vs_weather_values": ("weather_values", "state_values")}
SCOPE = "offline AQI observed-state against equal-dimensional control; no live input claim"
STATE_TIME_SEMANTICS = "observed high-AQI run lower bound; start flag only from immediately preceding observed clean hour"
STATE_SCOPE = "historical full-grid replay only; short-sequence and live inputs require separate validation"


@lru_cache(maxsize=1)
def proxy():
    path = Path(__file__).resolve().parents[1] / "web_recovery_gefs_proxy_20261003/verify_proxy.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROXY_SHA256:
        raise ValueError("frozen independent proxy verifier changed")
    spec = importlib.util.spec_from_file_location("frozen_independent_aqi_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bridge():
    return proxy().bridge()


def factors():
    return proxy().factors()


def observed_aqi_state(frame):
    old = bridge()
    old._require({"timestamp", "aqi"}.issubset(frame), "timestamp and AQI are required")
    times = old._hourly_times(frame.timestamp)
    old._require(times.equals(pd.date_range(times[0], times[-1], freq="h")), "state requires an ordered complete hourly grid")
    series = frame.aqi
    old._require(not pd.api.types.is_bool_dtype(series.dtype) and not pd.api.types.is_complex_dtype(series.dtype), "AQI must be real nonboolean observations")
    for value in series:
        old._require(not isinstance(value, (bool, np.bool_, complex, np.complexfloating)), "AQI must be real nonboolean observations")
        if value is None or value is pd.NA:
            continue
        old._require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)), "AQI must be real numeric observations")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    old._require(not np.isinf(values).any(), "infinite AQI observation")
    high = np.isfinite(values) & (values > 100)
    positions = np.arange(len(values), dtype=np.int64)
    last_break = np.maximum.accumulate(np.where(~high, positions, -1))
    run = np.where(high, positions - last_break, np.where(np.isfinite(values), 0., np.nan))
    start_flag = np.full(len(values), np.nan, dtype=float)
    high_positions = positions[high]
    predecessors = last_break[high]
    observed = predecessors >= 0
    flags = np.zeros(len(high_positions), dtype=float)
    prior_values = values[predecessors[observed]]
    flags[observed] = (np.isfinite(prior_values) & (prior_values <= 100)).astype(float)
    start_flag[high] = flags
    return pd.DataFrame({"timestamp": times, STATE_FIELDS[0]: run.astype(float), STATE_FIELDS[1]: start_flag})


def rebuild_population(parent_population, grid, parent_names):
    old = bridge()
    old._require(len(parent_names) == len(set(parent_names)) == 99, "99 original feature names required")
    old._require(not set(STATE_FIELDS).intersection(parent_population) and not set(STATE_FIELDS).intersection(parent_names), "state feature name collision")
    old._require(parent_population.timestamp.is_unique, "duplicate population origin")
    state = observed_aqi_state(grid[["timestamp", "aqi"]])
    positions = pd.DatetimeIndex(state.timestamp).get_indexer(parent_population.timestamp)
    old._require((positions >= 0).all(), "population outside complete observed state grid")
    actual = parent_population.copy().reset_index(drop=True)
    for name in STATE_FIELDS:
        actual[name] = state.iloc[positions][name].to_numpy(dtype=float)
    control = actual.copy()
    for name in STATE_FIELDS:
        values = actual[name].to_numpy(dtype=float)
        control[name] = np.where(np.isfinite(values), 0., np.nan)
    names = [*parent_names, *STATE_FIELDS]
    old.compare_array(np.isnan(actual[names]), np.isnan(control[names]), "two-arm identical missingness")
    old.compare_frame(parent_population, actual, list(parent_population), "unchanged parent 99 features and metadata")
    return actual, control, names


def validate_population(expected, actual, parent_population):
    old = bridge()
    old._require(set(actual) == set(expected), "state population exact schema differs")
    old.compare_frame(parent_population, actual, list(parent_population), "unchanged original parent population")
    old.compare_frame(expected, actual, list(expected), "all independently reconstructed observed state")


def validate_inventory(records, evaluation):
    old = bridge()
    wanted = {(arm, int(block)) for block in evaluation.continuity_block.unique() for arm in NEW_ARMS}
    actual = [(record["arm"], int(record["continuity_block"])) for record in records]
    old._require(len(actual) == len(wanted) and set(actual) == wanted, "state model inventory mismatch")
    ids = [record["model_id"] for record in records]
    old._require(len(ids) == len(set(ids)), "duplicate model identity")
    for record in records:
        old._require(record["model_id"] == f"{record['arm']}_b{record['continuity_block']}", "model identity mismatch")


def information_gate(control, candidate, paired):
    return proxy().information_gate(control, candidate, paired)


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
        information = information_gate(metrics["arms"]["state_control"], metrics["arms"][PRIMARY], metrics["paired"]["state_values_vs_state_control"])
        gates[str(year)] = {"primary": primary, "state_information": information,
                           "passed": primary["passed"] and information["passed"],
                           "failure_reasons": [f"{name}:{key}" for name, gate in (("primary", primary), ("state_information", information)) for key in gate["failure_reasons"]]}
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


def expected_output_inventory(evaluation):
    names = {"population.parquet", "evaluation.parquet", "features.npz", "environment.json", "training_members.parquet",
             "models.json", "predictions.parquet", "process_scores.parquet", "scores.json", "execution_record_at_start.md"}
    return names | {f"models/{arm}_b{int(block)}.json" for arm in NEW_ARMS for block in evaluation.continuity_block.unique()}


def validate_parent_chain(parent, root):
    old, p, factor = bridge(), proxy(), factors()
    parent = Path(parent).resolve()
    old._require(parent == (root / PARENT_RELATIVE).resolve(), "parent is not predeclared proxy run")
    report, verification = (old._read_json(parent / name) for name in ("report.json", "verification.json"))
    old._require(old.sha256(parent / "report.json") == PARENT_REPORT_SHA256, "parent report hash differs")
    old._require(old.sha256(parent / "verification.json") == PARENT_VERIFICATION_SHA256, "parent verification hash differs")
    expected = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
                "report_sha256": PARENT_REPORT_SHA256, "verifier_sha256": PROXY_SHA256,
                "fixed_origins_verified": 701, "models_verified": 64, "training_member_rows_verified": 533044,
                "refitted_predictions_verified": 1402, "production_ready": False, "deployed": False}
    for key, value in expected.items():
        old.compare_json(value, verification.get(key), f"parent passed evidence {key}")
    old._require(Path(verification["run_dir"]).resolve() == parent, "parent verified identity differs")
    old.compare_json(report["metrics"], verification["metrics"], "parent verified metrics")
    old._require(report.get("status") == "pending_independent_verification", "parent historical status differs")
    inventory, hashes = factor.artifact_locks(parent)
    old._require(len(inventory) == 76 and set(inventory) == set(report["output_sha256"]) | {"report.json", "verification.json"}, "parent 76 artifact inventory differs")
    old._require(len(report["frozen_sha256"]) == 621, "parent 621 frozen locks differ")
    old.verify_hashes(report["output_sha256"], parent)
    old.verify_hashes(report["frozen_sha256"], root)
    old._verify_environment(old._read_json(parent / "environment.json"), report["environment_sha256"])
    p.validate_protected_inventory(root, report["protected_sha256"])
    older_parent = Path(report["parent_run_dir"]).resolve()
    older_report, older_inventory, older_hashes = p.validate_parent_chain(older_parent, root)
    old.compare_json(older_inventory, report["parent_artifact_inventory"], "proxy parent's factor inventory")
    old.compare_json(older_hashes, report["parent_sha256"], "proxy parent's factor locks")
    old.compare_json(older_report["frozen_sha256"], report["original_scope_sha256"], "proxy parent's original scope")
    return report, inventory, hashes


def validate_same_members(members, parent_members):
    old = bridge()
    old._require(set(members.arm) == set(NEW_ARMS), "exact two training member arms required")
    parent = parent_members.loc[parent_members.arm.eq("weather_values")]
    old._require(not parent.empty, "parent weather members missing")
    columns = [name for name in parent if name not in ("model_id", "arm")]
    keys = ["evaluation_block", "population_row"]
    old._require(not parent.duplicated(keys).any(), "duplicate parent members")
    for arm in NEW_ARMS:
        actual = members.loc[members.arm.eq(arm)]
        old._require(not actual.duplicated(keys).any(), "duplicate state members")
        old.compare_frame(parent.sort_values(keys), actual.sort_values(keys), columns, f"same parent mature members {arm}")


def validate_feature_archive(path, evaluation, names, features):
    old = bridge()
    old._require(len(names) == len(set(names)) == 101 and names[-2:] == list(STATE_FIELDS), "101 ordered features ending in state required")
    with np.load(path, allow_pickle=False) as stored:
        old._require(set(stored.files) == {"timestamps", "feature_names", *NEW_ARMS}, "evaluation feature exact schema differs")
        old.compare_array(pd.DatetimeIndex(evaluation.timestamp).asi8, stored["timestamps"], "evaluation UTCns origins")
        old.compare_array(np.asarray(names), stored["feature_names"], "all 101 feature order")
        for arm in NEW_ARMS:
            old.compare_array(features[arm], stored[arm], f"all independent {arm} features")
        old.compare_array(np.isnan(stored[NEW_ARMS[0]]), np.isnan(stored[NEW_ARMS[1]]), "state two-arm missingness")


def validate_native_file(path, names):
    old = bridge()
    native = old._read_json(path)["learner"]
    old._require(int(native["learner_model_param"]["num_feature"]) == 101, "native feature count differs")
    old.compare_json(names, native["feature_names"], "native feature order")
    old._validate_native_trees(path, 101)


def reload_native_prediction(path, names, query):
    import xgboost as xgb

    old = bridge()
    old._require(len(names) == len(set(names)) == 101, "101 unique ordered features required")
    old._require(np.asarray(query).ndim == 2 and np.asarray(query).shape[1] == 101, "101-column query required")
    validate_native_file(path, names)
    booster = xgb.Booster(params={"nthread": 1})
    booster.load_model(path)
    old._require(booster.num_boosted_rounds() == 500 and booster.num_features() == 101, "loaded model size differs")
    old.compare_json(names, booster.feature_names, "loaded model feature order")
    objective = json.loads(booster.save_config())["learner"]["objective"]
    old._require(objective["name"] == "survival:aft" and objective["aft_loss_param"]["aft_loss_distribution"] == "normal", "loaded objective differs")
    old._require(np.isclose(float(objective["aft_loss_param"]["aft_loss_distribution_scale"]), 1.2, rtol=1e-6), "loaded AFT scale differs")
    prediction = booster.predict(xgb.DMatrix(pd.DataFrame(query, columns=names), missing=np.nan))
    old._require(np.asarray(prediction).shape == (len(query),) and np.isfinite(prediction).all() and (prediction >= 0).all(), "invalid native predictions")
    return prediction


def validate_scope(report):
    old = bridge()
    expected = {"schema_version": 1, "status": "pending_independent_verification", "production_ready": False,
                "deployed": False, "blind_test": False, "historical_shadow_evaluation": True,
                "archive_is_primary_reference": True, "protected_unchanged": True, "parent_unchanged": True,
                "primary_candidate": PRIMARY, "diagnostic_only_arms": ["state_control", "weather_values"], "new_refitted_arms": list(NEW_ARMS),
                "inherited_arms": list(INHERITED_ARMS), "years": [2024, 2025], "forecast_time_semantics": "archive_timestamp_proxy",
                "scope": SCOPE, "state_time_semantics": STATE_TIME_SEMANTICS, "state_scope": STATE_SCOPE,
                "state_fields": list(STATE_FIELDS)}
    for key, value in expected.items():
        if isinstance(value, bool):
            old._require(report.get(key) is value, f"scope {key} must be boolean {value}")
        else:
            old.compare_json(value, report.get(key), f"fixed scope {key}")


def validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes):
    old = bridge()
    groups = ("input_sha256", "protected_sha256", "code_sha256", "environment_sha256", "parent_sha256")
    old._require(all(isinstance(report.get(name), dict) and report[name] for name in (*groups, "frozen_sha256", "output_sha256")), "nonempty provenance groups required")
    old.compare_json(parent_inventory, report.get("parent_artifact_inventory"), "all 76 parent artifacts")
    old.compare_json(parent_hashes, report["parent_sha256"], "all parent artifact locks")
    old.compare_json(parent_report["frozen_sha256"], report.get("original_scope_sha256"), "parent frozen scope")
    for key, value in (("parent_report_sha256", PARENT_REPORT_SHA256), ("parent_verification_sha256", PARENT_VERIFICATION_SHA256),
                       ("parent_verification_report_sha256", PARENT_REPORT_SHA256), ("parent_verifier_sha256", PROXY_SHA256)):
        old.compare_json(value, report.get(key), key)
    for key in ("historical_input_sha256", "historical_audit_drift", "configuration", "target", "uncertainty_scope", "snapshot_report_sha256", "snapshot_source_sha256"):
        old.compare_json(parent_report[key], report.get(key), f"unchanged {key}")
    old._require(Path(report["parent_run_dir"]).resolve() == parent and Path(report["run_dir"]).resolve() == run, "run identity differs")
    union = old.combine_locks([parent_report["frozen_sha256"], *[report[key] for key in groups]])
    old.compare_json(union, report["frozen_sha256"], "complete parent and child frozen union")
    for group in groups[:-1]:
        old._require(all(report[group].get(path) == digest for path, digest in parent_report[group].items()), f"parent {group} lost")
    folder = root / EXPERIMENT
    source_names = ("PROTOCOL.md", "aqi_state_runner.py", "test_aqi_state_runner.py", "verify_state.py", "test_verify_state.py")
    old._require(all(str((folder / name).resolve()) in report["code_sha256"] for name in source_names), "five new source locks required")
    old._require(report["input_sha256"].get(str((root / GRID_RELATIVE).resolve())) == GRID_SHA256, "fixed source grid not locked")
    proxy().validate_protected_inventory(root, report["protected_sha256"])
    old.verify_hashes(union, root)
    required = expected_output_inventory(pd.read_parquet(parent / "evaluation.parquet"))
    old._require(set(report["output_sha256"]) == required and len(required) == 74, "exact 64 model and ten artifact output inventory required")
    validate_output_inventory(run, report["output_sha256"], required)
    validate_scope(report)
    old.compare_json(old._fixed_parameters(), report.get("aft_parameters"), "fixed AFT parameters")
    return union


def reconstruct_inputs(root, parent):
    """No model fitting or new-run write; reusable independent preflight evidence."""
    old, p, factor = bridge(), proxy(), factors()
    names = old._read_json(root / "artifacts/cnemc_2025_deployment_candidate/model_metadata.json")["feature_names"]
    old._require(len(names) == len(set(names)) == 87, "87 original ordered features required")
    grid = old.rebuild_grid({source: pd.read_parquet(root / path) for source, path in old.RAW_SOURCES.items()})
    grid_columns = ["timestamp", *names, "continuity_block", "episode_id", "eligible", "pollution_state", "input_hour_observed"]
    old._require(old.sha256(root / GRID_RELATIVE) == GRID_SHA256, "fixed source grid hash differs")
    old.compare_frame(grid, pd.read_parquet(root / GRID_RELATIVE), grid_columns, "complete grid rebuilt from five raw sources")
    legacy = old.rebuild_legacy({source: pd.read_parquet(root / path) for source, path in old.LEGACY_SOURCES.items()}, grid, names)
    base = factor.grid_population(legacy, grid, names)
    snapshots = pd.read_parquet(root / p.SNAPSHOT_RELATIVE)
    old._require(old.sha256(root / p.SNAPSHOT_RELATIVE) == p.SNAPSHOT_SHA256 and len(snapshots) == 548, "fixed 548 snapshot source differs")
    parent_true, parent_control, parent_names = p.rebuild_population(base, snapshots, names)
    old._require(len(parent_names) == 99, "99 reconstructed parent features required")
    old.compare_json(parent_names, old._read_json(parent / "report.json")["feature_names"], "parent exact feature order")
    p.validate_population(parent_true, pd.read_parquet(parent / "population.parquet"), base)
    archive = pd.read_parquet(root / "artifacts/strict_history_backtest/predictions.parquet")
    archive["timestamp"] = pd.to_datetime(archive.timestamp, utc=True).dt.tz_convert("Asia/Shanghai")
    archive = archive.loc[archive.timestamp.dt.year.isin((2024, 2025))].sort_values("timestamp").reset_index(drop=True)
    evaluation = old.rebuild_evaluation(archive, grid)
    old.compare_frame(evaluation, pd.read_parquet(parent / "evaluation.parquet"), list(evaluation), "parent all 701 evaluation origins and fullT bounds")
    old._require((len(grid), len(base), len(evaluation), evaluation.continuity_block.nunique()) == (102024, 8638, 701, 32), "fixed population and block counts differ")
    parent_features = {arm: population.set_index("timestamp").loc[evaluation.timestamp, parent_names].to_numpy(dtype=float)
                       for arm, population in (("metadata_control", parent_control), ("weather_values", parent_true))}
    p.validate_feature_archive(parent / "features.npz", evaluation, parent_names, parent_features)
    actual, control, all_names = rebuild_population(parent_true, grid, parent_names)
    features = {arm: population.set_index("timestamp").loc[evaluation.timestamp, all_names].to_numpy(dtype=float)
                for arm, population in (("state_control", control), ("state_values", actual))}
    for arm in NEW_ARMS:
        old.compare_array(parent_features["weather_values"], features[arm][:, :99], f"unaltered parent 99 evaluation columns {arm}")
    old.compare_array(np.isnan(features[NEW_ARMS[0]]), np.isnan(features[NEW_ARMS[1]]), "all 701 new-arm missingness identical")
    return grid, actual, control, evaluation, all_names, features


def verify_models(run, report, models, members, evaluation, grid, populations, features, names):
    old = bridge()
    validate_inventory(models["models"], evaluation)
    old._require(models.get("schema_version") == 1 and len(models["models"]) == report["fitted_models"] == 64, "all 64 models required")
    old.compare_json(names, models["feature_names"], "model feature order")
    old._require(set(members.model_id) == {record["model_id"] for record in models["models"]}, "foreign or missing member model")
    saved = pd.read_parquet(run / "predictions.parquet")
    old.compare_frame(evaluation, saved, list(evaluation), "saved prediction identity and order")
    result = {arm: np.full(len(evaluation), np.nan) for arm in NEW_ARMS}
    checked, arm_counts = 0, dict.fromkeys(NEW_ARMS, 0)
    for record in models["models"]:
        arm, block = record["arm"], int(record["continuity_block"])
        cutoff = grid.loc[grid.continuity_block.eq(block), "timestamp"].min()
        old._require(pd.Timestamp(record["cutoff"]) == cutoff, "cutoff is not true grid block start")
        population = populations[arm]
        expected = old.expected_members(population, cutoff, block)
        old._require(not expected.empty, "empty mature members")
        old._require((expected.known_at.dropna() < cutoff).all(), "training weather not known before cutoff")
        wanted = factors().training_members(population, record)
        actual = members.loc[members.model_id.eq(record["model_id"])]
        old._require(set(actual) == set(wanted), "training member exact schema differs")
        old.compare_frame(wanted, actual, list(wanted), f"all mature members {record['model_id']}")
        positions = np.flatnonzero(evaluation.continuity_block.eq(block).to_numpy())
        checked += len(expected)
        arm_counts[arm] += len(expected)
        old._require(record["training_member_path"] == "training_members.parquet", "wrong member artifact")
        old._require(record["parent_member_model_id"] == f"weather_values_b{block}", "parent weather member identity differs")
        old._require(record["train_rows"] == len(expected) and record["eval_rows"] == len(positions), "model/query counts differ")
        old.compare_json(old._label_counts(expected), record["label_counts"], "model canonical label counts")
        old._require(pd.Timestamp(record["training_last_timestamp"]) == expected.timestamp.max() and pd.Timestamp(record["max_label_available_at"]) == expected.label_available_at.max(), "maturity extrema differ")
        old._require(record["training_features_sha256"] == old.matrix_digest(expected[names].to_numpy(dtype=float)), "all 101 training matrix digest differs")
        timestamps_hash = hashlib.sha256(pd.DatetimeIndex(expected.timestamp).asi8.astype("<i8").tobytes()).hexdigest()
        old._require(record["training_timestamps_sha256"] == timestamps_hash, "training timestamp digest differs")
        for key in ("feature_names", "booster_feature_names"):
            old.compare_json(names, record[key], key)
        old.compare_json(report["configuration"], record["configuration"], "model fixed configuration")
        old.compare_json(old._fixed_parameters(), record["aft_parameters"], "model AFT parameters")
        old._require(record["num_boosted_rounds"] == 500, "model rounds differ")
        old.compare_json("global archive-end labels only; all mature rows retained", record.get("right_censor_policy"), "canonical censor policy")
        old.validate_trained_config(record["actual_trained_booster_config"], 101)
        old._require(record["model_path"] == f"models/{record['model_id']}.json", "native model path/identity differs")
        path = old.run_file(run, record["model_path"])
        old._require(report["output_sha256"].get(record["model_path"]) == record["model_sha256"] == old.sha256(path), "native model lock differs")
        prediction = reload_native_prediction(path, names, features[arm][positions])
        old.compare_array(prediction, saved.loc[positions, arm].to_numpy(dtype=float), f"all native predictions {record['model_id']}")
        result[arm][positions] = prediction
    old._require(checked == len(members) == 533044 and all(count == 266522 for count in arm_counts.values()), "mature member counts differ")
    return result, checked


def verify_run(run, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    old = bridge()
    run = old.validate_run_directory(run, root / EXPERIMENT)
    old._require(not (run / "verification.json").exists(), "verification.json already exists; no overwrite")
    parent = (root / PARENT_RELATIVE).resolve()
    parent_report, parent_inventory, parent_hashes = validate_parent_chain(parent, root)
    report_hash = old.sha256(run / "report.json")
    report = old._read_json(run / "report.json")
    frozen = validate_report(report, run, root, parent, parent_report, parent_inventory, parent_hashes)
    old._verify_environment(old._read_json(run / "environment.json"), report["environment_sha256"])
    grid, true, control, evaluation, names, features = reconstruct_inputs(root, parent)
    old.compare_json(names, report["feature_names"], "report 101 ordered features")
    validate_population(true, pd.read_parquet(run / "population.parquet"), pd.read_parquet(parent / "population.parquet"))
    old.compare_frame(evaluation, pd.read_parquet(run / "evaluation.parquet"), list(evaluation), "all fixed 701 new evaluation origins and bounds")
    old._require(old.sha256(run / "evaluation.parquet") == old.sha256(parent / "evaluation.parquet"), "parent evaluation copied unchanged")
    for key, value in parent_report["counts"].items():
        old.compare_json(value, report["counts"].get(key), f"unchanged count {key}")
    validate_feature_archive(run / "features.npz", evaluation, names, features)
    inherited = pd.read_parquet(parent / "predictions.parquet")
    old.compare_frame(evaluation, inherited, list(evaluation), "parent predicted cohort identity")
    parent_predictions = {arm: inherited[arm].to_numpy(dtype=float) for arm in proxy().ARMS}
    parent_scores, parent_process = proxy().compute_scores(evaluation, parent_predictions)
    old.compare_json(parent_scores, parent_report["metrics"], "independently recomputed parent scores")
    old.compare_frame(parent_process, pd.read_parquet(parent / "process_scores.parquet"), list(parent_process), "parent process scores")
    predictions = {arm: parent_predictions[arm] for arm in INHERITED_ARMS}
    old.compare_array(predictions["archive"], evaluation.archive_prediction_hours, "unchanged archive values")
    models, members = old._read_json(run / "models.json"), pd.read_parquet(run / "training_members.parquet")
    validate_same_members(members, pd.read_parquet(parent / "training_members.parquet"))
    inferred, member_count = verify_models(run, report, models, members, evaluation, grid,
                                          {"state_control": control, "state_values": true}, features, names)
    old.compare_json({arm: dict(zip(names, np.isnan(features[arm]).mean(axis=0).tolist())) for arm in NEW_ARMS}, report["feature_missingness"], "evaluation missingness")
    maturity = [{key: record[key] for key in ("model_id", "train_rows", "label_counts", "training_last_timestamp", "max_label_available_at")}
                for record in models["models"]]
    old.compare_json(maturity, report["training_maturity_summary"], "all maturity summary")
    predictions.update(inferred)
    rows = prediction_rows(evaluation, predictions)
    stored_rows = pd.read_parquet(run / "predictions.parquet")
    old._require(set(rows) == set(stored_rows), "prediction exact schema differs")
    old.compare_frame(rows, stored_rows, list(rows), "all predictions and fullT paired bounds")
    scores, process = compute_scores(evaluation, predictions)
    old.compare_json(scores, old._read_json(run / "scores.json"), "independent four-arm metrics")
    old.compare_json(scores, report["metrics"], "report independently recomputed metrics")
    old.compare_frame(process, pd.read_parquet(run / "process_scores.parquet"), list(process), "independent process scores")
    old._require(report.get("accuracy_passed") is scores["gates"]["passed"], "accuracy status is not both independent annual gates")
    old.verify_hashes(frozen, root)
    proxy().validate_protected_inventory(root, report["protected_sha256"])
    factors().validate_artifact_locks(parent, parent_inventory, parent_hashes)
    old._require(old.sha256(run / "report.json") == report_hash, "report changed during verification")
    validate_output_inventory(run, report["output_sha256"], set(report["output_sha256"]))
    result = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
              "run_dir": str(run), "report_sha256": report_hash, "verifier_sha256": old.sha256(__file__),
              "reused_independent_verifier_sha256": PROXY_SHA256, "parent_report_sha256": PARENT_REPORT_SHA256,
              "parent_verification_sha256": PARENT_VERIFICATION_SHA256,
              "grid_rows_reconstructed": len(grid), "population_origins_reconstructed": len(true),
              "fixed_origins_verified": len(evaluation), "models_verified": len(models["models"]),
              "training_member_rows_verified": member_count, "refitted_predictions_verified": 2 * len(evaluation),
              "feature_order": names, "metrics": scores, "accuracy_passed": scores["gates"]["passed"],
              "primary_candidate": PRIMARY, "production_ready": False, "deployed": False, "blind_test": False,
              "limits": ["Historical full-grid replay only; no short-sequence or live-input accuracy claim",
                         "Observed high-AQI run is a lower bound, not a confirmed pollution onset across missing hours",
                         "GEFS archive timestamps do not authenticate first public availability",
                         "Known historical years are not a blind test",
                         "Shared-truth error bounds are not statistical confidence intervals",
                         "Native replay and provenance checks are not independent retraining",
                         "Canonical, weather and scoring mathematics reuse frozen independent verifiers"]}
    old.write_exclusive_json(run / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing completed offline state run")
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
