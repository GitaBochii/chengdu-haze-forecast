"""Synthetic independent calendar-verifier tests; no real fit or prediction."""

import ast
import copy
import hashlib
import importlib.util
import json
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


class CalendarVerifierTests(unittest.TestCase):
    def module(self):
        path = HERE / "verify_calendar.py"
        self.assertTrue(path.is_file(), "calendar verifier not yet implemented")
        spec = importlib.util.spec_from_file_location("independent_calendar_verifier", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def grid(self, aqi):
        times = pd.date_range("2024-01-01", periods=len(aqi), freq="h", tz="Asia/Shanghai")
        return pd.DataFrame({"timestamp": times, "aqi": aqi, "x": np.nan,
                             "continuity_block": 1, "eligible": False,
                             "pollution_state": "unknown", "input_hour_observed": True})

    def score_frame(self):
        times = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="Asia/Shanghai"),
                                 pd.Timestamp("2024-02-01", tz="Asia/Shanghai"),
                                 pd.Timestamp("2025-01-01", tz="Asia/Shanghai"),
                                 pd.Timestamp("2025-02-01", tz="Asia/Shanghai")])
        return pd.DataFrame({"timestamp": times, "year": [2024, 2024, 2025, 2025],
                             "continuity_block": [1, 2, 3, 4], "event_key": ["a", "b", "c", "d"],
                             "lower_bound_hours": [30., 40., 30., 40.], "upper_bound_hours": [30., 40., 30., 40.],
                             "exact_hours": [30., 40., 30., 40.]})

    def test_population_keeps_every_clean_polluted_missing_and_ineligible_hour(self):
        module = self.module()
        grid = self.grid([np.nan, 150., *([50.] * 14)])
        population = module.calendar_population(grid, ["aqi", "x"])
        self.assertEqual(len(population), len(grid))
        self.assertEqual(population.population_row.tolist(), list(range(len(grid))))
        self.assertEqual(population.aqi_category.iloc[:3].tolist(), ["missing", "polluted", "clean"])
        self.assertTrue(population.x.isna().all())
        self.assertFalse(population.eligible.any())
        self.assertTrue(np.isposinf(population.upper_bound_hours).any())
        self.assertEqual(set(population.label_type), {"exact", "right_censored"})

    def test_clean_origin_target_is_next_window_start_one_not_zero_or_confirmation(self):
        module = self.module()
        population = module.calendar_population(self.grid([50.] * 20), ["aqi", "x"])
        self.assertEqual(population.exact_hours.iloc[0], 1.)
        self.assertEqual(population.label_available_at.iloc[0], population.timestamp.iloc[6])
        self.assertEqual(population.upper_bound_hours.iloc[-1], np.inf)
        self.assertEqual(population.lower_bound_hours.iloc[-1], 1.)

    def test_finite_interval_is_retained_not_midpoint_or_exact_only(self):
        module = self.module()
        population = module.calendar_population(self.grid([150., np.nan, *([50.] * 12)]), ["aqi", "x"])
        self.assertEqual(population.lower_bound_hours.iloc[0], 1.)
        self.assertEqual(population.upper_bound_hours.iloc[0], 2.)
        self.assertTrue(np.isnan(population.exact_hours.iloc[0]))
        self.assertEqual(population.label_type.iloc[0], "interval")

    def test_strict_maturity_purge_keeps_clean_missing_and_right_censor_members(self):
        module = self.module()
        population = module.calendar_population(self.grid([50.] * 15), ["aqi", "x"])
        cut = population.timestamp.iloc[10]
        population["label_available_at"] = cut - pd.Timedelta(hours=1)
        population.loc[1, "label_available_at"] = cut
        population.loc[2, "continuity_block"] = 99
        population.loc[3, "aqi_category"] = "missing"
        population.loc[4, ["upper_bound_hours", "exact_hours", "label_type", "reason"]] = [np.inf, np.nan, "right_censored", "right_censored"]
        members = module.training_members(population, {"model_id": "calendar_all_hours_b99", "arm": module.PRIMARY,
                                                        "continuity_block": 99, "cutoff": cut})
        self.assertEqual(members.population_row.tolist(), [0, 3, 4, 5, 6, 7, 8, 9])
        self.assertIn("right_censored", set(members.label_type))
        self.assertIn("missing", set(members.aqi_category))
        self.assertFalse(members.eligible.any())
        self.assertNotIn("x", members)

    def test_population_clock_must_be_complete_unique_and_ordered(self):
        module = self.module()
        grid = self.grid([50.] * 10)
        with self.assertRaises(ValueError):
            module.calendar_population(grid.drop(index=4), ["aqi", "x"])
        with self.assertRaises(ValueError):
            module.calendar_population(grid.iloc[::-1], ["aqi", "x"])

    def test_classification_counts_include_all_zero_classes(self):
        module = self.module()
        population = module.calendar_population(self.grid([50.] * 20), ["aqi", "x"])
        self.assertEqual(module.category_counts(population), {"clean": 20, "polluted": 0, "missing": 0})
        self.assertEqual(sum(module.label_counts(population).values()), len(population))
        changed = population.copy()
        changed.loc[0, "aqi_category"] = "foreign"
        with self.assertRaises(ValueError):
            module.category_counts(changed)

    def test_gate_remains_calendar_vs_archive_even_other_ref_is_better(self):
        module = self.module()
        frame = self.score_frame()
        truth = frame.exact_hours.to_numpy()
        predictions = {"archive": truth + 10., "interval_candidate": truth,
                       "grid_all_legacy_origins": truth, module.PRIMARY: truth + 9.}
        scores, processes = module.compute_scores(frame, predictions)
        self.assertFalse(scores["gates"]["passed"])
        self.assertIn("2024:exact_row_mae_20pct", scores["gates"]["failure_reasons"])
        self.assertEqual(set(scores["pooled"]["arms"]), set(module.ARMS))
        self.assertEqual(len(processes), len(frame) * 4)
        predictions[module.PRIMARY] = truth + 8.
        scores, _ = module.compute_scores(frame, predictions)
        self.assertTrue(scores["gates"]["passed"])

    def test_interval_and_right_censor_rows_are_retained_in_paired_bounds(self):
        module = self.module()
        frame = self.score_frame()
        frame.loc[1, ["upper_bound_hours", "exact_hours"]] = [50., np.nan]
        frame.loc[3, ["upper_bound_hours", "exact_hours"]] = [np.inf, np.nan]
        predictions = {arm: np.array([32., 42., 32., 42.]) for arm in module.ARMS}
        rows = module.prediction_rows(frame, predictions)
        scores, _ = module.compute_scores(frame, predictions)
        self.assertEqual(len(rows), 4)
        self.assertEqual(scores["pooled"]["arms"][module.PRIMARY]["finite_interval_rows"], 1)
        self.assertEqual(scores["pooled"]["arms"][module.PRIMARY]["right_censored_rows"], 1)
        self.assertEqual(scores["pooled"]["paired"][module.PRIMARY + "_vs_archive"]["all_row_delta_upper"], 0.)

    def test_model_inventory_requires_one_new_arm_per_block_and_unique_identity(self):
        module = self.module()
        evaluation = self.score_frame()
        records = [{"model_id": f"{module.PRIMARY}_b{block}", "arm": module.PRIMARY,
                    "continuity_block": block} for block in (1, 2, 3, 4)]
        module.validate_inventory(records, evaluation)
        with self.assertRaises(ValueError):
            module.validate_inventory(records[:-1], evaluation)
        with self.assertRaises(ValueError):
            module.validate_inventory(records + [records[0]], evaluation)
        changed = copy.deepcopy(records)
        changed[0]["arm"] = "grid_all_legacy_origins"
        with self.assertRaises(ValueError):
            module.validate_inventory(changed, evaluation)

    def test_hash_mutation_extra_file_exclusive_output_and_path_escape(self):
        module = self.module()
        old = module.bridge()
        with tempfile.TemporaryDirectory() as folder:
            experiment = Path(folder)
            run = experiment / "runs/a"
            run.mkdir(parents=True)
            old.write_exclusive_json(run / "report.json", {})
            old.write_exclusive_json(run / "outputs.json", {})
            locks = {"outputs.json": old.sha256(run / "outputs.json")}
            module.validate_output_inventory(run, locks, {"outputs.json"})
            inventory, hashes = module.artifact_locks(run)
            (run / "outputs.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                module.validate_artifact_locks(run, inventory, hashes)
            with self.assertRaises(FileExistsError):
                old.write_exclusive_json(run / "report.json", {})
            with self.assertRaises(ValueError):
                old.validate_run_directory(experiment, experiment)
            with self.assertRaises(ValueError):
                old.run_file(run, "../escape.json")

    def test_no_new_or_old_main_import_or_fitting(self):
        module = self.module()
        self.assertEqual(module.PRIMARY, "calendar_all_hours")
        self.assertEqual(len(module.ARMS), 4)
        tree = ast.parse((HERE / "verify_calendar.py").read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(name.endswith(("calendar_runner", "factor_runner", "bridge_runner")) for name in imports))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(set(calls) & {"fit", "fit_bounds", "train"})

    def test_all_shared_old_population_features_bounds_and_maturity_are_unchanged(self):
        module = self.module()
        population = module.calendar_population(self.grid([50.] * 20), ["aqi", "x"])
        old = population.iloc[[1, 4, 7]].copy()
        old["population_row"] = [0, 1, 2]
        module.validate_common_population(population, old, ["aqi", "x"])
        for column, value in (("aqi", 51.), ("lower_bound_hours", 2.),
                              ("label_available_at", old.label_available_at.iloc[0] + pd.Timedelta(hours=1))):
            changed = old.copy()
            changed.loc[changed.index[0], column] = value
            with self.assertRaises(ValueError):
                module.validate_common_population(population, changed, ["aqi", "x"])

    def test_exact_one_summary_uses_cutoff_members_not_full_archive_fraction(self):
        module = self.module()
        population = module.calendar_population(self.grid([50.] * 20), ["aqi", "x"])
        population.loc[0, ["lower_bound_hours", "upper_bound_hours", "exact_hours"]] = [9., 9., 9.]
        self.assertEqual(module.exact_one_stats(population.iloc[:2]), {"exact_t1_rows": 1, "exact_t1_fraction": .5})
        self.assertNotEqual(module.exact_one_stats(population)["exact_t1_fraction"], .5)

    def trained_config(self):
        return {"learner": {"generic_param": {"seed": "42", "nthread": "1"},
                "learner_model_param": {"num_feature": "87"},
                "gradient_booster": {"gbtree_model_param": {"num_trees": "500"},
                    "gbtree_train_param": {"tree_method": "hist"},
                    "tree_train_param": {"eta": ".05", "max_depth": "4", "min_child_weight": "8",
                                         "subsample": ".9", "colsample_bytree": ".8", "lambda": "3"}},
                "objective": {"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution": "normal",
                                                                      "aft_loss_distribution_scale": "1.2"}}}}

    def test_true_fitted_configuration_order_and_round_count_are_checked(self):
        module = self.module()
        names = ["aqi", *[f"x{index}" for index in range(86)]]
        record = {"feature_names": names, "booster_feature_names": names, "num_boosted_rounds": 500,
                  "configuration": module.CONFIGURATION, "aft_parameters": module.bridge()._fixed_parameters(),
                  "actual_trained_booster_config": self.trained_config()}
        module.validate_model_parameters(record, names)
        for key, value in (("num_boosted_rounds", 499), ("booster_feature_names", names[::-1])):
            with self.assertRaises(ValueError):
                module.validate_model_parameters({**record, key: value}, names)
        changed = copy.deepcopy(record)
        changed["actual_trained_booster_config"]["learner"]["gradient_booster"]["tree_train_param"]["lambda"] = "1"
        with self.assertRaises(ValueError):
            module.validate_model_parameters(changed, names)
        changed = copy.deepcopy(record)
        changed["actual_trained_booster_config"]["learner"]["generic_param"]["seed"] = "0"
        with self.assertRaises(ValueError):
            module.validate_model_parameters(changed, names)

    def test_exact_passed_parent_chain_and_descriptive_scope_guards(self):
        module = self.module()
        evidence = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
                    "report_sha256": module.PARENT_REPORT_SHA256, "verifier_sha256": module.PARENT_VERIFIER_SHA256,
                    "reused_independent_verifier_sha256": module.BRIDGE_SHA256,
                    "fixed_origins_verified": 701, "models_verified": 64, "refitted_predictions_verified": 1402,
                    "primary_candidate": "grid_all_legacy_origins", "production_ready": False, "deployed": False}
        module.validate_parent_evidence(evidence, module.PARENT_REPORT_SHA256, module.PARENT_VERIFICATION_SHA256)
        with self.assertRaises(ValueError):
            module.validate_parent_evidence({**evidence, "status": "failed"}, module.PARENT_REPORT_SHA256, module.PARENT_VERIFICATION_SHA256)
        with self.assertRaises(ValueError):
            module.validate_parent_evidence(evidence, "changed", module.PARENT_VERIFICATION_SHA256)

    def test_protected_directory_exact_inventory_missing_or_added_file_fails(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protected = root / "protected"
            protected.mkdir()
            old = module.bridge()
            old.write_exclusive_json(protected / "old.json", {})
            locks = {str((protected / "old.json").resolve()): old.sha256(protected / "old.json")}
            module.validate_protected_inventory(locks, root, ("protected",))
            old.write_exclusive_json(protected / "new.json", {})
            with self.assertRaises(ValueError):
                module.validate_protected_inventory(locks, root, ("protected",))
            with self.assertRaises(ValueError):
                module.validate_protected_inventory(locks, root, ("missing",))

    def test_nested_report_is_not_ignored_and_manifest_conflicts_fail(self):
        module = self.module()
        old = module.bridge()
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "nested").mkdir()
            old.write_exclusive_json(run / "report.json", {})
            old.write_exclusive_json(run / "outputs.json", {})
            old.write_exclusive_json(run / "nested/report.json", {})
            with self.assertRaises(ValueError):
                module.validate_output_inventory(run, {"outputs.json": old.sha256(run / "outputs.json")}, {"outputs.json"})
            with self.assertRaises(ValueError):
                old.combine_locks([{"file": "aaa"}, {"file": "bbb"}])

    def test_member_statistics_include_missing_observation_qualification_and_shared_rows(self):
        module = self.module()
        population = module.calendar_population(self.grid([np.nan, 150., *([50.] * 14)]), ["aqi", "x"])
        population.loc[0, "input_hour_observed"] = False
        population.loc[1, "eligible"] = True
        stats = module.member_statistics(population.iloc[:4], ["aqi", "x"], population.iloc[[0, 2]])
        self.assertEqual(stats["input_hour_observed_counts"], {"observed": 3, "unobserved": 1})
        self.assertEqual(stats["causal_eligible_counts"], {"eligible": 1, "ineligible": 3})
        self.assertEqual(stats["shared_legacy_member_rows"], 2)
        self.assertEqual(stats["additional_calendar_member_rows"], 2)
        self.assertEqual(stats["train_feature_missingness"], {"aqi": .25, "x": 1.})

    def test_common_mature_members_per_block_are_exactly_parent_grid_members(self):
        module = self.module()
        population = module.calendar_population(self.grid([50.] * 20), ["aqi", "x"])
        cutoff = population.timestamp.iloc[12]
        old_population = population.iloc[[0, 2, 4, 6]].copy()
        record = {"model_id": "calendar_all_hours_b99", "arm": module.PRIMARY,
                  "continuity_block": 99, "cutoff": cutoff}
        new_members = module.training_members(population, record)
        old_members = new_members.loc[new_members.timestamp.isin(old_population.timestamp)].copy()
        old_members["arm"] = "grid_all_legacy_origins"
        module.validate_common_members(new_members, old_population, old_members, record)
        with self.assertRaises(ValueError):
            module.validate_common_members(new_members, old_population, old_members.iloc[:-1], record)

    def test_native_reload_checks_all_rows_and_does_not_use_loaded_default_training_seed(self):
        module = self.module()
        old = module.bridge()
        names = ["aqi", *[f"x{index}" for index in range(86)]]
        loaded = copy.deepcopy(self.trained_config())
        loaded["learner"]["generic_param"]["seed"] = "0"
        record = {"model_id": module.PRIMARY + "_b1", "arm": module.PRIMARY, "continuity_block": 1,
                  "feature_names": names, "booster_feature_names": names, "num_boosted_rounds": 500,
                  "configuration": module.CONFIGURATION, "aft_parameters": old._fixed_parameters(),
                  "actual_trained_booster_config": self.trained_config(), "model_path": "models/a.json"}
        calls = []

        class Booster:
            def __init__(self, params):
                self.feature_names = names
                calls.append(params)

            def load_model(self, path):
                calls.append(str(path))

            def num_boosted_rounds(self):
                return 500

            def num_features(self):
                return 87

            def save_config(self):
                return json.dumps(loaded)

            def predict(self, matrix):
                calls.append(matrix.shape)
                return np.array([2., 3.])

        def dmatrix(frame, missing):
            self.assertTrue(np.isnan(missing))
            self.assertEqual(frame.columns.tolist(), names)
            self.assertTrue(frame.iloc[1].isna().all())
            return frame

        fake = types.SimpleNamespace(Booster=Booster, DMatrix=dmatrix)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "models").mkdir()
            tree = {"left_children": [-1], "right_children": [-1], "split_indices": [0], "split_conditions": [0.]}
            old.write_exclusive_json(run / record["model_path"], {"learner": {"gradient_booster": {"model": {"trees": [tree] * 500}}}})
            digest = old.sha256(run / record["model_path"])
            record["model_sha256"] = digest
            query = np.array([np.zeros(87), np.full(87, np.nan)])
            with patch.dict("sys.modules", {"xgboost": fake}):
                prediction = module.reload_predictions(run, record, query, names, {record["model_path"]: digest})
            np.testing.assert_array_equal(prediction, [2., 3.])
            self.assertIn((2, 87), calls)
            with self.assertRaises(ValueError):
                module.reload_predictions(run, record, query, names, {record["model_path"]: "changed"})

    def test_scope_never_turns_gate_pass_into_ready_or_weight_change(self):
        module = self.module()
        report = {"schema_version": 1, "status": "pending_independent_verification", "years": [2024, 2025],
                  "configuration": module.CONFIGURATION, "aft_parameters": module.bridge()._fixed_parameters(),
                  "primary_candidate": module.PRIMARY, "new_refitted_arms": [module.PRIMARY],
                  "inherited_arms": list(module.INHERITED_ARMS), "arms": list(module.ARMS), "scope": module.SCOPE,
                  "weight_policy": module.WEIGHT_POLICY, "historical_shadow_evaluation": True,
                  "archive_is_primary_reference": True, "blind_test": False, "deployed": False, "production_ready": False,
                  "protected_unchanged": True, "parent_unchanged": True, "shared_legacy_members_unchanged": True,
                  "inherited_predictions_unchanged": True}
        module.validate_scope(report)
        for key, value in (("production_ready", True), ("primary_candidate", "interval_candidate"),
                           ("weight_policy", "downweight clean")):
            with self.assertRaises(ValueError):
                module.validate_scope({**report, key: value})

    def test_all_three_inherited_predictions_are_checked_not_only_archive(self):
        module = self.module()
        parent = self.score_frame()
        for index, arm in enumerate(module.INHERITED_ARMS):
            parent[arm] = index + np.arange(len(parent), dtype=float)
        module.validate_inherited_predictions(parent, parent.copy())
        for arm in module.INHERITED_ARMS:
            changed = parent.copy()
            changed.loc[3, arm] += 1.
            with self.assertRaises(ValueError):
                module.validate_inherited_predictions(parent, changed)

    def test_new_code_lock_requires_protocol_main_test_verifier_and_verifier_test(self):
        module = self.module()
        root = Path("J:/synthetic_root")
        with self.assertRaises(ValueError):
            module.validate_code_locks({}, root)
        names = ("PROTOCOL.md", "calendar_runner.py", "test_calendar_runner.py", "verify_calendar.py", "test_verify_calendar.py")
        locks = {str((root / module.EXPERIMENT / name).resolve()): "digest" for name in names}
        module.validate_code_locks(locks, root)
        del locks[str((root / module.EXPERIMENT / "PROTOCOL.md").resolve())]
        with self.assertRaises(ValueError):
            module.validate_code_locks(locks, root)

    def test_every_one_of_32_models_members_and_final_prediction_is_checked_with_mock_reload(self):
        module = self.module()
        old = module.bridge()
        names = ["aqi", *[f"x{index}" for index in range(86)]]
        grid = self.grid([50.] * 80)
        for name in names[1:]:
            grid[name] = np.nan
        grid["continuity_block"] = [0] * 48 + list(range(1, 33))
        population = module.calendar_population(grid, names)
        parent_population = population.iloc[range(0, 40, 2)].copy()
        evaluation = population.iloc[48:].reset_index(drop=True)
        all_members, parent_members, records = [], [], []
        summary_keys = ("model_id", "train_rows", "label_counts", "aqi_category_counts", "input_hour_observed_counts", "causal_eligible_counts",
                        "exact_t1_rows", "exact_t1_fraction", "shared_legacy_member_rows", "additional_calendar_member_rows",
                        "training_last_timestamp", "max_label_available_at")
        for block in range(1, 33):
            cut = grid.loc[grid.continuity_block.eq(block), "timestamp"].min()
            record = {"model_id": f"{module.PRIMARY}_b{block}", "arm": module.PRIMARY, "continuity_block": block,
                      "cutoff": cut.isoformat(), "eval_rows": 1, "training_member_path": "training_members.parquet",
                      "model_path": f"models/{module.PRIMARY}_b{block}.json", "reload_predictions_equal": True,
                      "weight_policy": module.WEIGHT_POLICY, "right_censor_policy": module.RIGHT_CENSOR_POLICY}
            wanted = module.training_members(population, record)
            expected = population.iloc[wanted.population_row.to_numpy(dtype=int)]
            record.update(module.member_statistics(expected, names, parent_population))
            record.update({"train_rows": len(wanted), "training_last_timestamp": expected.timestamp.max().isoformat(),
                           "max_label_available_at": expected.label_available_at.max().isoformat(),
                           "training_features_sha256": old.matrix_digest(expected[names].to_numpy(dtype=float)),
                           "training_timestamps_sha256": hashlib.sha256(pd.DatetimeIndex(expected.timestamp).asi8.astype("<i8").tobytes()).hexdigest()})
            all_members.append(wanted)
            shared = wanted.loc[wanted.timestamp.isin(parent_population.timestamp)].copy()
            shared["arm"] = "grid_all_legacy_origins"
            parent_members.append(shared)
            records.append(record)
        members, parent_members = pd.concat(all_members, ignore_index=True), pd.concat(parent_members, ignore_index=True)
        models = {"schema_version": 1, "feature_names": names, "models": records}
        outputs = {"evaluation.parquet", "calendar_population.parquet", "features.npz", "training_members.parquet", "models.json",
                   "predictions.parquet", "process_scores.parquet", "scores.json", "environment.json", "execution_record_at_start.md"}
        outputs.update(record["model_path"] for record in records)
        report = {"fitted_models": 32, "output_sha256": dict.fromkeys(outputs, "synthetic"),
                  "training_maturity_summary": [{key: record[key] for key in summary_keys} for record in records]}
        features = np.zeros((32, 87))
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            saved = pd.DataFrame({module.PRIMARY: np.full(32, 2.)})
            saved.to_parquet(run / "predictions.parquet", index=False)
            with patch.object(module, "reload_predictions", side_effect=lambda *args: np.full(len(args[2]), 2.)) as reload:
                result, checked, shared_checked = module.verify_models(run, report, models, members, evaluation, grid, population,
                                                                       parent_population, parent_members, features, names)
            self.assertEqual(reload.call_count, 32)
            self.assertEqual(checked, len(members))
            self.assertEqual(shared_checked, len(parent_members))
            np.testing.assert_array_equal(result, np.full(32, 2.))
            changed = copy.deepcopy(models)
            changed["models"][-1]["training_features_sha256"] = "changed"
            with patch.object(module, "reload_predictions", side_effect=lambda *args: np.full(len(args[2]), 2.)):
                with self.assertRaises(ValueError):
                    module.verify_models(run, report, changed, members, evaluation, grid, population,
                                         parent_population, parent_members, features, names)
            saved.loc[31, module.PRIMARY] = 3.
            saved.to_parquet(run / "predictions.parquet", index=False)
            with patch.object(module, "reload_predictions", side_effect=lambda *args: np.full(len(args[2]), 2.)):
                with self.assertRaises(ValueError):
                    module.verify_models(run, report, models, members, evaluation, grid, population,
                                         parent_population, parent_members, features, names)

    def test_parent_feature_archive_has_four_exact_keys_child_only_candidate(self):
        module = self.module()
        names = ["aqi", "x"]
        timestamps = np.array([1_700_000_000_000_000_000], dtype=np.int64)
        features = np.array([[50., np.nan]])

        class Archive(dict):
            @property
            def files(self):
                return list(self)

        parent = Archive(timestamps=timestamps, feature_names=np.asarray(names), labels_only=features.copy(),
                         grid_all_legacy_origins=features.copy())
        child = Archive(timestamps=timestamps, feature_names=np.asarray(names), candidate=features.copy())
        module.validate_feature_archive(parent, names, timestamps, features, parent=True)
        module.validate_feature_archive(child, names, timestamps, features)
        with self.assertRaises(ValueError):
            module.validate_feature_archive(child, names, timestamps, features, parent=True)
        changed = Archive(parent)
        changed["timestamps"] = timestamps + 1
        with self.assertRaises(ValueError):
            module.validate_feature_archive(changed, names, timestamps, features, parent=True)


if __name__ == "__main__":
    unittest.main()
