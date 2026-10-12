"""Synthetic checks for the factor verifier; no research model is executed."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


class FactorVerifierTests(unittest.TestCase):
    def module(self):
        path = HERE / "verify_factors.py"
        self.assertTrue(path.is_file(), "factor verifier is not implemented")
        spec = importlib.util.spec_from_file_location("independent_factor_verifier", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def evaluation(self):
        times = pd.DatetimeIndex(["2024-01-02", "2024-01-02 01:00", "2025-01-02", "2025-01-02 01:00"], tz="Asia/Shanghai")
        return pd.DataFrame({"timestamp": times, "year": [2024, 2024, 2025, 2025],
            "continuity_block": [1, 1, 2, 2], "event_key": ["a", "a", "b", "b"],
            "lower_bound_hours": [30., 2., 30., 4.], "upper_bound_hours": [30., 2., 30., 8.],
            "exact_hours": [30., 2., 30., np.nan]})

    def predictions(self, module):
        return {arm: np.array([40., 4., 40., 10.]) for arm in module.ARMS}

    def test_reuses_exact_frozen_verifier_without_importing_runner(self):
        module = self.module()
        old = module.bridge()
        self.assertEqual(old.sha256(HERE.parent / "web_recovery_bridge_20261002/verify_bridge.py"), module.BRIDGE_SHA256)
        tree = ast.parse((HERE / "verify_factors.py").read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("factor_runner" in name or "bridge_runner" in name for name in imports))
        self.assertEqual(Path(old.__file__).name, "verify_bridge.py")

    def test_grid_population_preserves_all_origins_and_unknown_features(self):
        module = self.module()
        legacy = self.evaluation().iloc[:2].copy()
        legacy["population_row"] = [0, 1]
        legacy["aqi"] = [901., 902.]
        grid = pd.DataFrame({"timestamp": legacy.timestamp, "aqi": [130., np.nan]})
        rebuilt = module.grid_population(legacy, grid, ["aqi"])
        self.assertEqual(len(rebuilt), 2)
        self.assertEqual(rebuilt.population_row.tolist(), [0, 1])
        self.assertEqual(rebuilt.loc[0, "aqi"], 130.)
        self.assertTrue(np.isnan(rebuilt.loc[1, "aqi"]))
        pd.testing.assert_frame_equal(rebuilt.drop(columns="aqi"), legacy.drop(columns="aqi"))

    def test_grid_population_fails_for_timestamp_outside_grid(self):
        module = self.module()
        legacy = self.evaluation().iloc[:2].copy()
        legacy["aqi"] = [1., 2.]
        with self.assertRaises(ValueError):
            module.grid_population(legacy, pd.DataFrame({"timestamp": legacy.timestamp.iloc[:1], "aqi": [130.]}), ["aqi"])

    def test_inventory_requires_two_arms_for_every_block(self):
        module = self.module()
        records = [{"arm": arm, "continuity_block": block, "model_id": f"{arm}_b{block}"}
                   for block in (1, 2) for arm in module.NEW_ARMS]
        module.validate_inventory(records, self.evaluation())
        with self.assertRaises(ValueError):
            module.validate_inventory(records[:-1], self.evaluation())
        with self.assertRaises(ValueError):
            module.validate_inventory(records + records[:1], self.evaluation())
        changed = [dict(item) for item in records]
        changed[0]["arm"] = "interval_candidate"
        with self.assertRaises(ValueError):
            module.validate_inventory(changed, self.evaluation())

    def test_same_members_compares_ids_timestamps_and_bounds_not_only_counts(self):
        module = self.module()
        times = pd.date_range("2023-01-01", periods=2, freq="h", tz="Asia/Shanghai")
        rows = []
        for arm in module.NEW_ARMS:
            for index, time in enumerate(times):
                rows.append({"model_id": f"{arm}_b7", "arm": arm, "evaluation_block": 7,
                             "population_row": index, "timestamp": time, "lower_bound_hours": float(index + 1)})
        members = pd.DataFrame(rows)
        module.validate_same_members(members)
        changed = members.copy()
        changed.loc[2, "lower_bound_hours"] = 20.
        with self.assertRaises(ValueError):
            module.validate_same_members(changed)
        changed = members.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            module.validate_same_members(changed)

    def test_five_arms_seven_pairs_and_fixed_primary_gate(self):
        module = self.module()
        frame = self.evaluation()
        predictions = self.predictions(module)
        predictions["labels_only"] = np.array([30., 2., 30., 6.])
        scores, process = module.compute_scores(frame, predictions)
        self.assertEqual(set(scores["pooled"]["arms"]), set(module.ARMS))
        self.assertEqual(set(scores["pooled"]["paired"]), set(module.PAIRS))
        self.assertEqual(len(process), 10)
        self.assertFalse(scores["gates"]["passed"], "diagnostic arm must not select the winning gate")
        predictions["grid_all_legacy_origins"] = predictions["labels_only"].copy()
        scores, _ = module.compute_scores(frame, predictions)
        self.assertTrue(scores["gates"]["passed"])

    def test_scores_retain_interval_and_right_censored_rows(self):
        module = self.module()
        frame = self.evaluation()
        frame.loc[3, "upper_bound_hours"] = np.inf
        scores, _ = module.compute_scores(frame, self.predictions(module))
        for score in scores["pooled"]["arms"].values():
            self.assertEqual(score["rows"], 4)
            self.assertEqual(score["right_censored_rows"], 1)
            self.assertIsNone(score["all_row_error_upper"])
        self.assertEqual(scores["pooled"]["paired"]["grid_all_legacy_origins_vs_archive"]["rows"], 4)

    def test_row_errors_and_paired_bounds_follow_all_five_arms(self):
        module = self.module()
        frame = self.evaluation()
        values = self.predictions(module)
        values["grid_all_legacy_origins"] = np.array([30., 2., 30., 6.])
        result = module.prediction_rows(frame, values)
        self.assertEqual(len(result), len(frame))
        self.assertEqual(result.loc[3, "grid_all_legacy_origins_error_lower"], 0.)
        self.assertEqual(result.loc[3, "grid_all_legacy_origins_error_upper"], 2.)
        self.assertEqual(result.loc[0, "grid_all_legacy_origins_vs_archive_delta_upper"], -10.)
        for pair in module.PAIRS:
            self.assertIn(pair + "_delta_lower", result)
            self.assertIn(pair + "_delta_upper", result)

    def test_deployment_scope_remains_false_even_when_gate_passes(self):
        module = self.module()
        scope = {"primary_candidate": "grid_all_legacy_origins", "diagnostic_only_arms": ["labels_only"],
                 "production_ready": False, "deployed": False, "blind_test": False,
                 "historical_shadow_evaluation": True, "archive_is_primary_reference": True}
        module.validate_scope(scope)
        for key in ("production_ready", "deployed", "blind_test"):
            changed = dict(scope)
            changed[key] = True
            with self.assertRaises(ValueError):
                module.validate_scope(changed)
        changed = dict(scope)
        changed["primary_candidate"] = "labels_only"
        with self.assertRaises(ValueError):
            module.validate_scope(changed)

    def test_output_inventory_rejects_extra_and_escaped_artifact(self):
        module = self.module()
        old = module.bridge()
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            old.write_exclusive_json(run / "scores.json", {"x": 1})
            hashes = {"scores.json": old.sha256(run / "scores.json")}
            module.validate_output_inventory(run, hashes, required={"scores.json"})
            old.write_exclusive_json(run / "extra.json", {"x": 1})
            with self.assertRaises(ValueError):
                module.validate_output_inventory(run, hashes, required={"scores.json"})
            with self.assertRaises(ValueError):
                module.validate_output_inventory(run, {"../outside.json": "0" * 64}, required=set())

    def test_exclusive_verification_write_and_run_boundary(self):
        module = self.module()
        old = module.bridge()
        with tempfile.TemporaryDirectory() as folder:
            experiment = Path(folder)
            run = experiment / "runs/example"
            run.mkdir(parents=True)
            self.assertEqual(old.validate_run_directory(run, experiment), run.resolve())
            with self.assertRaises(ValueError):
                old.validate_run_directory(experiment, experiment)
            old.write_exclusive_json(run / "verification.json", {"status": "passed"})
            with self.assertRaises(FileExistsError):
                old.write_exclusive_json(run / "verification.json", {"status": "failed"})

    def test_hash_conflict_is_never_silently_overridden(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module.bridge().combine_locks([{"x": "a" * 64}, {"x": "b" * 64}])

    def test_parent_passed_evidence_requires_exact_hash_and_report_binding(self):
        module = self.module()
        verification = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
                        "report_sha256": module.PARENT_REPORT_SHA256, "verifier_sha256": module.BRIDGE_SHA256,
                        "fixed_origins_verified": 701, "models_verified": 64, "refitted_predictions_verified": 1402}
        module.validate_parent_evidence(verification, module.PARENT_REPORT_SHA256, module.PARENT_VERIFICATION_SHA256)
        for key, value in (("status", "failed"), ("report_sha256", "0" * 64), ("models_verified", 63)):
            changed = dict(verification)
            changed[key] = value
            with self.assertRaises(ValueError):
                module.validate_parent_evidence(changed, module.PARENT_REPORT_SHA256, module.PARENT_VERIFICATION_SHA256)
        with self.assertRaises(ValueError):
            module.validate_parent_evidence(verification, module.PARENT_REPORT_SHA256, "0" * 64)

    def test_member_reconstruction_keeps_old_metadata_but_uses_canonical_maturity(self):
        module = self.module()
        times = pd.date_range("2023-01-01", periods=5, freq="h", tz="Asia/Shanghai")
        population = pd.DataFrame({"timestamp": times, "population_row": np.arange(5),
            "continuity_block": [1, 1, 7, 1, 1], "label_available_at": times.shift(1, freq="h"),
            "lower_bound_hours": [2.] * 5, "upper_bound_hours": [2.] * 5, "exact_hours": [2.] * 5,
            "informative": [True] * 5, "reason": ["exact"] * 5, "legacy_remaining_hours": [99.] * 5})
        record = {"arm": "labels_only", "model_id": "labels_only_b7", "continuity_block": 7, "cutoff": times[3]}
        members = module.training_members(population, record)
        self.assertEqual(members.population_row.tolist(), [0, 1])
        self.assertEqual(members.legacy_remaining_hours.tolist(), [99., 99.])
        self.assertEqual(members.exact_hours.tolist(), [2., 2.])
        self.assertEqual(set(members.evaluation_block), {7})

    def test_parent_inventory_detects_added_missing_and_path_drift(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            module.bridge().write_exclusive_json(parent / "report.json", {"x": 1})
            names, hashes = module.artifact_locks(parent)
            self.assertEqual(names, ["report.json"])
            self.assertEqual(set(hashes), {str((parent / "report.json").resolve())})
            module.validate_artifact_locks(parent, names, hashes)
            module.bridge().write_exclusive_json(parent / "unexpected.json", {"x": 2})
            with self.assertRaises(ValueError):
                module.validate_artifact_locks(parent, names, hashes)

    def test_static_source_has_only_load_predict_not_fit_or_train(self):
        self.module()
        tree = ast.parse((HERE / "verify_factors.py").read_text(encoding="utf-8"))
        forbidden = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr in ("fit", "fit_bounds", "train")]
        self.assertEqual(forbidden, [])
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "verify_run" for node in ast.walk(tree)))

    def test_new_members_match_parent_control_full_metadata(self):
        module = self.module()
        old = pd.DataFrame({"model_id": ["legacy_control_b7"], "arm": ["legacy_control"], "evaluation_block": [7],
                            "population_row": [10], "legacy_remaining_hours": [99.], "exact_hours": [2.]})
        new = pd.concat([old.assign(model_id=f"{arm}_b7", arm=arm) for arm in module.NEW_ARMS], ignore_index=True)
        module.validate_parent_members(new, old)
        changed = new.copy()
        changed.loc[0, "legacy_remaining_hours"] = 2.
        with self.assertRaises(ValueError):
            module.validate_parent_members(changed, old)

    def test_scope_booleans_are_not_numeric_zero_and_grid_lookup_not_duplicate(self):
        module = self.module()
        scope = {"primary_candidate": "grid_all_legacy_origins", "diagnostic_only_arms": ["labels_only"],
                 "production_ready": 0, "deployed": False, "blind_test": False,
                 "historical_shadow_evaluation": True, "archive_is_primary_reference": True}
        with self.assertRaises(ValueError):
            module.validate_scope(scope)
        legacy = self.evaluation().iloc[:2].copy()
        legacy["timestamp"] = legacy.timestamp.iloc[0]
        legacy["aqi"] = 5.
        grid = pd.DataFrame({"timestamp": [legacy.timestamp.iloc[0]], "aqi": [130.]})
        with self.assertRaises(ValueError):
            module.grid_population(legacy, grid, ["aqi"])

    def test_artifact_inventory_uses_relative_string_order_and_rejects_nested_report(self):
        module = self.module()
        old = module.bridge()
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "models").mkdir()
            old.write_exclusive_json(run / "models/z.json", {})
            old.write_exclusive_json(run / "models.json", {})
            names, hashes = module.artifact_locks(run)
            self.assertEqual(names, ["models.json", "models/z.json"])
            outputs = {name: hashes[str((run / name).resolve())] for name in names}
            old.write_exclusive_json(run / "models/report.json", {})
            with self.assertRaises(ValueError):
                module.validate_output_inventory(run, outputs, required=set(outputs))


if __name__ == "__main__":
    unittest.main()
