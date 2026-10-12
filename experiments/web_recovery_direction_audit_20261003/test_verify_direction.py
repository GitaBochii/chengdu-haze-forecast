"""Synthetic direction-verifier tests without fitting or model prediction."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


class DirectionVerifierTests(unittest.TestCase):
    def module(self):
        path = HERE / "verify_direction.py"
        self.assertTrue(path.is_file(), "direction verifier is not implemented")
        spec = importlib.util.spec_from_file_location("independent_direction_verifier", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def frame(self, lower, upper=None, blocks=None, predictions=None):
        lower = np.asarray(lower, dtype=float)
        upper = lower.copy() if upper is None else np.asarray(upper, dtype=float)
        times = pd.date_range("2024-01-01", periods=len(lower), freq="h", tz="Asia/Shanghai")
        frame = pd.DataFrame({"timestamp": times, "label_available_at": times + pd.Timedelta(hours=10),
                             "continuity_block": list(range(len(lower))) if blocks is None else blocks,
                             "lower_bound_hours": lower, "upper_bound_hours": upper,
                             "exact_hours": np.where(lower == upper, lower, np.nan),
                             "origin_index": np.arange(len(lower))})
        for arm in self.module().ARMS:
            frame[arm] = 1. if predictions is None else predictions
        return frame

    def test_bound_directions_keep_finite_interval_and_right_censor(self):
        module = self.module()
        frame = self.frame([5., 5., 2.], [8., 8., np.inf], predictions=[4., 6., 20.])
        rates = module.rates(frame, module.PRIMARY)
        self.assertEqual(rates["rows"], 3)
        self.assertEqual(rates["row_early_lower"], 1 / 3)
        self.assertEqual(rates["row_early_upper"], 1.)
        self.assertEqual(rates["row_late_lower"], 0.)
        self.assertEqual(rates["row_late_upper"], 2 / 3)

    def test_equality_is_neither_early_nor_late(self):
        module = self.module()
        rates = module.rates(self.frame([5.], predictions=[5.]), module.PRIMARY)
        self.assertEqual([rates[key] for key in ("row_early_lower", "row_early_upper", "row_late_lower", "row_late_upper")], [0.] * 4)

    def test_process_mean_is_not_row_weighted(self):
        module = self.module()
        frame = self.frame([10.] * 10, blocks=[1] * 9 + [2], predictions=[1.] * 9 + [20.])
        rates = module.rates(frame, module.PRIMARY)
        self.assertEqual(rates["row_early_lower"], .9)
        self.assertEqual(rates["block_early_lower"], .5)
        self.assertEqual(rates["early_majority_blocks"], 1)

    def test_minimum_six_blocks_and_exact_two_thirds_raise(self):
        module = self.module()
        frame = self.frame([10.] * 6, predictions=[1.] * 4 + [20.] * 2)
        self.assertEqual(module.signal(module.rates(frame, module.PRIMARY)), "raise")
        self.assertEqual(module.signal(module.rates(frame.iloc[:5], module.PRIMARY)), "insufficient")

    def test_lower_signal_and_strict_half_threshold(self):
        module = self.module()
        frame = self.frame([10.] * 6, predictions=[20.] * 4 + [1.] * 2)
        self.assertEqual(module.signal(module.rates(frame, module.PRIMARY)), "lower")
        frame[module.PRIMARY] = [20.] * 3 + [1.] * 3
        self.assertEqual(module.signal(module.rates(frame, module.PRIMARY)), "none")

    def test_two_thirds_guard_is_not_only_overall_process_mean(self):
        module = self.module()
        frame = self.frame([10.] * 6, predictions=[1.] * 3 + [20.] * 3)
        rates = module.rates(frame, module.PRIMARY)
        rates["block_early_lower"] = .9
        self.assertEqual(module.signal(rates), "none")

    def test_empty_rates_remain_explicit_insufficient(self):
        module = self.module()
        rates = module.rates(self.frame([]), module.PRIMARY)
        self.assertEqual(rates["rows"], 0)
        self.assertEqual(rates["blocks"], 0)
        self.assertIsNone(rates["block_early_lower"])
        self.assertEqual(module.signal(rates), "insufficient")

    def test_prefix_strict_cutoff_maturity_block_and_prediction_bucket(self):
        module = self.module()
        frame = self.frame([100.] * 6, predictions=[23., 24., 23., 23., 23., 23.])
        cut = frame.timestamp.iloc[4]
        frame["label_available_at"] = [cut - pd.Timedelta(hours=1), cut - pd.Timedelta(hours=1), cut,
                                       cut - pd.Timedelta(hours=1), cut - pd.Timedelta(hours=1), cut - pd.Timedelta(hours=1)]
        frame.loc[3, "continuity_block"] = 99
        low = module.prefix(frame, cut, 99, module.PRIMARY, "lt24")
        high = module.prefix(frame, cut, 99, module.PRIMARY, "ge24")
        self.assertEqual(low.origin_index.tolist(), [0])
        self.assertEqual(high.origin_index.tolist(), [1])

    def test_current_mutation_cannot_change_past_signal(self):
        module = self.module()
        frame = self.frame([10.] * 8)
        cutoff = frame.timestamp.iloc[7]
        frame["label_available_at"] = frame.timestamp
        before = module.rates(module.prefix(frame, cutoff, 7, module.PRIMARY, "lt24"), module.PRIMARY)
        changed = frame.copy()
        changed.loc[7, ["lower_bound_hours", "upper_bound_hours", "exact_hours", module.PRIMARY]] = [100., np.inf, np.nan, 1000.]
        after = module.rates(module.prefix(changed, cutoff, 7, module.PRIMARY, "lt24"), module.PRIMARY)
        self.assertEqual(before, after)

    def test_current_unknown_empty_and_opposite_remain_distinct(self):
        module = self.module()
        self.assertEqual(module.current_direction(module.rates(self.frame([]), module.PRIMARY)), "current_empty")
        self.assertEqual(module.current_direction(module.rates(self.frame([1.], [50.], predictions=[20.]), module.PRIMARY)), "unknown")
        self.assertEqual(module.agreement("raise", "raise"), "consistent")
        self.assertEqual(module.agreement("raise", "lower"), "opposite")
        self.assertEqual(module.agreement("insufficient", "lower"), "insufficient_history")

    def test_exclusive_json_and_run_traversal(self):
        module = self.module()
        helper = module.bridge()
        with tempfile.TemporaryDirectory() as folder:
            experiment = Path(folder)
            run = experiment / "runs/a"
            run.mkdir(parents=True)
            helper.write_exclusive_json(run / "verification.json", {"status": "passed"})
            with self.assertRaises(FileExistsError):
                helper.write_exclusive_json(run / "verification.json", {})
            with self.assertRaises(ValueError):
                helper.validate_run_directory(experiment, experiment)
            with self.assertRaises(ValueError):
                helper.run_file(run, "../outside.json")

    def test_no_main_import_no_fit_predict_and_fixed_primary(self):
        module = self.module()
        self.assertEqual(module.PRIMARY, "grid_all_legacy_origins")
        self.assertEqual(set(module.ARMS), {"archive", "interval_candidate", module.PRIMARY})
        tree = ast.parse((HERE / "verify_direction.py").read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("direction_audit" in name for name in imports))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(set(calls) & {"fit", "fit_bounds", "predict", "train"})

    def test_every_arm_bucket_block_including_empty_is_preserved(self):
        module = self.module()
        frame = self.frame([10.] * 8, predictions=[1.] * 8)
        frame["year"], frame["reason"] = 2024, "exact"
        frame["cutoff"], frame["label_available_at"] = frame.timestamp, frame.timestamp
        signals, current, members, summary, counts = module.recount(frame)
        self.assertEqual(len(signals), 8 * 3 * 2)
        self.assertEqual(len(current), len(signals))
        self.assertEqual(summary["all"]["combinations"], len(signals))
        self.assertEqual(summary["all"]["current_empty_combinations"], 24)
        self.assertEqual(summary["all"]["empty_prefix_combinations"], 27)
        self.assertEqual(summary["all"]["signal_combinations"], 6)
        self.assertEqual(summary["all"]["comparable_combinations"], 6)
        self.assertEqual(summary["all"]["consistent_fraction_all"], 6 / 48)
        self.assertEqual(summary["all"]["consistent_fraction_comparable"], 1.)
        self.assertEqual(summary["by_arm_bucket"][module.PRIMARY]["ge24"]["combinations"], 8)
        self.assertIsNone(summary["by_arm_bucket"][module.PRIMARY]["ge24"]["consistent_fraction_comparable"])
        self.assertEqual(set(summary["all"]["direction_counts"]), set(module.DIRECTIONS))
        self.assertEqual(counts["prefix_member_rows"], 28 * 3)
        self.assertEqual(len(members), counts["prefix_member_rows"])
        last = members.loc[members.evaluation_block.eq(7) & members.arm.eq(module.PRIMARY)]
        self.assertEqual(last.origin_index.tolist(), list(range(7)))
        self.assertTrue((last.label_available_at < last.cutoff).all())

    def test_tied_blocks_do_not_disappear_from_vote_denominator(self):
        module = self.module()
        frame = self.frame([10.] * 12, blocks=np.repeat(range(6), 2),
                           predictions=[1.] * 6 + [1., 20.] * 3)
        stats = module.rates(frame, module.PRIMARY)
        self.assertEqual(stats["blocks"], 6)
        self.assertEqual(stats["early_majority_blocks"], 3)
        self.assertEqual(stats["block_early_lower"], .75)
        self.assertEqual(module.signal(stats), "none")

    def test_current_empty_agreement_priority_and_unknown_denominator(self):
        module = self.module()
        self.assertEqual(module.agreement("insufficient", "current_empty"), "current_empty")
        self.assertEqual(module.agreement("none", "unknown"), "no_signal")
        self.assertEqual(module.agreement("raise", "unknown"), "unknown")
        rows = pd.DataFrame({"past_rows": [0, 8, 8], "support_sufficient": [False, True, True],
                             "direction": ["insufficient", "raise", "none"],
                             "current_direction": ["current_empty", "unknown", "lower"],
                             "agreement": ["current_empty", "unknown", "no_signal"]})
        summary = module.summary_leaf(rows)
        self.assertEqual(summary["unknown_fraction_all"], 1 / 3)
        self.assertEqual(summary["current_empty_combinations"], 1)
        self.assertEqual(summary["insufficient_history_combinations"], 1)
        self.assertEqual(summary["no_signal_combinations"], 1)
        self.assertEqual(summary["comparable_combinations"], 0)
        self.assertIsNone(summary["consistent_fraction_comparable"])

    def test_real_artifact_mutation_and_inventory_addition_are_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            module.bridge().write_exclusive_json(path / "a.json", {"lower": 1})
            inventory, hashes = module.artifact_locks(path)
            module.validate_artifact_locks(path, inventory, hashes)
            (path / "a.json").write_text(json.dumps({"lower": 2}), encoding="utf-8")
            with self.assertRaises(ValueError):
                module.validate_artifact_locks(path, inventory, hashes)
            inventory, hashes = module.artifact_locks(path)
            module.bridge().write_exclusive_json(path / "foreign.json", {})
            with self.assertRaises(ValueError):
                module.validate_artifact_locks(path, inventory, hashes)

    def test_nested_report_is_not_silently_omitted_from_inventory(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "nested").mkdir()
            helper = module.bridge()
            helper.write_exclusive_json(path / "report.json", {})
            helper.write_exclusive_json(path / "signals.json", {})
            expected = {"signals.json": helper.sha256(path / "signals.json")}
            module.validate_output_inventory(path, expected, {"signals.json"})
            helper.write_exclusive_json(path / "nested/report.json", {})
            with self.assertRaises(ValueError):
                module.validate_output_inventory(path, expected, {"signals.json"})

    def test_exact_parent_chain_rejects_wrong_report_verifier_or_scope(self):
        module = self.module()
        evidence = {"schema_version": 1, "status": "passed", "independent_verification_passed": True,
                    "report_sha256": module.PARENT_REPORT_SHA256, "verifier_sha256": module.PARENT_VERIFIER_SHA256,
                    "reused_independent_verifier_sha256": module.BRIDGE_SHA256,
                    "fixed_origins_verified": 701, "models_verified": 64, "refitted_predictions_verified": 1402,
                    "primary_candidate": module.PRIMARY, "production_ready": False, "deployed": False}
        module.validate_parent_evidence(evidence, module.PARENT_REPORT_SHA256, module.PARENT_VERIFICATION_SHA256)
        for key, value in (("report_sha256", "wrong"), ("verifier_sha256", "wrong"), ("production_ready", True)):
            changed = {**evidence, key: value}
            with self.assertRaises(ValueError):
                module.validate_parent_evidence(changed, module.PARENT_REPORT_SHA256, module.PARENT_VERIFICATION_SHA256)
        with self.assertRaises(ValueError):
            module.validate_parent_evidence(evidence, "wrong", module.PARENT_VERIFICATION_SHA256)

    def test_invalid_prediction_bounds_and_duplicate_origins_fail_closed(self):
        module = self.module()
        with self.assertRaises(ValueError):
            module.rates(self.frame([10.], predictions=[np.nan]), module.PRIMARY)
        with self.assertRaises(ValueError):
            module.rates(self.frame([10.], [9.]), module.PRIMARY)
        frame = self.frame([10.] * 2)
        frame["year"], frame["reason"] = 2024, "exact"
        frame["cutoff"] = frame.timestamp
        frame["origin_index"] = 0
        with self.assertRaises(ValueError):
            module.recount(frame)

    def test_declared_scope_cannot_claim_training_deployment_or_accuracy_gate(self):
        module = self.module()
        report = {"schema_version": 1, "status": "pending_independent_verification", "years": [2024, 2025],
                  "arms": list(module.ARMS), "buckets": list(module.BUCKETS), "primary_arm": module.PRIMARY,
                  "configuration": module.CONFIGURATION, "scope": module.SCOPE,
                  "fit_performed": False, "predict_performed": False, "calibration_performed": False,
                  "deployed": False, "production_ready": False, "blind_test": False,
                  "descriptive_screen_only": True, "historical_shadow_evaluation": True,
                  "protected_unchanged": True, "parent_unchanged": True}
        module.validate_scope(report)
        for key in ("fit_performed", "predict_performed", "production_ready", "blind_test"):
            with self.assertRaises(ValueError):
                module.validate_scope({**report, key: True})
        with self.assertRaises(ValueError):
            module.validate_scope({**report, "primary_arm": "archive"})
        with self.assertRaises(ValueError):
            module.validate_scope({**report, "yearly_acceptance_passed": True})
        changed_config = {**module.CONFIGURATION, "strict_timestamp_before_cutoff": 1}
        with self.assertRaises(ValueError):
            module.validate_scope({**report, "configuration": changed_config})

    def test_wrong_block_cutoff_or_table_identity_is_rejected(self):
        module = self.module()
        frame = self.frame([10.] * 2, blocks=[1, 1])
        frame["year"], frame["reason"] = 2024, "exact"
        frame["cutoff"] = frame.timestamp
        with self.assertRaises(ValueError):
            module.recount(frame)
        expected = pd.DataFrame({"evaluation_block": [1], "cutoff": frame.timestamp.iloc[:1],
                                 "evaluation_year": [2024], "arm": [module.PRIMARY], "bucket": ["lt24"],
                                 "past_rows": [1]})
        changed = expected.copy()
        changed.loc[0, "past_rows"] = 2
        with self.assertRaises(ValueError):
            module.compare_table(expected, changed, "all signals")
        changed = expected.copy()
        changed["foreign"] = 1
        with self.assertRaises(ValueError):
            module.compare_table(expected, changed, "all signals")

    def test_manifest_conflict_or_missing_new_protocol_lock_is_rejected(self):
        module = self.module()
        helper = module.bridge()
        with self.assertRaises(ValueError):
            helper.combine_locks([{"a": "aaa"}, {"a": "bbb"}])
        with self.assertRaises(ValueError):
            module.validate_code_locks({}, Path("J:/project"))

    def test_protected_inventory_detects_added_file_and_missing_directory(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protected = root / "protected"
            protected.mkdir()
            helper = module.bridge()
            helper.write_exclusive_json(protected / "old.json", {})
            locks = {str((protected / "old.json").resolve()): helper.sha256(protected / "old.json")}
            module.validate_protected_inventory(locks, root, ("protected",))
            helper.write_exclusive_json(protected / "extra.json", {})
            with self.assertRaises(ValueError):
                module.validate_protected_inventory(locks, root, ("protected",))
            with self.assertRaises(ValueError):
                module.validate_protected_inventory(locks, root, ("missing",))

    def test_evaluation_year_is_origin_year_not_cutoff_year(self):
        module = self.module()
        frame = self.frame([10.])
        frame["year"], frame["reason"] = 2024, "exact"
        frame["cutoff"] = pd.Timestamp("2023-12-31", tz="Asia/Shanghai")
        signals, current, _, _, _ = module.recount(frame)
        self.assertEqual(set(signals.evaluation_year), {2024})
        self.assertEqual(set(current.evaluation_year), {2024})


if __name__ == "__main__":
    unittest.main()
