"""Independent verifier tests use synthetic observations, never research runs."""

import importlib.util
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


class IndependentVerifierTests(unittest.TestCase):
    def verifier(self):
        path = HERE / "verify_bridge.py"
        self.assertTrue(path.is_file(), "independent verifier is not implemented")
        spec = importlib.util.spec_from_file_location("independent_bridge_verifier", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def observations(self, aqi):
        return pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=len(aqi), freq="h", tz="Asia/Shanghai"),
            "aqi": aqi,
            "pm2_5": np.full(len(aqi), 100.0),
            "pm10": np.full(len(aqi), 180.0),
            "wind_u_ms": np.full(len(aqi), 3.0),
            "wind_v_ms": np.full(len(aqi), 4.0),
        })

    def test_gap_grid_preserves_unknown_and_true_24_hour_lag(self):
        module = self.verifier()
        raw = self.observations(np.arange(27.0) + 101)
        raw.loc[:, "pm2_5"] = np.arange(27.0) + 76
        raw = raw.drop(index=4)
        grid = module.rebuild_grid({"s": raw})
        self.assertEqual(len(grid), 27)
        self.assertTrue(np.isnan(grid.loc[4, "aqi"]))
        self.assertEqual(grid.loc[25, "aqi_lag_24"], 102)
        self.assertTrue(np.isnan(grid.loc[27 - 1, "pm2_5_iaqi"]))
        self.assertEqual(grid.loc[26, "wind_speed_ms"], 5)
        self.assertFalse(bool(grid.loc[4, "input_hour_observed"]))

    def test_iaqi_breakpoint_and_missing_indicator(self):
        module = self.verifier()
        raw = self.observations(np.full(25, 130.0))
        raw["pm2_5"] = 75.0
        raw["pm10"] = 150.0
        grid = module.rebuild_grid({"s": raw})
        self.assertEqual(grid.loc[24, "pm2_5_iaqi"], 100)
        self.assertFalse(bool(grid.loc[24, "particle_primary"]))
        self.assertEqual(grid.loc[24, "boundary_layer_height_m_is_missing"], 1)
        self.assertTrue(np.isnan(grid.loc[23, "aqi_lag_24"]))
        self.assertEqual(grid.loc[24, "aqi_lag_24"], 130)

    def test_causal_first_two_hours_not_eligible_gap_does_not_reset_block(self):
        module = self.verifier()
        raw = self.observations([130, 130, 130, np.nan, 130, 130, 130] + [90] * 7)
        raw["pm2_5_iaqi"] = 150.0
        raw["pm10_iaqi"] = 150.0
        state = module.rebuild_states(raw)
        self.assertEqual(state.eligible.tolist()[:7], [False, False, True, False, False, False, True])
        self.assertEqual(state.continuity_block.tolist()[:13], [1] * 13)
        self.assertEqual(state.loc[13, "continuity_block"], 2)
        self.assertEqual(state.loc[6, "episode_id"], 2)

    def test_exact_label_starts_window_not_confirmation(self):
        module = self.verifier()
        raw = self.observations([130, 130] + [90] * 6)
        result = module.recover_bounds(raw, raw.timestamp.iloc[:1])
        self.assertEqual(result.loc[0, "exact_hours"], 2)
        self.assertEqual(result.loc[0, "label_available_at"], raw.timestamp.iloc[7])

    def test_missing_window_yields_interval_and_later_maturity(self):
        module = self.verifier()
        raw = self.observations([130] + [np.nan] * 6 + [130] + [90] * 6)
        result = module.recover_bounds(raw, raw.timestamp.iloc[:1])
        self.assertEqual(result.loc[0, "lower_bound_hours"], 1)
        self.assertEqual(result.loc[0, "upper_bound_hours"], 8)
        self.assertTrue(np.isnan(result.loc[0, "exact_hours"]))
        self.assertEqual(result.loc[0, "label_available_at"], raw.timestamp.iloc[13])

    def test_right_censor_is_not_fake_exact_and_matures_only_at_archive_end(self):
        module = self.verifier()
        raw = self.observations([130] * 8)
        result = module.recover_bounds(raw, raw.timestamp.iloc[[0, 7]])
        self.assertEqual(result.loc[0, "lower_bound_hours"], 8)
        self.assertTrue(np.isposinf(result.loc[0, "upper_bound_hours"]))
        self.assertTrue(np.isnan(result.loc[0, "exact_hours"]))
        self.assertEqual(result.loc[0, "label_available_at"], raw.timestamp.iloc[-1])
        self.assertEqual(result.loc[1, "lower_bound_hours"], 1)

    def test_train_members_strict_cutoff_and_not_current_block(self):
        module = self.verifier()
        times = pd.date_range("2020-01-01", periods=5, freq="h", tz="Asia/Shanghai")
        frame = pd.DataFrame({"timestamp": times, "label_available_at": times.shift(1, freq="h"),
                              "continuity_block": [1, 1, 2, 1, 1]})
        selected = module.expected_members(frame, times[3], 2)
        self.assertEqual(selected.timestamp.tolist(), times[:2].tolist())

    def test_paired_interval_bounds_share_truth_and_handle_infinity(self):
        module = self.verifier()
        frame = pd.DataFrame({"lower_bound_hours": [2.0, 5.0, 3.0],
                              "upper_bound_hours": [2.0, np.inf, 8.0]})
        low, high = module.paired_bounds(frame, np.array([4, 2, 5]), np.array([1, 7, 1]))
        np.testing.assert_allclose(low, [-1, -5, 0])
        np.testing.assert_allclose(high, [-1, -1, 4])

    def test_score_keeps_censor_and_weights_processes_equally(self):
        module = self.verifier()
        frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC"),
                              "continuity_block": [1, 1, 2, 3], "lower_bound_hours": [2, 4, 30, 5],
                              "upper_bound_hours": [2, 4, 30, np.inf], "exact_hours": [2, 4, 30, np.nan]})
        score = module.score_predictions(frame, np.array([1, 3, 20, 1]))
        self.assertEqual(score["rows"], 4)
        self.assertEqual(score["right_censored_rows"], 1)
        self.assertEqual(score["exact_row_mae"], 4)
        self.assertEqual(score["exact_block_mae"], 5.5)
        self.assertEqual(score["tail24_block_mae"], 10)
        self.assertIn("exact_signed_bias", score, "score schema needs signed bias")
        self.assertEqual(score["exact_signed_bias"], -4)
        self.assertEqual(score["exact_early_fraction"], 1)
        self.assertEqual(score["exact_late_fraction"], 0)
        self.assertEqual(score["tail24_blocks"], 1)
        self.assertEqual(score["subgroups"]["right_censored"]["row_distance"], 4)
        self.assertIsNone(score["all_row_error_upper"])

    def test_gate_rejects_missing_long_tail_even_with_mae_gain(self):
        module = self.verifier()
        score = {"prediction_coverage": 1.0, "exact_row_mae": 4.0,
                 "exact_block_mae": 4.0, "tail24_block_mae": None,
                 "rows": 10, "exact_rows": 10, "exact_blocks": 2, "tail24_rows": 0}
        archive = dict(score, exact_row_mae=10.0, exact_block_mae=10.0)
        paired = {"all_row_delta_upper": -1.0, "all_block_delta_upper": -1.0}
        gate = module.acceptance_gate(archive, score, paired)
        self.assertFalse(gate["passed"])
        self.assertIn("checks", gate, "gate schema needs explicit checks")
        self.assertFalse(gate["checks"]["tail24_block_not_worse"])

    def test_hash_lock_rejects_changed_file(self):
        module = self.verifier()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.txt"
            path.write_text("before", encoding="ascii")
            locked = module.sha256(path)
            path.write_text("after", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "hash"):
                module.verify_hashes({str(path): locked}, Path(temporary))

    def test_exclusive_json_never_overwrites(self):
        module = self.verifier()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "verification.json"
            module.write_exclusive_json(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                module.write_exclusive_json(path, {"ok": False})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})

    def test_independence_has_no_runner_or_shared_logic_imports(self):
        self.verifier()
        syntax = ast.parse((HERE / "verify_bridge.py").read_text(encoding="utf-8"))
        prohibited = {"bridge_runner", "interval_recovery_labels", "observed_business", "observed_hourly_features",
                      "features", "labels", "cohort_scorecard", "paired_interval_score", "observed_recovery_labels"}
        for node in ast.walk(syntax):
            if isinstance(node, ast.ImportFrom):
                self.assertTrue(prohibited.isdisjoint((node.module or "").split(".")))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertTrue(prohibited.isdisjoint(alias.name.split(".")))

    def test_strict_frame_comparison_rejects_dropped_or_reordered_members(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "compare_frame"), "strict frame comparison is not implemented")
        original = pd.DataFrame({"population_row": [0, 1, 3], "value": [1.0, np.nan, np.inf]})
        module.compare_frame(original, original.copy(), list(original), "members")
        with self.assertRaisesRegex(ValueError, "members"):
            module.compare_frame(original, original.iloc[:2], list(original), "members")
        with self.assertRaisesRegex(ValueError, "members"):
            module.compare_frame(original, original.iloc[::-1], list(original), "members")

    def test_training_matrix_digest_preserves_order_and_nan(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "matrix_digest"), "matrix digest is not implemented")
        values = np.array([[1, np.nan], [2, 3]], dtype=float)
        expected = hashlib.sha256(b'{"dtype": "<f8", "shape": [2, 2]}\n' + values.astype('<f8').tobytes()).hexdigest()
        self.assertEqual(module.matrix_digest(values), expected)
        self.assertEqual(module.matrix_digest(values), module.matrix_digest(values.copy()))
        self.assertNotEqual(module.matrix_digest(values), module.matrix_digest(values[::-1]))
        self.assertNotEqual(module.matrix_digest(values), module.matrix_digest(np.nan_to_num(values)))

    def test_run_paths_reject_parent_traversal_and_nonrun_destination(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "run_file"), "run path validation is not implemented")
        self.assertTrue(hasattr(module, "validate_run_directory"), "run destination validation is not implemented")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "fixed"
            run.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "outside"):
                module.run_file(run, "../foreign.json")
            with self.assertRaisesRegex(ValueError, "runs"):
                module.validate_run_directory(root, root)

    def test_numeric_comparison_rejects_nan_infinity_and_real_value_changes(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "compare_array"), "numeric comparison is not implemented")
        values = np.array([1.0, np.nan, np.inf])
        module.compare_array(values, values.copy(), "values")
        for wrong in [np.array([1, 0, np.inf]), np.array([1, np.nan, 100]), np.array([1.1, np.nan, np.inf])]:
            with self.assertRaisesRegex(ValueError, "values"):
                module.compare_array(values, wrong, "values")

    def test_recursive_score_comparison_requires_every_field(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "compare_json"), "recursive comparison is not implemented")
        expected = {"arms": {"archive": {"rows": 3, "exact_row_mae": 4.0}}, "passed": False}
        module.compare_json(expected, expected, "scores")
        for wrong in [{"arms": expected["arms"]}, {"arms": {"archive": {"rows": 3, "exact_row_mae": 1.0}}, "passed": False}]:
            with self.assertRaisesRegex(ValueError, "scores"):
                module.compare_json(expected, wrong, "scores")

    def test_infinite_exact_label_is_rejected(self):
        module = self.verifier()
        frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=1, freq="h", tz="UTC"),
                              "continuity_block": [1], "lower_bound_hours": [2],
                              "upper_bound_hours": [np.inf], "exact_hours": [np.inf]})
        with self.assertRaisesRegex(ValueError, "exact"):
            module.score_predictions(frame, [1])

    def test_legacy_population_keeps_origins_not_in_causal_population(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "rebuild_legacy"), "legacy reconstruction is not implemented")
        grid = module.rebuild_grid({"raw": self.observations([130] * 4 + [90] * 6)})
        legacy = self.observations([130] * 3)
        legacy["event_id"] = 1
        legacy["remaining_hours"] = [4, 3, 2]
        restored = module.rebuild_legacy({"legacy": legacy}, grid, ["aqi"])
        self.assertEqual(len(restored), 3)
        self.assertFalse(grid.eligible.any())
        self.assertEqual(restored.population_row.tolist(), [0, 1, 2])
        self.assertEqual(restored.legacy_remaining_hours.tolist(), [4, 3, 2])
        self.assertEqual(restored.exact_hours.tolist(), [4, 3, 2])

    def test_evaluation_reconstruction_keeps_missing_and_ineligible_rows(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "rebuild_evaluation"), "evaluation reconstruction is not implemented")
        grid = module.rebuild_grid({"raw": self.observations([130, np.nan, 130] + [90] * 6)})
        archive = grid.iloc[:3][["timestamp", "aqi"]].copy()
        archive["event_key"] = "legacy:1"
        archive["remaining_hours"] = [3, 2, 1]
        archive["strict_prediction_hours"] = [7, 8, 9]
        evaluation = module.rebuild_evaluation(archive, grid)
        self.assertEqual(len(evaluation), 3)
        self.assertFalse(evaluation.causal_eligible.any())
        self.assertEqual(evaluation.archive_prediction_hours.tolist(), [7, 8, 9])
        self.assertTrue(evaluation.cutoff.eq(grid.timestamp.iloc[0]).all())

    def test_zero_archive_error_cannot_claim_relative_improvement(self):
        module = self.verifier()
        score = {"prediction_coverage": 1.0, "exact_row_mae": 0.0, "exact_block_mae": 0.0,
                 "tail24_block_mae": 0.0, "rows": 2, "exact_rows": 2, "exact_blocks": 1,
                 "tail24_rows": 1, "tail24_blocks": 1}
        paired = {"all_row_delta_upper": 0.0, "all_block_delta_upper": 0.0}
        gate = module.acceptance_gate(score, score, paired)
        self.assertFalse(gate["passed"])

    def test_hash_union_rejects_conflicting_shared_file(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "combine_locks"), "lock union is not implemented")
        self.assertEqual(module.combine_locks([{"a": "00"}, {"a": "00", "b": "11"}]), {"a": "00", "b": "11"})
        with self.assertRaisesRegex(ValueError, "conflict"):
            module.combine_locks([{"a": "00"}, {"a": "ff"}])

    def test_inventory_requires_both_arms_for_every_block(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "validate_inventory"), "inventory check is not implemented")
        rows = pd.DataFrame({"continuity_block": [1, 2]})
        records = [{"model_id": f"{arm}_b{block}", "arm": arm, "continuity_block": block}
                   for block in [1, 2] for arm in ["legacy_control", "interval_candidate"]]
        module.validate_inventory(records, rows)
        with self.assertRaisesRegex(ValueError, "inventory"):
            module.validate_inventory(records[:-1], rows)
        with self.assertRaisesRegex(ValueError, "inventory"):
            module.validate_inventory(records + records[:1], rows)

    def test_actual_trained_configuration_rejects_wrong_seed_and_rounds(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "validate_trained_config"), "trained parameter validation is not implemented")
        cfg = {"learner": {"generic_param": {"seed": "42", "nthread": "1"},
            "learner_model_param": {"num_feature": "87"},
            "objective": {"name": "survival:aft", "aft_loss_param": {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": "1.20000005"}},
            "gradient_booster": {"gbtree_model_param": {"num_trees": "500"},
                "gbtree_train_param": {"tree_method": "hist"}, "tree_train_param": {
                    "eta": ".05", "max_depth": "4", "min_child_weight": "8", "subsample": ".9",
                    "colsample_bytree": ".8", "lambda": "3"}}}}
        module.validate_trained_config(cfg, 87)
        wrong = json.loads(json.dumps(cfg))
        wrong["learner"]["generic_param"]["seed"] = "99"
        with self.assertRaisesRegex(ValueError, "seed"):
            module.validate_trained_config(wrong, 87)
        wrong = json.loads(json.dumps(cfg))
        wrong["learner"]["gradient_booster"]["gbtree_model_param"]["num_trees"] = "499"
        with self.assertRaisesRegex(ValueError, "round"):
            module.validate_trained_config(wrong, 87)

    def test_complete_score_set_and_process_rows_reject_tampering(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "compute_scores"), "complete score reconstruction is not implemented")
        frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "year": [2024] * 3, "continuity_block": [1, 1, 2], "event_key": ["a", "a", "b"],
            "lower_bound_hours": [30., 29., 2.], "upper_bound_hours": [30., 29., np.inf],
            "exact_hours": [30., 29., np.nan]})
        prediction = {"archive": np.array([10, 10, 3]), "legacy_control": np.array([20, 20, 3]),
                      "interval_candidate": np.array([30, 29, 3])}
        scores, process = module.compute_scores(frame, prediction, years=(2024,))
        self.assertEqual(len(process), 6)
        self.assertTrue(scores["gates"]["passed"])
        wrong = json.loads(json.dumps(scores))
        wrong["annual"]["2024"]["arms"]["interval_candidate"]["exact_row_mae"] = 9.
        with self.assertRaisesRegex(ValueError, "scores"):
            module.compare_json(scores, wrong, "scores")

    def test_run_verification_rejects_unapproved_output_directory_before_reading_models(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "verify_run"), "run verification is not implemented")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "runs"):
                module.verify_run(Path(temporary), root=Path(temporary))

    def test_report_verification_fails_closed_without_hash_groups(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "validate_run_report"), "report validation is not implemented")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "hash"):
                module.validate_run_report({}, Path(temporary), Path(temporary))

    def test_utc_nanosecond_timestamp_comparison_is_exact(self):
        module = self.verifier()
        values = np.array([1700000000000000000], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            module.compare_array(values, values + 1, "timestamp")

    def test_historical_audit_exception_requires_exact_app_pair(self):
        module = self.verifier()
        self.assertTrue(hasattr(module, "historical_drift_records"), "historical drift audit is not implemented")
        root = Path("J:/大创").resolve()
        app = str(root / "src/haze_forecast/app.py")
        original = {app: "92bc58a1a818ce6ecaa58d6d8c0d4d01e6fac603eb21b346f2e8b4ad0396e1f7", "other": "unchanged"}
        current = {app: "7f222647d2a9bc31ee3040b121f987da5b95621fe2913a9c697e175b1568724c", "other": "unchanged"}
        drift = module.historical_drift_records(original, current, root)
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]["path"], app)
        for wrong in [dict(current, other="changed"), dict(current, **{app: "unknown"}), dict(original)]:
            with self.assertRaisesRegex(ValueError, "historical|drift"):
                module.historical_drift_records(original, wrong, root)
        with self.assertRaisesRegex(ValueError, "historical|drift"):
            module.historical_drift_records(dict(original, **{app: "unknown"}), current, root)


if __name__ == "__main__":
    unittest.main()
