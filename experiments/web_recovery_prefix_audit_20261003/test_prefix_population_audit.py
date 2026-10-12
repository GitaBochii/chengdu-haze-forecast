"""Synthetic prefix-population checks, without fitting or predicting models."""

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


class PrefixPopulationTests(unittest.TestCase):
    def module(self):
        path = HERE / "prefix_population_audit.py"
        self.assertTrue(path.is_file(), "prefix audit is not implemented")
        spec = importlib.util.spec_from_file_location("independent_prefix_population_audit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def observations(self, aqi):
        return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=len(aqi), freq="h", tz="Asia/Shanghai"),
                             "aqi": aqi, "pm2_5": 100., "pm10": 180.})

    def completed(self):
        return self.observations([90.] * 24 + [130.] * 3 + [90.] * 9)

    def test_full_event_waits_for_six_actual_clean_hours(self):
        module = self.module()
        raw = self.completed()
        self.assertTrue(module.completed_legacy_origins(raw.iloc[:32]).empty)
        selected = module.completed_legacy_origins(raw.iloc[:33])
        self.assertEqual(selected.timestamp.tolist(), raw.timestamp.iloc[24:27].tolist())
        self.assertEqual(selected.old_confirmation_at.tolist(), [raw.timestamp.iloc[32]] * 3)
        self.assertEqual(selected.old_event_id.tolist(), [1] * 3)

    def test_first_two_hours_backfilled_only_after_completion(self):
        module = self.module()
        raw = self.completed()
        self.assertTrue(module.completed_legacy_origins(raw.iloc[:26]).empty)
        selected = module.completed_legacy_origins(raw)
        self.assertIn(raw.timestamp.iloc[24], selected.timestamp.tolist())
        self.assertIn(raw.timestamp.iloc[25], selected.timestamp.tolist())

    def test_missing_start_hour_prevents_three_hour_start(self):
        module = self.module()
        raw = self.completed().drop(index=25)
        self.assertTrue(module.completed_legacy_origins(raw).empty)

    def test_clean_gap_cannot_confirm_six_hour_recovery(self):
        module = self.module()
        raw = self.completed().iloc[:33].drop(index=30)
        self.assertTrue(module.completed_legacy_origins(raw).empty)

    def test_ongoing_archive_tail_is_excluded_not_fake_completed(self):
        module = self.module()
        raw = self.observations([90.] * 24 + [130.] * 20)
        self.assertTrue(module.completed_legacy_origins(raw).empty)

    def test_particle_iaqi_needs_twenty_four_observed_rows(self):
        module = self.module()
        raw = self.observations([130.] * 10 + [90.] * 6)
        self.assertTrue(module.completed_legacy_origins(raw).empty)

    def test_historical_row_clock_behavior_is_reproduced_not_corrected(self):
        module = self.module()
        raw = self.observations([90.] * 24 + [130.] * 6 + [90.] * 6)
        raw = raw.drop(index=27)
        selected = module.completed_legacy_origins(raw)
        self.assertEqual(len(selected), 5, "old membership keeps a middle event gap")
        self.assertEqual(selected.old_confirmation_at.iloc[0], raw.timestamp.iloc[-1])

    def test_canonical_grid_retains_gap_and_rejects_source_overlap(self):
        module = self.module()
        raw = self.completed().drop(index=25)
        grid = module.canonical_grid({"s": raw})
        self.assertEqual(len(grid), 36)
        self.assertTrue(np.isnan(grid.loc[25, "aqi"]))
        with self.assertRaises(ValueError):
            module.canonical_grid({"s": raw, "other": raw.iloc[:1]})

    def test_prefix_bounds_use_window_start_and_full_confirmation_maturity(self):
        module = self.module()
        raw = self.completed()
        old = module.canonical_math()
        bounds = old.recover_bounds(module.canonical_grid({"s": raw.iloc[:33]}), raw.timestamp.iloc[24:27])
        self.assertEqual(bounds.exact_hours.tolist(), [3., 2., 1.])
        self.assertEqual(bounds.label_available_at.tolist(), [raw.timestamp.iloc[32]] * 3)

    def test_blocks_change_after_six_clean_not_at_missing_gap(self):
        module = self.module()
        grid = self.observations([130., np.nan, 130.] + [90.] * 7)
        blocks = module.continuity_blocks(grid)
        self.assertEqual(blocks.tolist(), [1] * 9 + [2])

    def test_output_is_explicit_inside_audit_directory_and_exclusive(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as folder:
            allowed = Path(folder) / "audit"
            allowed.mkdir()
            path = allowed / "result.json"
            module.write_result(path, {"status": "passed"}, allowed)
            with self.assertRaises(FileExistsError):
                module.write_result(path, {"status": "failed"}, allowed)
            with self.assertRaises(ValueError):
                module.write_result(Path(folder) / "outside.json", {}, allowed)

    def test_no_production_label_runner_or_model_calls(self):
        self.module()
        tree = ast.parse((HERE / "prefix_population_audit.py").read_text(encoding="utf-8"))
        forbidden = {"fit", "fit_bounds", "predict", "train", "label_pollution_events", "label_city_data", "training_rows"}
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))]
        self.assertFalse(set(calls) & forbidden)

    def population(self, module):
        raw = self.completed()
        selected = module.completed_legacy_origins(raw)
        population = module.canonical_math().recover_bounds(module.canonical_grid({"s": raw}), selected.timestamp)
        population["population_row"] = np.arange(len(population))
        population["continuity_block"] = module.continuity_blocks(module.canonical_grid({"s": raw}))[24]
        population["legacy_remaining_hours"] = [3., 2., 1.]
        members = population.assign(model_id="legacy_control_b9", arm="legacy_control")
        return raw, population, members

    def test_audit_block_compares_exact_population_and_prefix_canonical_bounds(self):
        module = self.module()
        raw, population, members = self.population(module)
        result = module.audit_block({"s": raw}, population, members, raw.timestamp.iloc[33], 9)
        self.assertEqual(result["parent_training_rows"], 3)
        self.assertEqual(result["missing_rows"], 0)
        self.assertTrue(result["canonical_bounds_and_maturity_match"])
        with self.assertRaises(ValueError):
            module.audit_block({"s": raw}, population, members.iloc[1:], raw.timestamp.iloc[33], 9)
        changed = members.copy()
        changed.loc[0, "lower_bound_hours"] = 30.
        with self.assertRaises(ValueError):
            module.audit_block({"s": raw}, population, changed, raw.timestamp.iloc[33], 9)

    def test_early_cutoff_cannot_use_future_selected_members(self):
        module = self.module()
        raw, population, members = self.population(module)
        with self.assertRaises(ValueError):
            module.audit_block({"s": raw}, population, members, raw.timestamp.iloc[32], 9)

    def test_original_lock_conflicts_cannot_be_masked_by_current_source_hashes(self):
        module = self.module()
        self.assertEqual(module.merge_locks([{"x": "a" * 64}, {"x": "a" * 64}]), {"x": "a" * 64})
        with self.assertRaises(ValueError):
            module.merge_locks([{"x": "a" * 64}, {"x": "b" * 64}])


if __name__ == "__main__":
    unittest.main()
