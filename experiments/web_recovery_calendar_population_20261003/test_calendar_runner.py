"""Synthetic calendar population tests, without real fitting or prediction."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import calendar_runner as runner


def grid(n=24):
    return pd.DataFrame(dict(timestamp=pd.date_range('2024-12-30', periods=n, freq='h', tz='Asia/Shanghai'),
        aqi=np.full(n, 50.), x=np.arange(n, dtype=float), continuity_block=np.ones(n, dtype=int),
        pollution_state='clean', input_hour_observed=True, eligible=False))


def score_frame():
    time = list(pd.date_range('2024-01-01', periods=2, freq='h', tz='Asia/Shanghai'))
    time += list(pd.date_range('2025-01-01', periods=2, freq='h', tz='Asia/Shanghai'))
    return pd.DataFrame(dict(timestamp=time, eval_row=[0, 1, 2, 3], year=[2024, 2024, 2025, 2025],
        event_key=['a', 'b', 'c', 'd'], continuity_block=[1, 2, 3, 4],
        lower_bound_hours=[30., 10., 10., 10.], upper_bound_hours=[30., 10., 10., 10.],
        exact_hours=[30., 10., 10., 10.]))


class CalendarPopulationTests(unittest.TestCase):
    def test_population_keeps_clean_polluted_missing_unqualified_and_nan_features(self):
        full = grid()
        full.loc[0, 'aqi'] = 200.
        full.loc[1, 'aqi'] = np.nan
        full.loc[1, 'input_hour_observed'] = False
        full.loc[2, 'x'] = np.nan
        full.loc[3, 'eligible'] = True
        population = runner.calendar_population(full, ['aqi', 'x'])
        self.assertEqual(len(population), len(full))
        np.testing.assert_array_equal(population.population_row, np.arange(len(full)))
        np.testing.assert_array_equal(population.timestamp, full.timestamp)
        self.assertEqual(population.aqi_category.iloc[:4].tolist(), ['polluted', 'missing', 'clean', 'clean'])
        self.assertTrue(np.isnan(population.x.iloc[2]))
        self.assertFalse(population.input_hour_observed.iloc[1])
        self.assertEqual(int(population.eligible.sum()), 1)

    def test_clean_origin_still_has_future_t_one_not_zero(self):
        population = runner.calendar_population(grid(), ['aqi', 'x'])
        self.assertEqual(population.exact_hours.iloc[0], 1.)
        self.assertEqual(population.label_available_at.iloc[0], population.timestamp.iloc[6])
        self.assertTrue(population.lower_bound_hours.ge(1).all())
        self.assertEqual(population.label_type.iloc[0], 'exact')
        self.assertEqual(population.label_type.iloc[-1], 'right_censored')
        self.assertTrue(np.isposinf(population.upper_bound_hours.iloc[-1]))

    def test_gap_retains_interval_without_midpoint_or_drop(self):
        full = grid()
        full.loc[2, 'aqi'] = np.nan
        population = runner.calendar_population(full, ['aqi', 'x'])
        self.assertEqual(population.lower_bound_hours.iloc[1], 1.)
        self.assertEqual(population.upper_bound_hours.iloc[1], 2.)
        self.assertTrue(np.isnan(population.exact_hours.iloc[1]))
        self.assertEqual(population.label_type.iloc[1], 'interval')
        self.assertEqual(population.label_available_at.iloc[1], full.timestamp.iloc[8])

    def test_only_strict_time_maturity_and_current_block_control_members(self):
        population = runner.calendar_population(grid(), ['aqi', 'x'])
        cutoff = population.timestamp.iloc[12]
        population.loc[0, 'label_available_at'] = cutoff
        population.loc[1, 'continuity_block'] = 9
        selected = runner.training_members(population, cutoff, 9)
        self.assertEqual(selected.population_row.tolist(), [2, 3, 4, 5])
        self.assertTrue(selected.timestamp.lt(cutoff).all())
        self.assertTrue(selected.label_available_at.lt(cutoff).all())
        self.assertFalse(selected.continuity_block.eq(9).any())

    def test_mature_right_censors_and_uninformative_rows_remain(self):
        population = runner.calendar_population(grid(), ['aqi', 'x'])
        selected = runner.training_members(population, population.timestamp.iloc[-1] + pd.Timedelta(hours=1), 9)
        self.assertEqual(len(selected), len(population))
        self.assertEqual(int(selected.label_type.eq('right_censored').sum()), 6)
        self.assertTrue(selected.informative.eq(False).any())

    def test_cross_year_current_cutoff_does_not_use_issue_year_as_training_filter(self):
        full = grid(80)
        population = runner.calendar_population(full, ['aqi', 'x'])
        cutoff = pd.Timestamp('2024-12-31', tz='Asia/Shanghai')
        selected = runner.training_members(population, cutoff, 9)
        self.assertEqual(len(selected), 18)
        self.assertTrue(selected.year.eq(2024).all())
        self.assertTrue(population.year.eq(2025).any())

    def test_future_ground_truth_and_features_cannot_change_already_mature_members(self):
        full = grid(80)
        cutoff = full.timestamp.iloc[30]
        before = runner.training_members(runner.calendar_population(full, ['aqi', 'x']), cutoff, 9)
        full.loc[full.timestamp.ge(cutoff), 'aqi'] = 200.
        full.loc[full.timestamp.ge(cutoff), 'x'] = 10000.
        after = runner.training_members(runner.calendar_population(full, ['aqi', 'x']), cutoff, 9)
        pd.testing.assert_frame_equal(before, after)

    def test_common_old_members_keep_x_bounds_and_maturity_exactly(self):
        population = runner.calendar_population(grid(), ['aqi', 'x'])
        old = population.iloc[[1, 4, 7]].copy()
        runner.assert_common_members(population, old, ['aqi', 'x'])
        for name, value in [('x', -1.), ('lower_bound_hours', 99.), ('continuity_block', 9),
                ('label_available_at', pd.Timestamp('2025-01-01', tz='Asia/Shanghai'))]:
            changed = old.copy()
            changed.loc[changed.index[0], name] = value
            with self.subTest(name=name), self.assertRaises(AssertionError):
                runner.assert_common_members(population, changed, ['aqi', 'x'])

    def test_each_cutoff_reports_its_actual_exact_one_fraction(self):
        population = runner.calendar_population(grid(), ['aqi', 'x'])
        population.loc[0, ['lower_bound_hours', 'upper_bound_hours', 'exact_hours']] = 10.
        selected = runner.training_members(population, population.timestamp.iloc[12], 9)
        summary = runner.member_summary(selected, ['aqi', 'x'])
        self.assertEqual(summary['exact_t1_rows'], 5)
        self.assertAlmostEqual(summary['exact_t1_fraction'], 5/6)
        self.assertEqual(summary['aqi_category_counts'], {'clean': 6, 'polluted': 0, 'missing': 0})
        self.assertEqual(summary['causal_eligible_counts'], {'eligible': 0, 'ineligible': 6})

    def test_bad_grid_identity_and_infinite_features_fail_instead_of_drop(self):
        for mutate in ('duplicate', 'gap', 'inf'):
            full = grid()
            if mutate == 'duplicate':
                full.loc[1, 'timestamp'] = full.timestamp.iloc[0]
            elif mutate == 'gap':
                full = full.drop(index=1)
            else:
                full.loc[0, 'x'] = np.inf
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                runner.calendar_population(full, ['aqi', 'x'])


class ScoresAndChainTests(unittest.TestCase):
    def test_parent_factor_npz_uses_grid_arm_key_with_nan_float_order_and_time_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'features.npz'
            timestamps = pd.date_range('2024-12-30', periods=2, freq='h', tz='Asia/Shanghai')
            names = ['aqi', 'x']
            features = pd.DataFrame([[np.nan, 1.], [3., 4.]], columns=names)
            np.savez_compressed(path, timestamps=timestamps.asi8, feature_names=np.asarray(names),
                labels_only=np.full((2, 2), 99.), grid_all_legacy_origins=features.to_numpy())
            runner.assert_parent_features(path, timestamps, names, features)

    def test_parent_factor_npz_missing_grid_arm_rejects_old_candidate_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'features.npz'
            timestamps = pd.date_range('2024-12-30', periods=2, freq='h', tz='Asia/Shanghai')
            features = pd.DataFrame([[1., 2.], [3., 4.]], columns=['aqi', 'x'])
            np.savez_compressed(path, timestamps=timestamps.asi8, feature_names=features.columns,
                candidate=features.to_numpy())
            with self.assertRaises(ValueError):
                runner.assert_parent_features(path, timestamps, list(features.columns), features)

    def test_parent_factor_npz_order_time_and_float_matrix_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'features.npz'
            timestamps = pd.date_range('2024-12-30', periods=2, freq='h', tz='Asia/Shanghai')
            names = ['aqi', 'x']
            features = pd.DataFrame([[np.nan, 1.], [3., 4.]], columns=names)
            original = dict(timestamps=timestamps.asi8, feature_names=np.asarray(names),
                grid_all_legacy_origins=features.to_numpy())
            for key, value in [('feature_names', np.asarray(names[::-1])),
                    ('timestamps', timestamps.asi8[::-1]),
                    ('grid_all_legacy_origins', np.array([[0., 1.], [3., 4.]])),
                    ('grid_all_legacy_origins', np.array([[0, 1], [3, 4]], dtype=np.int64))]:
                np.savez_compressed(path, **dict(original, **{key: value}))
                with self.subTest(key=key), self.assertRaises((ValueError, AssertionError)):
                    runner.assert_parent_features(path, timestamps, names, features)

    def test_saved_references_are_inherited_identically_not_refitted(self):
        evaluation = score_frame()
        saved = evaluation.copy()
        for arm in runner.REFERENCE_ARMS:
            saved[arm] = [12., 20., 12., 20.]
        result = runner.inherit_references(evaluation, saved)
        self.assertEqual(tuple(result), runner.REFERENCE_ARMS)
        for arm in result:
            np.testing.assert_array_equal(result[arm], saved[arm])
        saved.loc[0, runner.REFERENCE_ARMS[0]] = np.nan
        with self.assertRaises(ValueError):
            runner.inherit_references(evaluation, saved)

    def test_changed_inherited_reference_cannot_reach_scores_as_a_second_factor(self):
        inherited = {arm: np.array([12., 20.]) for arm in runner.REFERENCE_ARMS}
        result = {arm: values.copy() for arm, values in inherited.items()}
        result[runner.PRIMARY_CANDIDATE] = np.array([10., 10.])
        runner.assert_inherited_predictions(result, inherited)
        result[runner.REFERENCE_ARMS[1]][0] += 1.
        with self.assertRaises(AssertionError):
            runner.assert_inherited_predictions(result, inherited)

    def test_gate_uses_only_fixed_calendar_candidate_against_archive(self):
        frame = score_frame()
        predictions = {arm: np.array([50., 30., 30., 30.]) for arm in runner.REFERENCE_ARMS}
        predictions[runner.PRIMARY_CANDIDATE] = np.array([30., 10., 10., 10.])
        metrics = runner.compute_scores(frame, predictions)
        self.assertTrue(metrics['gates']['annual']['2024']['passed'])
        self.assertFalse(metrics['gates']['annual']['2025']['passed'])
        self.assertFalse(metrics['gates']['passed'])
        self.assertEqual(set(metrics['pooled']['arms']), set(runner.ARMS))
        self.assertEqual(set(metrics['pooled']['paired']),
            {f'calendar_all_hours_vs_{arm}' for arm in runner.REFERENCE_ARMS})

    def test_model_contract_keeps_no_weights_and_actual_native_evidence(self):
        self.assertEqual(runner.bridge.MODEL_CONFIG, {'seed': 42, 'num_boost_round': 500, 'nthread': 1})
        self.assertEqual(runner.NEW_ARMS, ('calendar_all_hours',))
        self.assertEqual(runner.bridge.AFT_PARAMETERS['aft_loss_distribution_scale'], 1.2)
        with self.assertRaises(ValueError):
            runner.validate_fitted_model(499, ['aqi'], ['aqi'])
        with self.assertRaises(ValueError):
            runner.validate_fitted_model(500, ['wrong'], ['aqi'])
        runner.validate_fitted_model(500, ['aqi'], ['aqi'])

    def test_existing_native_model_fails_before_any_fit_or_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'models').mkdir()
            (run / 'models/calendar_all_hours_b9.json').write_text('existing', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                runner.fit_block({'feature_names': ['aqi']}, None, 9, None, run)

    def test_parent_passed_chain_and_conflicting_hashes_fail_closed(self):
        report = {'status': 'pending_independent_verification'}
        verification = dict(status='passed', independent_verification_passed=True,
            report_sha256=runner.PARENT_REPORT_SHA256, verifier_sha256=runner.PARENT_VERIFIER_SHA256)
        runner.validate_parent(report, verification, runner.PARENT_REPORT_SHA256)
        for field, value in [('status', 'failed'), ('report_sha256', 'other')]:
            with self.assertRaises(ValueError):
                runner.validate_parent(report, dict(verification, **{field: value}), runner.PARENT_REPORT_SHA256)
        with self.assertRaises(ValueError):
            runner.bridge.merge_hash_manifests({'a': 'x'}, {'a': 'y'})

    def test_inventory_hash_and_nonempty_protection_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            for name in ('report.json', 'verification.json', 'predictions.parquet'):
                (parent / name).write_text(name, encoding='utf-8')
            report = dict(output_sha256={'predictions.parquet': runner.bridge.sha(parent / 'predictions.parquet')})
            locked, inventory = runner.parent_artifact_manifest(parent, report)
            runner.assert_directory_inventory([parent], locked)
            manifests = dict(frozen_sha256=locked, parent_run_dir=str(parent), parent_artifact_inventory=inventory)
            runner.assert_locks(manifests)
            (parent / 'predictions.parquet').write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.assert_locks(manifests)
            with self.assertRaises(ValueError):
                runner.assert_directory_inventory([parent / 'missing'], {})
            (parent / 'empty').mkdir()
            with self.assertRaises(ValueError):
                runner.assert_directory_inventory([parent / 'empty'], {})
            (parent / 'extra').write_text('extra', encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.parent_artifact_manifest(parent, report)

    def test_utc_run_and_json_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            stamp = datetime(2026, 10, 3, tzinfo=timezone.utc)
            run = runner.bridge.new_run_directory(Path(directory), stamp)
            with self.assertRaises(FileExistsError):
                runner.bridge.new_run_directory(Path(directory), stamp)
            runner.write_new_json(run / 'report.json', {'status': 'test'})
            with self.assertRaises(FileExistsError):
                runner.write_new_json(run / 'report.json', {'status': 'overwrite'})
            invalid = run / 'nonfinite.json'
            with self.assertRaises(ValueError):
                runner.write_new_json(invalid, {'rate': np.nan})
            self.assertFalse(invalid.exists())


if __name__ == '__main__':
    unittest.main()
