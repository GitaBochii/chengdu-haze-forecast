"""Synthetic factor tests; no formal run or real model fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import factor_runner as run


def population():
    cutoff = pd.Timestamp('2025-01-01', tz='Asia/Shanghai')
    frame = pd.DataFrame(dict(
        population_row=np.arange(5),
        timestamp=cutoff-pd.to_timedelta([20, 19, 18, 17, 16], unit='h'),
        continuity_block=[1, 1, 1, 9, 1],
        lower_bound_hours=[10., 8., 3., 2., 1.],
        upper_bound_hours=[10., 30., np.inf, 2., 1.],
        exact_hours=[10., np.nan, np.nan, 2., 1.],
        informative=True,
        reason=['exact', 'interval', 'right_censored', 'exact', 'exact'],
        label_available_at=[cutoff-pd.Timedelta(hours=1)]*4+[cutoff],
        legacy_remaining_hours=[9., 7., 2., 1., 1.],
        f1=[1., np.nan, 3., 4., 5.]))
    return frame, cutoff


def evaluation():
    return pd.DataFrame(dict(timestamp=pd.date_range('2025-01-01', periods=3, freq='h', tz='Asia/Shanghai'),
        year=2025, continuity_block=[1, 2, 3], event_key=['old:1', 'old:2', 'old:3'],
        lower_bound_hours=[100., 30., 4.], upper_bound_hours=[100., 30., 20.],
        exact_hours=[100., 30., np.nan]))


class GridAndMembershipTests(unittest.TestCase):
    def test_grid_population_preserves_every_legacy_origin_and_bounds_with_nan(self):
        legacy, _ = population()
        grid = pd.DataFrame(dict(timestamp=legacy.timestamp, f1=[11., 12., np.nan, 14., 15.],
            eligible=[False, True, False, True, False]))
        candidate = run.grid_population(legacy, grid, ['f1'])
        self.assertEqual(len(candidate), 5)
        pd.testing.assert_frame_equal(candidate.drop(columns='f1'), legacy.drop(columns='f1'))
        np.testing.assert_array_equal(candidate.f1, [11., 12., np.nan, 14., 15.])
        np.testing.assert_array_equal(legacy.f1, [1., np.nan, 3., 4., 5.])

    def test_grid_origin_missing_is_failure_not_population_drop(self):
        legacy, _ = population()
        with self.assertRaisesRegex(ValueError, 'origin'):
            run.grid_population(legacy, legacy.iloc[:-1], ['f1'])

    def test_both_arms_use_identical_strict_mature_members_and_keep_all_label_classes(self):
        legacy, cutoff = population()
        grid = legacy.copy()
        grid['f1'] = 99.
        for arm, frame in zip(run.NEW_ARMS, (legacy, grid)):
            members = run.mature_members(frame, cutoff, 9, arm, f'{arm}_b9', [0, 1, 2])
            self.assertEqual(members.population_row.tolist(), [0, 1, 2])
            self.assertEqual(members.legacy_remaining_hours.tolist(), [9., 7., 2.])
            self.assertEqual(run.bridge.label_counts(members), dict(exact=1, interval=1, right_censored=1))
            self.assertTrue(members.label_available_at.lt(cutoff).all())

    def test_parent_member_disagreement_fails_closed(self):
        legacy, cutoff = population()
        with self.assertRaisesRegex(ValueError, 'Parent mature'):
            run.mature_members(legacy, cutoff, 9, run.NEW_ARMS[0], 'm', [0, 1])


class ModelTests(unittest.TestCase):
    def model(self):
        model = MagicMock()
        model.booster.save_config.return_value = '{"actual":"before_reload"}'
        model.booster.num_boosted_rounds.return_value = 500
        model.booster.feature_names = ['f1']
        model.booster.save_model.side_effect = lambda path: Path(path).write_text('{}', encoding='utf-8')
        model.predict.return_value = np.array([12.])
        return model

    def test_two_new_arms_only_fit_bounds_fixed_config_and_save_fit_time_evidence(self):
        legacy, cutoff = population()
        for arm in run.NEW_ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / 'models').mkdir()
                model = self.model()
                restored = MagicMock()
                restored.predict.return_value = np.array([12.])
                with patch.object(run.bridge, 'ObservedAFTSurvivalRegressor', return_value=model) as factory, \
                        patch.object(run.bridge.xgb, 'Booster', return_value=restored), \
                        patch.object(run.bridge.xgb, 'DMatrix', side_effect=lambda frame, **kwargs: frame):
                    values, members, record = run.fit_block(legacy, ['f1'], pd.DataFrame({'f1': [np.nan]}),
                        9, cutoff, arm, output, [0, 1, 2])
                factory.assert_called_once_with(seed=42, num_boost_round=500, nthread=1)
                model.fit.assert_not_called()
                model.fit_bounds.assert_called_once()
                np.testing.assert_array_equal(model.fit_bounds.call_args.args[2], [10., 30., np.inf])
                self.assertTrue(np.isnan(model.fit_bounds.call_args.args[0].f1.iloc[1]))
                self.assertEqual(record['actual_trained_booster_config'], {'actual': 'before_reload'})
                self.assertEqual(record['num_boosted_rounds'], 500)
                self.assertEqual(record['booster_feature_names'], ['f1'])
                self.assertEqual(record['arm'], arm)
                self.assertEqual(record['model_id'], f'{arm}_b9')
                self.assertEqual(record['parent_member_model_id'], 'legacy_control_b9')
                self.assertEqual(members.arm.unique().tolist(), [arm])
                restored.save_config.assert_not_called()
                methods = [item[0] for item in model.booster.method_calls]
                self.assertLess(methods.index('save_config'), methods.index('save_model'))

    def test_nonfixed_rounds_or_feature_order_are_rejected_before_save(self):
        legacy, cutoff = population()
        for bad in ('rounds', 'features'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                model = self.model()
                if bad == 'rounds':
                    model.booster.num_boosted_rounds.return_value = 499
                else:
                    model.booster.feature_names = ['wrong']
                with patch.object(run.bridge, 'ObservedAFTSurvivalRegressor', return_value=model):
                    with self.assertRaises(ValueError):
                        run.fit_block(legacy, ['f1'], pd.DataFrame({'f1': [1.]}), 9, cutoff,
                            run.NEW_ARMS[0], Path(directory), [0, 1, 2])
                model.booster.save_model.assert_not_called()

    def test_old_fitting_arm_is_never_retrained(self):
        legacy, cutoff = population()
        with patch.object(run.bridge, 'ObservedAFTSurvivalRegressor') as factory:
            with self.assertRaisesRegex(ValueError, 'new'):
                run.fit_block(legacy, ['f1'], pd.DataFrame({'f1': [1.]}), 9, cutoff,
                    'legacy_control', Path('unused'), [0, 1, 2])
        factory.assert_not_called()


class ScoreTests(unittest.TestCase):
    def predictions(self):
        return {arm: np.array([120., 50., 5.]) for arm in run.ARMS}

    def test_fixed_main_candidate_cannot_be_replaced_by_better_labels_only(self):
        predictions = self.predictions()
        predictions['labels_only'] = np.array([100., 30., 5.])
        predictions['grid_all_legacy_origins'] = np.array([130., 60., 5.])
        score = run.score_cohort(evaluation(), predictions)
        self.assertFalse(run.main_gates(score)['passed'])
        self.assertEqual(run.PRIMARY_CANDIDATE, 'grid_all_legacy_origins')
        self.assertEqual(run.DIAGNOSTIC_ONLY_ARMS, ('labels_only',))
        self.assertEqual(run.bridge.ARMS, ('archive', 'legacy_control', 'interval_candidate'))

    def test_five_uncapped_arm_scores_and_seven_shared_truth_pairs_are_complete(self):
        predictions = self.predictions()
        predictions['grid_all_legacy_origins'] = np.array([110., 40., 15.])
        score = run.score_cohort(evaluation(), predictions)
        self.assertEqual(set(score['arms']), set(run.ARMS))
        self.assertEqual(len(score['paired']), 7)
        self.assertEqual(score['arms']['archive']['exact_row_mae'], 20.)
        self.assertEqual(score['arms'][run.PRIMARY_CANDIDATE]['exact_row_mae'], 10.)
        rows = run.row_scores(evaluation(), predictions)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows.grid_all_legacy_origins_exact_error.iloc[0], 10.)
        self.assertEqual(rows.grid_all_legacy_origins_vs_archive_delta_lower.iloc[2], -10.)
        self.assertEqual(rows.grid_all_legacy_origins_vs_archive_delta_upper.iloc[2], 10.)
        self.assertTrue(run.main_gates(score)['passed'])

    def test_any_of_five_missing_predictions_fails_not_drops(self):
        predictions = self.predictions()
        predictions['interval_candidate'][1] = np.nan
        with self.assertRaises(ValueError):
            run.score_cohort(evaluation(), predictions)


class ChainAndSafetyTests(unittest.TestCase):
    def test_parent_passed_chain_is_required_not_just_accuracy_status(self):
        report = {'status': 'pending_independent_verification'}
        verification = dict(status='passed', independent_verification_passed=True,
            report_sha256=run.PARENT_REPORT_SHA256, verifier_sha256=run.PARENT_VERIFIER_SHA256)
        run.validate_parent(report, verification, run.PARENT_REPORT_SHA256)
        for field, value in [('status', 'failed'), ('independent_verification_passed', False),
                ('report_sha256', 'wrong'), ('verifier_sha256', 'wrong')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.validate_parent(report, dict(verification, **{field: value}), run.PARENT_REPORT_SHA256)

    def test_parent_report_identity_must_be_the_predeclared_sha(self):
        with self.assertRaises(ValueError):
            run.validate_parent({'status': 'pending_independent_verification'},
                dict(status='passed', independent_verification_passed=True,
                    report_sha256='other', verifier_sha256=run.PARENT_VERIFIER_SHA256), 'other')

    def test_parent_lock_includes_report_verification_and_rejects_output_or_inventory_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            for name in ('report.json', 'verification.json', 'predictions.parquet'):
                (parent / name).write_text(name, encoding='utf-8')
            report = dict(output_sha256={'predictions.parquet': run.bridge.sha(parent / 'predictions.parquet')})
            locked, inventory = run.parent_artifact_manifest(parent, report)
            self.assertEqual(len(locked), 3)
            self.assertIn(str((parent / 'verification.json').resolve()), locked)
            run.assert_locks(dict(frozen_sha256=locked, parent_run_dir=str(parent),
                parent_artifact_inventory=inventory))
            (parent / 'unexpected').write_text('new', encoding='utf-8')
            with self.assertRaises(ValueError):
                run.parent_artifact_manifest(parent, report)
            with self.assertRaises(ValueError):
                run.assert_locks(dict(frozen_sha256=locked, parent_run_dir=str(parent),
                    parent_artifact_inventory=inventory))

    def test_unique_utc_run_refuses_overwrite_and_hash_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            stamp = datetime(2026, 10, 2, 16, tzinfo=timezone.utc)
            path = run.bridge.new_run_directory(Path(directory), stamp)
            with self.assertRaises(FileExistsError):
                run.bridge.new_run_directory(Path(directory), stamp)
            original = path / 'protected'
            original.write_text('old', encoding='utf-8')
            locked = {str(original): run.bridge.sha(original)}
            original.write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                run.bridge.assert_hashes(locked)


if __name__ == '__main__':
    unittest.main()
