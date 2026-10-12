"""Synthetic tests only; never invoke the formal experiment or fit real models."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import bridge_runner as run


def measurements(n=90, start='2024-12-30'):
    frame = pd.DataFrame({'timestamp': pd.date_range(start, periods=n, freq='h', tz='Asia/Shanghai')})
    for name in run.MEASUREMENT_COLUMNS:
        frame[name] = 10.
    frame['aqi'] = 150.
    frame['pm2_5'] = 120.
    frame['pm10'] = 180.
    return frame


def cohort(lower, upper, blocks=None):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    n = len(lower)
    return pd.DataFrame(dict(timestamp=pd.date_range('2025-01-01', periods=n, freq='h', tz='Asia/Shanghai'),
        continuity_block=blocks if blocks is not None else np.arange(n),
        lower_bound_hours=lower, upper_bound_hours=upper,
        exact_hours=np.where(lower == upper, lower, np.nan)))


class LabelAndFeatureTests(unittest.TestCase):
    def test_gap_keeps_finite_interval_and_real_calendar_hours(self):
        raw = measurements(n=30)
        raw.loc[3:5, 'aqi'] = np.nan
        raw.loc[6:13, 'aqi'] = 150.
        raw.loc[14:, 'aqi'] = 80.
        bounds = run.canonical_bounds(raw, raw.timestamp.iloc[[0]])
        self.assertEqual(bounds.lower_bound_hours.iloc[0], 14.)
        self.assertEqual(bounds.upper_bound_hours.iloc[0], 14.)
        raw.loc[3:8, 'aqi'] = np.nan
        bounds = run.canonical_bounds(raw, raw.timestamp.iloc[[0]])
        self.assertEqual(bounds.lower_bound_hours.iloc[0], 3.)
        self.assertEqual(bounds.upper_bound_hours.iloc[0], 14.)
        self.assertEqual(bounds.reason.iloc[0], 'interval')
        self.assertEqual(bounds.label_available_at.iloc[0], raw.timestamp.iloc[19])

    def test_no_confirmed_recovery_is_right_censored_until_archive_end(self):
        raw = measurements(n=30)
        bounds = run.canonical_bounds(raw, raw.timestamp.iloc[[0, 29]])
        self.assertTrue(np.isposinf(bounds.upper_bound_hours).all())
        self.assertTrue(bounds.label_available_at.eq(raw.timestamp.iloc[-1]).all())
        self.assertEqual(bounds.reason.tolist(), ['right_censored', 'right_censored'])

    def test_rebuild_missing_hour_keeps_nan_and_true_24_hour_lag(self):
        raw = measurements(n=55)
        raw['pm2_5'] = np.arange(len(raw), dtype=float) + 100
        raw = raw.drop(index=15).reset_index(drop=True)
        names = ['pm2_5', 'pm2_5_lag_24', 'boundary_layer_height_m_is_missing',
            'boundary_layer_height_m_change_3_is_missing']
        full = run.rebuild_grid(raw, names)
        self.assertEqual(len(full), 55)
        self.assertTrue(np.isnan(full.loc[15, 'pm2_5']))
        self.assertEqual(full.loc[40, 'pm2_5_lag_24'], 116.)
        self.assertTrue(np.isnan(full.loc[39, 'pm2_5_lag_24']))

    def test_future_measurements_do_not_change_prior_features_or_state(self):
        raw = measurements(n=90)
        names = ['aqi', 'pm2_5_lag_24', 'pm10_roll_24', 'particle_primary']
        before = run.rebuild_grid(raw, names)
        raw.loc[50:, list(run.MEASUREMENT_COLUMNS)] = 999.
        after = run.rebuild_grid(raw, names)
        pd.testing.assert_frame_equal(before.loc[:49, names], after.loc[:49, names])
        pd.testing.assert_series_equal(before.loc[:49, 'eligible'], after.loc[:49, 'eligible'])

    def test_first_two_polluted_hours_retained_in_evaluation(self):
        raw = measurements(n=90)
        raw.loc[:29, 'aqi'] = 80.
        raw.loc[50:, 'aqi'] = 80.
        names = ['aqi', 'pm2_5_lag_24']
        full = run.rebuild_grid(raw, names)
        old = full.loc[[30, 31, 32], ['timestamp', *names]].copy()
        old['event_key'] = 'old:event'
        old['remaining_hours'] = [20., 19., 18.]
        old['strict_prediction_hours'] = [22., 21., 20.]
        evaluation, candidate, legacy = run.prepare_evaluation(old, full, names)
        self.assertEqual(len(evaluation), 3)
        self.assertEqual(evaluation.causal_eligible.tolist(), [False, False, True])
        self.assertEqual(candidate.shape, (3, 2))
        self.assertEqual(legacy.shape, (3, 2))

    def test_cross_year_cutoff_is_full_block_start_not_event_or_year_start(self):
        raw = measurements(n=90, start='2024-12-30')
        raw.loc[:5, 'aqi'] = 80.
        raw.loc[80:, 'aqi'] = 80.
        full = run.rebuild_grid(raw, ['aqi'])
        starts = run.block_cutoffs(full)
        row = full.iloc[60]
        self.assertEqual(row.timestamp.year, 2025)
        self.assertEqual(starts.loc[row.continuity_block], full.timestamp.iloc[6])
        self.assertEqual(starts.loc[row.continuity_block].year, 2024)

    def test_legacy_population_is_not_inner_joined_to_candidate(self):
        full = run.rebuild_grid(measurements(), ['aqi'])
        old = full.iloc[[24, 25, 26]][['timestamp', 'aqi']].copy()
        old['remaining_hours'] = [30., 29., 28.]
        old['event_key'] = 'old:event'
        population = run.prepare_legacy_population(old, full, ['aqi'])
        self.assertEqual(len(population), 3)
        self.assertIn('legacy_remaining_hours', population)
        self.assertTrue(population.upper_bound_hours.eq(np.inf).all())


class MaturityTests(unittest.TestCase):
    def test_strict_timestamp_and_label_available_cutoff_and_current_block(self):
        cutoff = pd.Timestamp('2025-01-01', tz='Asia/Shanghai')
        origins = pd.DataFrame(dict(
            timestamp=[cutoff-pd.Timedelta(hours=4), cutoff-pd.Timedelta(hours=3),
                cutoff, cutoff-pd.Timedelta(hours=2), cutoff-pd.Timedelta(hours=5)],
            label_available_at=[cutoff-pd.Timedelta(hours=1), cutoff, cutoff-pd.Timedelta(hours=1),
                cutoff-pd.Timedelta(hours=1), cutoff+pd.Timedelta(hours=1)],
            continuity_block=[1, 1, 1, 9, 1]))
        mask = run.mature_mask(origins, cutoff, 9)
        np.testing.assert_array_equal(mask, [True, False, False, False, False])

    def test_all_mature_classes_kept_including_right_censor(self):
        cutoff = pd.Timestamp('2025-01-01', tz='Asia/Shanghai')
        origins = cohort([10, 8, 3], [10, 30, np.inf])
        origins['timestamp'] = cutoff - pd.to_timedelta([20, 19, 18], unit='h')
        origins['label_available_at'] = cutoff - pd.Timedelta(hours=1)
        origins['continuity_block'] = 1
        mask = run.mature_mask(origins, cutoff, 9)
        self.assertTrue(mask.all())
        self.assertEqual(run.label_counts(origins), dict(exact=1, interval=1, right_censored=1))


class ScoreTests(unittest.TestCase):
    def test_full_time_is_not_capped_and_exact_error_diagnostics(self):
        frame = cohort([100, 10], [100, 10], [1, 1])
        score = run.score_arm(frame, np.array([80., 15.]))
        self.assertEqual(score['exact_row_mae'], 12.5)
        self.assertEqual(score['exact_block_mae'], 12.5)
        self.assertEqual(score['tail24_block_mae'], 20.)
        self.assertEqual(score['exact_signed_bias'], -7.5)
        self.assertEqual(score['exact_early_fraction'], .5)

    def test_shared_truth_bounds_not_difference_of_separate_distance(self):
        frame = cohort([4], [20])
        bounds = run.paired_bounds(frame, [5], [15])
        self.assertEqual(bounds.lower.iloc[0], -10.)
        self.assertEqual(bounds.upper.iloc[0], 10.)

    def test_right_censor_paired_upper_is_finite(self):
        frame = cohort([30], [np.inf])
        bounds = run.paired_bounds(frame, [10], [20])
        self.assertEqual(bounds.lower.iloc[0], -10.)
        self.assertEqual(bounds.upper.iloc[0], -10.)

    def test_missing_prediction_is_failure_not_drop(self):
        with self.assertRaises(ValueError):
            run.score_arm(cohort([10, 20], [10, 20]), [10, np.nan])

    def test_yearly_gates_need_twenty_percent_and_nonempty_tail(self):
        frame = cohort([30, 40], [30, 40], [1, 2])
        metrics = run.score_year(frame, dict(archive=np.array([50, 60]),
            legacy_control=np.array([45, 55]), interval_candidate=np.array([35, 45])))
        gates = run.year_gates(metrics)
        self.assertTrue(gates['passed'])
        frame = cohort([3, 4], [3, 4], [1, 2])
        metrics = run.score_year(frame, dict(archive=np.array([5, 6]),
            legacy_control=np.array([5, 6]), interval_candidate=np.array([3, 4])))
        self.assertFalse(run.year_gates(metrics)['passed'])
        self.assertIn('nonempty_exact_block_and_tail', run.year_gates(metrics)['failure_reasons'])


class SafetyTests(unittest.TestCase):
    def test_hash_change_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.txt'
            path.write_text('before', encoding='utf-8')
            manifest = {str(path): run.sha(path)}
            run.assert_hashes(manifest)
            path.write_text('after', encoding='utf-8')
            with self.assertRaises(ValueError):
                run.assert_hashes(manifest)

    def test_run_directory_utc_unique_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            stamp = datetime(2026, 10, 2, 12, 0, 0, 1, tzinfo=timezone.utc)
            path = run.new_run_directory(Path(directory), stamp)
            self.assertEqual(path.name, '20261002T120000000001Z')
            with self.assertRaises(FileExistsError):
                run.new_run_directory(Path(directory), stamp)

    def test_fit_configuration_is_fixed_and_no_selection(self):
        self.assertEqual(run.MODEL_CONFIG, dict(seed=42, num_boost_round=500, nthread=1))
        self.assertEqual(run.ARMS, ('archive', 'legacy_control', 'interval_candidate'))
        self.assertEqual(run.YEARS, (2024, 2025))

    def test_hash_manifest_merge_rejects_conflicting_source_evidence(self):
        self.assertEqual(run.merge_hash_manifests({'same': 'a'}, {'same': 'a', 'other': 'b'}),
            {'same': 'a', 'other': 'b'})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            run.merge_hash_manifests({'same': 'a'}, {'same': 'changed'})

    def test_protected_directories_must_exist_and_be_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Protected directory'):
                run.protected_directory_files([parent / 'missing'])
            empty = parent / 'empty'
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, 'Protected directory'):
                run.protected_directory_files([empty])
            file = parent / 'not_directory'
            file.write_text('x', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Protected directory'):
                run.protected_directory_files([file])
            preserved = empty / 'original'
            preserved.write_text('x', encoding='utf-8')
            self.assertEqual(run.protected_directory_files([empty]), [preserved])


class ModelEvidenceTests(unittest.TestCase):
    def population(self):
        cutoff = pd.Timestamp('2025-01-01', tz='Asia/Shanghai')
        frame = cohort([10, 8, 3], [10, 30, np.inf], [1, 1, 1])
        frame['timestamp'] = cutoff-pd.to_timedelta([20, 19, 18], unit='h')
        frame['label_available_at'] = cutoff-pd.Timedelta(hours=1)
        frame['legacy_remaining_hours'] = [9., 7., 2.]
        frame['population_row'] = [0, 1, 2]
        frame['informative'] = True
        frame['reason'] = ['exact', 'interval', 'right_censored']
        frame['f1'] = [1., np.nan, 3.]
        return frame, cutoff

    def fake_model(self):
        model = MagicMock()
        config = {'learner': {'generic_param': {'seed': '42', 'nthread': '1'},
            'learner_train_param': {'objective': 'survival:aft'}}}
        model.booster.save_config.return_value = json.dumps(config)
        model.booster.num_boosted_rounds.return_value = 500
        model.booster.feature_names = ['f1']
        model.booster.save_model.side_effect = lambda path: Path(path).write_text('{}', encoding='utf-8')
        model.predict.return_value = np.array([12.])
        return model, config

    def test_actual_training_config_captured_before_native_save_and_reload(self):
        population, cutoff = self.population()
        for arm in ('legacy_control', 'interval_candidate'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / 'models').mkdir()
                model, config = self.fake_model()
                restored = MagicMock()
                restored.predict.return_value = np.array([12.])
                restored.save_config.return_value = '{"loaded_generic_defaults":true}'
                with patch.object(run, 'ObservedAFTSurvivalRegressor', return_value=model) as factory, \
                        patch.object(run.xgb, 'Booster', return_value=restored), \
                        patch.object(run.xgb, 'DMatrix', side_effect=lambda frame, **kwargs: frame):
                    values, members, record = run.fit_block(population, ['f1'],
                        pd.DataFrame({'f1': [np.nan]}), 9, cutoff, arm, output)
                self.assertEqual(record['actual_trained_booster_config'], config)
                self.assertEqual(record['num_boosted_rounds'], 500)
                self.assertEqual(record['booster_feature_names'], ['f1'])
                calls = [call[0] for call in model.booster.method_calls]
                self.assertLess(calls.index('save_config'), calls.index('save_model'))
                restored.save_config.assert_not_called()
                factory.assert_called_once_with(**run.MODEL_CONFIG)
                self.assertEqual(len(members), 3)
                if arm == 'legacy_control':
                    model.fit.assert_called_once()
                    model.fit_bounds.assert_not_called()
                    np.testing.assert_array_equal(model.fit.call_args.args[1], [9., 7., 2.])
                else:
                    model.fit_bounds.assert_called_once()
                    model.fit.assert_not_called()
                    np.testing.assert_array_equal(model.fit_bounds.call_args.args[2], [10., 30., np.inf])

    def test_wrong_actual_round_count_fails_before_model_save(self):
        population, cutoff = self.population()
        model, _ = self.fake_model()
        model.booster.num_boosted_rounds.return_value = 499
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run, 'ObservedAFTSurvivalRegressor', return_value=model):
            with self.assertRaisesRegex(ValueError, 'round'):
                run.fit_block(population, ['f1'], pd.DataFrame({'f1': [1.]}), 9,
                    cutoff, 'interval_candidate', Path(directory))
        model.booster.save_model.assert_not_called()


    def test_wrong_actual_feature_order_fails_before_model_save(self):
        population, cutoff = self.population()
        model, _ = self.fake_model()
        model.booster.feature_names = ['wrong']
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run, 'ObservedAFTSurvivalRegressor', return_value=model):
            with self.assertRaisesRegex(ValueError, 'feature'):
                run.fit_block(population, ['f1'], pd.DataFrame({'f1': [1.]}), 9,
                    cutoff, 'interval_candidate', Path(directory))
        model.booster.save_model.assert_not_called()


class HistoricalAuditDriftTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(run, 'resolve_historical_hashes', None)),
            'Exact historical audit-drift resolver is not implemented')
        self.app = str((run.ROOT / 'src/haze_forecast/app.py').resolve())
        self.old = '92bc58a1a818ce6ecaa58d6d8c0d4d01e6fac603eb21b346f2e8b4ad0396e1f7'
        self.current = '7f222647d2a9bc31ee3040b121f987da5b95621fe2913a9c697e175b1568724c'
        self.compute = str((run.ROOT / 'src/haze_forecast/interval_recovery_labels.py').resolve())

    def test_only_approved_exact_app_audit_pair_allowed_and_history_preserved(self):
        historical = {self.app: self.old, self.compute: 'same'}
        current = {self.app: self.current, self.compute: 'same'}
        frozen, drift = run.resolve_historical_hashes(historical, current)
        self.assertEqual(frozen, current)
        self.assertEqual(historical, {self.app: self.old, self.compute: 'same'})
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]['path'], self.app)
        self.assertEqual(drift[0]['historical_sha256'], self.old)
        self.assertEqual(drift[0]['current_sha256'], self.current)
        self.assertTrue(drift[0]['reason'])

    def test_other_historical_computational_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            run.resolve_historical_hashes({self.app: self.old, self.compute: 'same'},
                {self.app: self.current, self.compute: 'changed'})

    def test_unapproved_current_app_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, 'audit'):
            run.resolve_historical_hashes({self.app: self.old}, {self.app: 'unknown'})

    def test_unapproved_historical_app_hash_rejected_even_if_current_matches(self):
        with self.assertRaisesRegex(ValueError, 'audit'):
            run.resolve_historical_hashes({self.app: self.current}, {self.app: self.current})

if __name__ == '__main__':
    unittest.main()
