"""Synthetic tests for the independent fixed-model stress audit."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('independent_stress', HERE/'verify_stress.py')
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def sample(start='2021-01-01', count=100):
    dates = pd.date_range(start, periods=count, freq='h', tz='Asia/Shanghai')
    return pd.DataFrame(dict(timestamp=dates, pm2_5=np.arange(count)+20.,
        pm10=np.arange(count)+30., aqi=np.full(count, 80.), so2=np.full(count, 3.),
        no2=np.full(count, 4.), co=np.full(count, .5), o3_8h=np.full(count, 20.)))


class PopulationTests(unittest.TestCase):
    def test_issue_pm_and_extra_missing_do_not_exclude_issue(self):
        raw = sample()
        raw.loc[45, ['pm2_5', 'pm10']] = np.nan
        raw.loc[:, ['so2', 'no2', 'co', 'o3_8h']] = np.nan
        _, data, reasons = VERIFY.rebuild_year(raw, 2021)
        self.assertIn(45, data['indices'])
        self.assertEqual(reasons[45], 'eligible')
        self.assertEqual(reasons[44], 'future_nonfinite')
        self.assertEqual(reasons[46], 'history_nonfinite')
        self.assertTrue(np.isnan(data['features323'][:, 139:235]).all())

    def test_negative_values_remain_eligible_and76_remains_raw(self):
        raw = sample()
        raw.loc[25, 'pm2_5'] = -3.
        _, data, _ = VERIFY.rebuild_year(raw, 2021)
        self.assertIn(30, data['indices'])
        row = np.flatnonzero(data['indices'] == 30)[0]
        self.assertEqual(data['features76'][row, 19*3], -3.)
        self.assertTrue(np.isnan(data['features323'][row, 19*3]))

    def test_all_year_edges_and_issue_aqi_missing_are_excluded(self):
        raw = sample('2020-12-30', 200)
        _, data, reasons = VERIFY.rebuild_year(raw, 2021)
        self.assertEqual(len(reasons), 152)
        self.assertTrue((reasons[:24] == 'year_edge').all())
        self.assertTrue((reasons[-24:] == 'year_edge').all())
        self.assertEqual(data['indices'][0], 24)
        yr = sample()
        yr.loc[40, 'aqi'] = np.nan
        _, data, reasons = VERIFY.rebuild_year(yr, 2021)
        self.assertNotIn(40, data['indices'])
        self.assertEqual(reasons[40], 'origin_aqi_nonfinite')

    def test_gap_excludes_entire_window_without_resorting(self):
        raw = sample().drop(index=40).reset_index(drop=True)
        _, data, reasons = VERIFY.rebuild_year(raw, 2021)
        self.assertEqual(reasons[45], 'hour_gap')
        self.assertNotIn(45, data['indices'])
        with self.assertRaises(ValueError):
            VERIFY.validate_source(raw)
        with self.assertRaises(ValueError):
            VERIFY.rebuild_year(sample().iloc[::-1], 2021)

    def test_source_timezone_duplicate_and_calendar_checks(self):
        raw = sample()
        VERIFY.validate_source(raw)
        raw.loc[10, 'timestamp'] = raw.loc[9, 'timestamp']
        with self.assertRaises(ValueError):
            VERIFY.validate_source(raw)
        naive = sample()
        naive.timestamp = naive.timestamp.dt.tz_localize(None)
        with self.assertRaises(ValueError):
            VERIFY.validate_source(naive)
        with self.assertRaises(ValueError):
            VERIFY.validate_source(sample(), complete_years=(2021,))

    def test_grouping_uses_unqualified_pollution_start_and_sixth_clean_hour(self):
        raw = sample()
        raw.loc[10:19, 'aqi'] = 150.
        _, data, _ = VERIFY.rebuild_year(raw, 2021)
        groups = VERIFY.assign_groups(pd.DatetimeIndex(raw.timestamp), raw.aqi.to_numpy())
        self.assertEqual(groups[10], groups[24])
        self.assertEqual(groups[10], groups[25])
        self.assertNotEqual(groups[25], groups[26])
        self.assertEqual(data['groups'][0], groups[24])

    def test_grouping_resets_at_year_gap_and_aqi_missing(self):
        dates = pd.date_range('2020-12-31 20:00', periods=12, freq='h', tz='Asia/Shanghai')
        aqi = np.full(12, 150.)
        groups = VERIFY.assign_groups(dates, aqi)
        self.assertNotEqual(groups[3], groups[4])
        aqi[6] = np.nan
        groups = VERIFY.assign_groups(dates, aqi)
        self.assertEqual(groups[6], '')
        self.assertNotEqual(groups[5], groups[7])
        gap = dates.delete(6)
        groups = VERIFY.assign_groups(gap, np.full(11, 150.))
        self.assertNotEqual(groups[5], groups[6])
        clean = VERIFY.assign_groups(dates, np.full(12, 80.))
        anchor = pd.Timestamp('2014-01-01', tz='Asia/Shanghai').value
        self.assertTrue(clean[0].endswith(':c'+str((dates[0].value-anchor)//(72*VERIFY.HOUR))))

    def test_month_clock_not_dayofyear_and_no_issue_or_future_inputs(self):
        raw = sample('2021-02-01')
        _, data, _ = VERIFY.rebuild_year(raw, 2021)
        row = np.flatnonzero(data['indices'] == 40)[0]
        self.assertEqual(data['features76'].dtype, np.float64)
        self.assertEqual(data['features323'].dtype, np.float32)
        self.assertAlmostEqual(data['features76'][row, 74], np.sin(2*np.pi*2/12))
        self.assertAlmostEqual(float(data['features323'][row, 74]), np.sin(2*np.pi*33/366), places=6)
        raw.loc[40:, ['pm2_5', 'pm10', 'aqi', 'so2', 'no2', 'co', 'o3_8h']] = 250.
        _, after, _ = VERIFY.rebuild_year(raw, 2021)
        pos = np.flatnonzero(after['indices'] == 40)[0]
        np.testing.assert_array_equal(data['features76'][row], after['features76'][pos])
        np.testing.assert_array_equal(data['features323'][row], after['features323'][pos])


class MetricTests(unittest.TestCase):
    def test_six_hour_start_and_censoring(self):
        curves = np.full((4, 24), 101.)
        curves[0, :6] = 100.
        curves[1, 18:24] = 90.
        curves[2, 19:] = 90.
        curves[3, :5] = 90.
        np.testing.assert_array_equal(VERIFY.starts(curves), [1, 19, 20, 20])
        with self.assertRaises(ValueError):
            VERIFY.starts(np.full((1, 24), np.nan))

    def test_recovery_fpr_specific_denominator_and_group_equal(self):
        result = VERIFY.recovery_metrics(np.array([1, 20, 20, 20]), np.array([2, 3, 20, 20]),
                                         np.array(['a', 'a', 'b', 'b']))
        self.assertEqual(result['confusion'], dict(tp=1, fn=0, fp=1, tn=2))
        self.assertEqual(result['fpr'], 1/3)
        self.assertEqual(result['mae'], 4.5)
        self.assertEqual(result['group_equal_mae'], 4.5)
        result = VERIFY.recovery_metrics(np.array([1]), np.array([20]), np.array(['a']))
        self.assertIsNone(result['fpr'])
        self.assertEqual(result['confusion']['fp']+result['confusion']['tn'], 0)
        empty = VERIFY.recovery_metrics(np.array([], int), np.array([], int), np.array([], str))
        self.assertIsNone(empty['mae'])

    def test_onset_task_is_not_recovery_confusion(self):
        result = VERIFY.binary_metrics(np.array([True, False, False]), np.array([False, True, False]))
        self.assertEqual(result['confusion'], dict(tp=0, fn=1, fp=1, tn=1))
        self.assertEqual(result['fpr'], .5)

    def test_concentration_equal_groups_and_leads(self):
        target = np.zeros((3, 24))
        pred = np.repeat(np.array([0., 0., 6.])[:, None], 24, axis=1)
        result = VERIFY.concentration_metrics(target, pred, np.array(['a', 'a', 'b']))
        self.assertEqual(result['mae'], 2.)
        self.assertEqual(result['group_equal_mae'], 3.)
        self.assertEqual(len(result['lead_mae']), 24)
        self.assertEqual(result['group_equal_lead_mae'], [3.]*24)

    def test_equal_years_preserves_null_valid_count(self):
        result = VERIFY.year_equal([dict(mae=1., recall=None), dict(mae=3., recall=.5)])
        self.assertEqual(result['mae'], dict(value=2., valid_years=2, total_years=2))
        self.assertEqual(result['recall'], dict(value=.5, valid_years=1, total_years=2))

    def test_fixed_buffer_uses_float32_before_float64_and_no_fit(self):
        class Head:
            def predict(self, x, num_threads):
                return np.repeat(-1. if len(x) == 1 else 100.000003, len(x))
        raw, conservative, covered = VERIFY.predict_fixed([Head() for _ in range(24)], np.zeros((2, 76)), 14.)
        self.assertEqual(raw.dtype, np.float32)
        self.assertEqual(conservative.dtype, np.float64)
        np.testing.assert_array_equal(conservative, raw.astype(np.float64)+14.)
        self.assertTrue(covered.all())
        raw, _, _ = VERIFY.predict_fixed([Head() for _ in range(24)], np.zeros((1, 76)), 14.)
        self.assertTrue((raw == 0).all())
        tree = ast.parse((HERE/'verify_stress.py').read_text('utf-8'))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn('fit', calls)
        self.assertNotIn('train', calls)
        updates = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'update'
                   and isinstance(node.func.value, ast.Name) and node.func.value.id in ('head', 'model')]
        self.assertEqual(updates, [])

    def test_aggregate_coverage_is_per_arm_common_is_intersection(self):
        _, data, _ = VERIFY.rebuild_year(sample(), 2021)
        count = len(data['indices'])
        for name in ('76', '323'):
            data['raw_curve'+name] = np.full((count, 24), 80., np.float32)
            data['conservative_curve'+name] = np.full((count, 24), 110., np.float64)
            data['covered'+name] = np.ones(count, bool)
        data['covered76'][0] = False
        data['covered323'][1] = False
        record = VERIFY.evaluate(data)
        self.assertEqual(record['recovery']['raw76']['all']['n'], count-1)
        self.assertEqual(record['recovery']['raw76']['common']['n'], count-2)
        self.assertEqual(record['clean_onset']['raw76']['negative_n'], count-1)

    def test_missing_class_gate_fails_even_with_mae_improvement(self):
        _, data, _ = VERIFY.rebuild_year(sample(), 2021)
        count = len(data['indices'])
        data['origin_aqi'][:] = 120.
        for name in ('76', '323'):
            data['raw_curve'+name] = np.full((count, 24), 80., np.float32)
            data['conservative_curve'+name] = np.full((count, 24), 110., np.float64)
            data['covered'+name] = np.ones(count, bool)
        annual = {str(y): VERIFY.evaluate(data) for y in VERIFY.YEARS}
        counts = {str(y): VERIFY.population_counts(data, np.full(count+48, 'eligible')) for y in VERIFY.YEARS}
        stability, improvement = VERIFY.gates(annual, VERIFY.evaluate(data), counts)
        self.assertFalse(stability['passed'])
        self.assertFalse(improvement['passed'])

    def test_sha_mutation_detected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path = Path(tmp)/'frozen.txt'
            path.write_text('original', encoding='utf-8')
            frozen = VERIFY.freeze([path])
            VERIFY.assert_unchanged(frozen)
            path.write_text('changed', encoding='utf-8')
            with self.assertRaises(AssertionError):
                VERIFY.assert_unchanged(frozen)

    def test_empty_inference_is_covered_empty_without_predicting(self):
        class Head:
            def predict(self, x, num_threads):
                raise AssertionError('An empty population must not call the predictor')
        raw, conservative, covered = VERIFY.predict_fixed([Head() for _ in range(24)], np.empty((0, 76)), 14.)
        self.assertEqual(raw.shape, (0, 24))
        self.assertEqual(conservative.shape, (0, 24))
        self.assertEqual(covered.shape, (0,))

    def test_bundle_dtype_identity_and_group_membership_tampering_rejected(self):
        _, data, _ = VERIFY.rebuild_year(sample(), 2021)
        VERIFY.verify_bundle(data, data, 2021)
        changed = {key: value.copy() for key, value in data.items()}
        changed['features76'] = changed['features76'].astype(np.float32)
        with self.assertRaises(AssertionError):
            VERIFY.verify_bundle(changed, data, 2021)
        changed = {key: value.copy() for key, value in data.items()}
        changed['timestamps'][0] += VERIFY.HOUR
        with self.assertRaises(AssertionError):
            VERIFY.verify_bundle(changed, data, 2021)
        changed = {key: value.copy() for key, value in data.items()}
        changed['groups'][0] = '2021:changed'
        with self.assertRaises(AssertionError):
            VERIFY.verify_bundle(changed, data, 2021)

    def test_cli_rejects_directory_outside_isolated_runs(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            with self.assertRaises(AssertionError):
                VERIFY.main(Path(tmp))

    def test_cli_never_overwrites_an_existing_audit(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            for artifact in ('audit_manifest.json', 'verification.json'):
                with self.subTest(artifact=artifact):
                    run = root/'runs'/artifact.replace('.json', '')
                    run.mkdir(parents=True)
                    existing = run/artifact
                    existing.write_text('{"preserve": "original evidence"}\n', encoding='utf-8')
                    frozen = VERIFY.freeze([existing])
                    with patch.object(VERIFY, 'HERE', root), self.assertRaisesRegex(AssertionError, 'append-once'):
                        VERIFY.main(run)
                    VERIFY.assert_unchanged(frozen)
                    self.assertEqual(list(run.iterdir()), [existing])

    def test_chemical_coverage_uses_transformed_history_and_negative_diagnostic_raw(self):
        raw = sample()
        raw.loc[25, 'so2'] = -1.
        raw.loc[26, 'co'] = np.nan
        _, data, _ = VERIFY.rebuild_year(raw, 2021)
        coverage = VERIFY.chemical_coverage(data)
        self.assertLess(coverage['all']['so2']['history_fraction'], 1.)
        self.assertLess(coverage['all']['co']['history_fraction'], 1.)
        self.assertEqual(coverage['all']['no2']['history_fraction'], 1.)
        negatives = VERIFY.negative_diagnostics(raw, data)
        self.assertEqual(negatives['raw']['so2']['negative_cells'], 1)
        self.assertEqual(negatives['raw']['co']['nonfinite_cells'], 1)

    def test_rows_and_group_frames_retain_identity_and_every_arm(self):
        _, data, _ = VERIFY.rebuild_year(sample(), 2021)
        count = len(data['indices'])
        for name in ('76', '323'):
            data['raw_curve'+name] = np.full((count, 24), 80., np.float32)
            data['conservative_curve'+name] = np.full((count, 24), 110., np.float64)
            data['covered'+name] = np.ones(count, bool)
        rows = VERIFY.row_frame(data, 2021)
        np.testing.assert_array_equal(rows.source_index, data['indices'])
        self.assertEqual(rows.raw_start76.iloc[0], 1)
        self.assertEqual(rows.conservative_start76.iloc[0], 20)
        self.assertEqual(rows.conservative76_absolute_error.iloc[0], 19)
        groups = VERIFY.group_frame(data, 2021)
        self.assertEqual(groups.n.sum(), count)
        self.assertTrue((groups.raw76_mae == 0).all())


if __name__ == '__main__':
    unittest.main()
