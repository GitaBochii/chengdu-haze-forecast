"""Synthetic regression checks for the frozen cross-year stress runner."""
import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import stress_runner as run


def frame(n=100, start='2021-03-01'):
    ts = pd.date_range(start, periods=n, freq='h', tz='Asia/Shanghai')
    values = np.arange(n, dtype=float)
    result = pd.DataFrame(dict(timestamp=ts, pm2_5=30 + values / 10,
        pm10=60 + values / 10, aqi=80 + values / 10))
    for name in run.CHEMICALS:
        result[name] = 5 + values / 100
    return result


class QualificationTests(unittest.TestCase):
    def test_edges_and_index_are_year_local(self):
        bundle, excluded = run.prepare_year(frame(), 2021)
        np.testing.assert_array_equal(bundle['indices'], np.arange(24, 76))
        self.assertEqual(len(excluded), 48)
        self.assertEqual(set(excluded.reason), {'year_edge'})
        self.assertTrue(all(g.startswith('2021:') for g in bundle['groups']))

    def test_issue_pm_missing_is_not_input_eligibility(self):
        raw = frame()
        raw.loc[50, ['pm2_5', 'pm10']] = np.nan
        bundle, _ = run.prepare_year(raw, 2021)
        self.assertIn(50, bundle['indices'])
        self.assertNotIn(49, bundle['indices'])
        self.assertNotIn(51, bundle['indices'])

    def test_issue_aqi_missing_and_history_future_nonfinite_excluded(self):
        raw = frame()
        raw.loc[50, 'aqi'] = np.nan
        bundle, excluded = run.prepare_year(raw, 2021)
        self.assertNotIn(50, bundle['indices'])
        self.assertIn('origin_aqi_nonfinite', set(excluded.reason))
        self.assertIn('history_nonfinite', set(excluded.reason))
        self.assertIn('future_nonfinite', set(excluded.reason))

    def test_chemical_missing_and_negative_do_not_drop_origins(self):
        raw = frame()
        raw.loc[:, list(run.CHEMICALS)] = np.nan
        raw.loc[30, 'pm2_5'] = -1
        bundle, _ = run.prepare_year(raw, 2021)
        self.assertEqual(len(bundle['indices']), 52)
        self.assertEqual(bundle['features76'].dtype, np.float64)
        self.assertEqual(bundle['features323'].dtype, np.float32)
        self.assertEqual(bundle['features323'].shape, (52, 323))
        self.assertTrue(np.isnan(bundle['features323']).any())
        row = np.flatnonzero(bundle['indices'] == 31)[0]
        self.assertEqual(bundle['features76'][row, 69], -1.)
        self.assertTrue(np.isnan(bundle['features323'][row, 69]))

    def test_bad_source_fails_closed_without_sorting(self):
        for raw in (frame().drop(index=40).reset_index(drop=True),
                    frame().iloc[::-1].reset_index(drop=True),
                    pd.concat([frame(), frame().iloc[:1]], ignore_index=True)):
            with self.subTest(raw=len(raw)), self.assertRaises(ValueError):
                run.prepare_year(raw, 2021)

    def test_naive_source_and_mixed_year_rejected(self):
        raw = frame()
        raw['timestamp'] = raw.timestamp.dt.tz_localize(None)
        with self.assertRaises(ValueError):
            run.prepare_year(raw, 2021)
        with self.assertRaises(ValueError):
            run.prepare_year(frame(start='2020-12-31'), 2021)

    def test_full_calendar_verification(self):
        with self.assertRaises(ValueError):
            run.validate_source(frame(), require_full=True)


class GroupTests(unittest.TestCase):
    def test_six_clean_hours_still_belong_to_active_process(self):
        ts = pd.date_range('2021-01-01', periods=10, freq='h', tz='Asia/Shanghai')
        aqi = np.array([80, 150, 110, 90, 90, 90, 90, 90, 90, 90.])
        groups = run.process_groups(ts, aqi)
        self.assertEqual(len(set(groups[1:9])), 1)
        self.assertNotEqual(groups[8], groups[9])

    def test_gap_missing_aqi_and_year_reset(self):
        ts = pd.DatetimeIndex(['2021-12-31 21:00', '2021-12-31 22:00',
            '2022-01-01 00:00', '2022-01-01 01:00', '2022-01-01 02:00',
            '2022-01-01 03:00'], tz='Asia/Shanghai')
        groups = run.process_groups(ts, np.array([140, 140, 140, np.nan, 140, 80.]))
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[1], groups[2])
        self.assertEqual(groups[3], '2022:invalid')
        self.assertNotEqual(groups[2], groups[4])
        self.assertEqual(groups[4], groups[5])

    def test_groups_use_all_hours_before_qualification(self):
        raw = frame()
        raw.loc[:, 'aqi'] = 80.
        raw.loc[20, 'aqi'] = 140.
        bundle, _ = run.prepare_year(raw, 2021)
        self.assertIn(':p', bundle['groups'][0])
        self.assertIn(':p', bundle['groups'][2])
        self.assertIn(':c', bundle['groups'][3])

    def test_clean_blocks_have_fixed_72_hour_anchor(self):
        ts = pd.date_range('2021-01-01', periods=200, freq='h', tz='Asia/Shanghai')
        groups = run.process_groups(ts, np.full(200, 80.))
        epoch = pd.Timestamp('2014-01-01', tz='Asia/Shanghai').value
        expected = (ts.asi8 - epoch) // (72 * run.HOUR_NS)
        np.testing.assert_array_equal([int(g.split(':c')[1]) for g in groups], expected)


class FeatureAndInferenceTests(unittest.TestCase):
    def test_month_calendar_not_chemistry_dayofyear(self):
        bundle, _ = run.prepare_year(frame(start='2021-03-01'), 2021)
        issue = pd.to_datetime(bundle['timestamps'], utc=True).tz_convert('Asia/Shanghai')
        np.testing.assert_array_equal(bundle['features76'][:, 74], np.sin(2*np.pi*issue.month/12))
        np.testing.assert_array_equal(bundle['features323'][:, 74],
            np.sin(2*np.pi*issue.dayofyear/366).astype(np.float32))
        self.assertFalse(np.array_equal(bundle['features76'][:, 74], bundle['features323'][:, 74]))

    def test_same_legal_history_ignores_issue_and_future_values(self):
        raw = frame()
        before, _ = run.prepare_year(raw, 2021)
        raw.loc[50:, ['pm2_5', 'pm10', 'aqi', *run.CHEMICALS]] = 2000.
        after, _ = run.prepare_year(raw, 2021)
        first = np.flatnonzero(before['indices'] == 50)[0]
        second = np.flatnonzero(after['indices'] == 50)[0]
        np.testing.assert_array_equal(before['features76'][first], after['features76'][second])
        np.testing.assert_array_equal(before['features323'][first], after['features323'][second])
        class Booster:
            def predict(self, x, num_threads):
                return np.nan_to_num(x).sum(axis=1)
        for width in (76, 323):
            a = run.predict_heads(before[f'features{width}'][[first]], [Booster()] * 24)
            b = run.predict_heads(after[f'features{width}'][[second]], [Booster()] * 24)
            np.testing.assert_array_equal(a, b)

    def test_clipping_float32_and_fixed_float64_buffers(self):
        class Booster:
            def predict(self, x, num_threads):
                return np.full(len(x), -3.)
        raw = run.predict_heads(np.ones((3, 76)), [Booster()] * 24)
        self.assertEqual(raw.dtype, np.float32)
        np.testing.assert_array_equal(raw, 0.)
        for width, delta in ((76, 14), (323, 18)):
            conservative = run.fixed_conservative(raw, width)
            self.assertEqual(conservative.dtype, np.float64)
            np.testing.assert_array_equal(conservative, delta)

    def test_no_training_or_selection_calls(self):
        tree = ast.parse(Path(run.__file__).read_text(encoding='utf-8'))
        forbidden = {'fit', 'select_buffer', 'select_theta', 'build_report', 'train'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                self.assertNotIn(name, forbidden)


class MetricTests(unittest.TestCase):
    def test_recovery_start_and_right_censor(self):
        curves = np.full((4, 24), 150.)
        curves[0, :6] = 100
        curves[1, 18:] = 90
        curves[2, 19:] = 90
        np.testing.assert_array_equal(run.recovery(curves), [1, 19, 20, 20])
        curves[0, 0] = np.nan
        with self.assertRaises(ValueError):
            run.recovery(curves)

    def test_empty_denominators_are_null(self):
        metrics = run.recovery_metrics(np.array([1]), np.array([20]), np.array(['g']))
        self.assertIsNone(metrics['fpr'])
        self.assertIsNone(metrics['precision'])
        self.assertEqual(metrics['recall'], 0.)
        empty = run.recovery_metrics(np.array([]), np.array([]), np.array([], str))
        self.assertIsNone(empty['mae'])
        self.assertIsNone(empty['recall'])

    def test_confusion_early_and_group_equal(self):
        metrics = run.recovery_metrics(np.array([5, 10, 20, 20]),
            np.array([3, 20, 1, 20]), np.array(['a', 'a', 'b', 'c']))
        self.assertEqual(metrics['confusion'], dict(tp=1, fn=1, fp=1, tn=1))
        self.assertEqual(metrics['early_count'], 2)
        self.assertEqual(metrics['group_equal_mae'], (6 + 19 + 0) / 3)

    def test_all_population_metrics_and_separate_clean_onset(self):
        bundle, _ = run.prepare_year(frame(), 2021)
        n = len(bundle['indices'])
        for width in (76, 323):
            bundle[f'raw_curve{width}'] = np.full((n, 24), 90., np.float32)
            bundle[f'conservative_curve{width}'] = run.fixed_conservative(bundle[f'raw_curve{width}'], width)
            bundle[f'covered{width}'] = np.ones(n, bool)
        bundle['raw_curve76'][0] = np.nan
        bundle['conservative_curve76'][0] = np.nan
        bundle['covered76'][0] = False
        metrics = run.bundle_metrics(bundle)
        self.assertEqual(metrics['recovery']['raw76']['all']['n'], n-1)
        self.assertEqual(metrics['recovery']['raw323']['all']['n'], n)
        self.assertEqual(metrics['recovery']['raw323']['common']['n'], n-1)
        self.assertIn('confusion', metrics['clean_onset']['raw76'])
        self.assertNotIn('brier', str(metrics).lower())

    def test_empty_population_fails_gates(self):
        bundle, _ = run.prepare_year(frame(), 2021)
        n = len(bundle['indices'])
        for width in (76, 323):
            bundle[f'raw_curve{width}'] = np.full((n, 24), 90., np.float32)
            bundle[f'conservative_curve{width}'] = run.fixed_conservative(bundle[f'raw_curve{width}'], width)
            bundle[f'covered{width}'] = np.ones(n, bool)
        metrics = run.bundle_metrics(bundle)
        counts = run.bundle_counts(bundle, 100, 48)
        stable, meaningful = run.evaluate_gates({'2021': metrics}, metrics, {'2021': counts}, years=[2021])
        self.assertFalse(stable['passed'])
        self.assertFalse(meaningful['passed'])
        self.assertIn('2021:polluted_positive_and_negative', stable['failure_reasons'])

    def test_gate_thresholds_require_all_years_and_all_denominators(self):
        def metrics(mae, recall, fpr):
            return dict(recovery={
                'conservative76': {'polluted': dict(mae=mae, group_equal_mae=mae, recall=recall, fpr=fpr)},
                'conservative323': {'polluted': dict(mae=10., group_equal_mae=10., recall=.4, fpr=.04)}})
        annual = {str(year): metrics(9., .45, .05) for year in run.YEARS}
        counts = {str(year): dict(coverage76=1., coverage323=1., polluted_positive=1,
            polluted_negative=1) for year in run.YEARS}
        stable, meaningful = run.evaluate_gates(annual, metrics(9., .45, .05), counts)
        self.assertTrue(stable['passed'])
        self.assertTrue(meaningful['passed'])
        annual['2023']['recovery']['conservative76']['polluted']['fpr'] = None
        stable, meaningful = run.evaluate_gates(annual, metrics(9., .45, .05), counts)
        self.assertFalse(stable['passed'])
        self.assertFalse(meaningful['passed'])
        self.assertIn('2023:fpr_5pct', stable['failure_reasons'])

    def test_year_equal_uses_nonnull_values_and_reports_valid_years(self):
        annual = {str(year): dict(mae=float(year-2020), fpr=None, lead=[None]) for year in run.YEARS}
        annual['2023']['mae'] = None
        result = run.year_equal_summary(annual)
        self.assertEqual(result['mae'], dict(value=3., valid_years=4, total_years=5))
        self.assertEqual(result['fpr'], dict(value=None, valid_years=0, total_years=5))
        self.assertEqual(result['lead'][0]['valid_years'], 0)

    def test_clean_onset_is_not_recovery_confusion(self):
        curve = np.full((2, 24), 80.)
        curve[0, 10] = 120.
        np.testing.assert_array_equal(run.recovery(curve), [1, 1])
        onset = run.onset_metrics(np.array([True, False]), (curve>100).any(axis=1), np.array(['a', 'b']))
        self.assertEqual(onset['confusion'], dict(tp=1, fn=0, fp=0, tn=1))
        self.assertEqual(onset['recall'], 1.)
        self.assertEqual(onset['fpr'], 0.)

    def test_tables_partition_and_uncertainty_contract(self):
        bundle, _ = run.prepare_year(frame(), 2021)
        n = len(bundle['indices'])
        for width in (76, 323):
            bundle[f'raw_curve{width}'] = np.full((n, 24), 90., np.float32)
            bundle[f'conservative_curve{width}'] = run.fixed_conservative(bundle[f'raw_curve{width}'], width)
            bundle[f'covered{width}'] = np.ones(n, bool)
        eligible, rows, groups = run.tables_for_year(bundle, 2021)
        self.assertEqual(len(eligible), n)
        self.assertEqual(int(groups.n.sum()), n)
        for arm in run.ARMS:
            self.assertIn(run.START_KEYS[arm], rows)
            self.assertIn(f'{arm}_absolute_error', rows)
            self.assertIn(f'{arm}_concentration_mae', groups)
        self.assertEqual(set(rows.year), {2021})
        self.assertEqual(rows.source_index.iloc[0], 24)

    def test_empty_eligible_population_preserves_candidate_hour_accounting(self):
        bundle, excluded = run.prepare_year(frame(n=40), 2021)
        self.assertEqual(len(bundle['indices']), 0)
        self.assertEqual(len(excluded), 40)
        self.assertEqual(bundle['features76'].shape, (0, 76))
        self.assertEqual(bundle['features323'].shape, (0, 323))
        self.assertEqual(bundle['paths'].shape, (0, 24, 3))


if __name__ == '__main__':
    unittest.main()
