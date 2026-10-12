"""Synthetic counting and provenance tests; no model fitting or inference."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import direction_audit as audit


def origins():
    cutoff = pd.Timestamp('2025-01-10', tz='Asia/Shanghai')
    times = list(pd.date_range('2024-12-01', periods=6, freq='h', tz='Asia/Shanghai'))
    times += [cutoff+pd.Timedelta(hours=1), cutoff+pd.Timedelta(hours=20)]
    frame = pd.DataFrame(dict(origin_index=np.arange(8), timestamp=times,
        year=[2024]*6+[2025]*2, continuity_block=[1, 2, 3, 4, 5, 6, 9, 10],
        cutoff=[times[0]]*6+[cutoff, cutoff+pd.Timedelta(hours=19)],
        label_available_at=[time+pd.Timedelta(hours=35) for time in times],
        lower_bound_hours=30., upper_bound_hours=30., exact_hours=30., reason='exact'))
    for arm in audit.ARMS:
        frame[arm] = 10.
    cutoffs = pd.DataFrame(dict(evaluation_block=[9], cutoff=[cutoff], evaluation_year=[2025]))
    return frame, cutoffs


def bounds(lower, upper, blocks=None):
    return pd.DataFrame(dict(lower_bound_hours=np.asarray(lower, float), upper_bound_hours=np.asarray(upper, float),
        continuity_block=blocks if blocks is not None else np.arange(len(lower))))


class DirectionRateTests(unittest.TestCase):
    def test_exact_interval_right_censor_and_equal_ties_have_sharp_direction_bounds(self):
        result = audit.rates(bounds([10]*4, [10, 20, np.inf, 10]), [10, 15, 5, 12])
        self.assertEqual(result['row_certain_early'], .25)
        self.assertEqual(result['row_possible_early'], .5)
        self.assertEqual(result['row_certain_late'], .25)
        self.assertEqual(result['row_possible_late'], .5)
        tie = audit.rates(bounds([10], [10]), [10])
        self.assertTrue(all(tie[name] == 0 for name in audit.RATE_COLUMNS))

    def test_process_equal_screen_does_not_follow_large_process_row_weight(self):
        frame = bounds([5]*100+[30]*5, [5]*100+[30]*5, [1]*100+[2, 3, 4, 5, 6])
        result = audit.rates(frame, [10]*105)
        self.assertAlmostEqual(result['row_certain_early'], 5/105)
        self.assertAlmostEqual(result['block_certain_early'], 5/6)
        self.assertEqual(audit.screen(result), 'raise')

    def test_votes_at_two_thirds_pass_but_tie_blocks_stay_in_denominator(self):
        result = audit.rates(bounds([30]*4+[10]*2, [30]*4+[10]*2), [10]*6)
        self.assertEqual(result['past_blocks'], 6)
        self.assertEqual(result['certain_early_majority_blocks'], 4)
        self.assertEqual(audit.screen(result), 'raise')
        ties = audit.rates(bounds([30]*3+[10]*3, [30]*3+[10]*3), [10]*6)
        self.assertEqual(ties['block_certain_early'], .5)
        self.assertEqual(audit.screen(ties), 'none')

    def test_six_processes_are_required_for_either_direction(self):
        early = audit.rates(bounds([30]*5, [30]*5), [10]*5)
        self.assertEqual(audit.screen(early), 'insufficient')
        late = audit.rates(bounds([5]*6, [5]*6), [10]*6)
        self.assertEqual(audit.screen(late), 'lower')

    def test_empty_group_is_explicit_and_invalid_predictions_or_bounds_fail(self):
        result = audit.rates(bounds([], []), [])
        self.assertEqual(result['past_rows'], 0)
        self.assertEqual(result['past_blocks'], 0)
        self.assertTrue(all(result[name] is None for name in audit.RATE_COLUMNS))
        self.assertEqual(audit.screen(result), 'insufficient')
        for frame, values in [(bounds([10], [10]), [np.nan]), (bounds([10], [9]), [10]),
                (bounds([10], [np.nan]), [10]), (bounds([10], [10]), [-1])]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                audit.rates(frame, values)


class PrefixIsolationTests(unittest.TestCase):
    def test_timestamp_and_maturity_equal_cutoff_and_current_block_are_excluded(self):
        frame, cutoffs = origins()
        cutoff = cutoffs.cutoff.iloc[0]
        frame.loc[0, 'label_available_at'] = cutoff
        frame.loc[1, 'timestamp'] = cutoff
        frame.loc[2, 'continuity_block'] = 9
        mask = audit.prefix_mask(frame, cutoff, 9, audit.ARMS[0], 'lt24')
        np.testing.assert_array_equal(mask, [False, False, False, True, True, True, False, False])

    def test_current_and_future_ground_truth_and_predictions_never_feed_same_cutoff_signal(self):
        frame, cutoffs = origins()
        signals, members = audit.build_signals(frame, cutoffs)
        current = audit.current_diagnostics(frame, signals)
        modified = frame.copy()
        modified.loc[6:, ['lower_bound_hours', 'upper_bound_hours', 'exact_hours']] = 5.
        modified.loc[6:, list(audit.ARMS)] = 100.
        changed_signals, changed_members = audit.build_signals(modified, cutoffs)
        pd.testing.assert_frame_equal(signals, changed_signals)
        pd.testing.assert_frame_equal(members, changed_members)
        self.assertFalse(current.equals(audit.current_diagnostics(modified, changed_signals)))

    def test_prediction_not_future_truth_sets_bucket_and_every_empty_combination_is_kept(self):
        frame, cutoffs = origins()
        frame.loc[0, audit.ARMS[0]] = 24.
        signals, members = audit.build_signals(frame, cutoffs)
        self.assertEqual(len(signals), 6)
        row = signals.loc[signals.arm.eq(audit.ARMS[0]) & signals.bucket.eq('ge24')].iloc[0]
        self.assertEqual(row.past_rows, 1)
        self.assertEqual(row.past_blocks, 1)
        self.assertEqual(row.direction, 'insufficient')
        self.assertTrue(members.loc[members.arm.eq(audit.ARMS[0]) & members.bucket.eq('ge24')].origin_index.eq(0).all())
        self.assertEqual(len(audit.current_diagnostics(frame, signals)), 6)

    def test_current_unknown_and_empty_are_not_removed_or_cast_as_agreement(self):
        frame, cutoffs = origins()
        frame.loc[6, ['lower_bound_hours', 'upper_bound_hours', 'exact_hours']] = [5., 30., np.nan]
        frame.loc[6, 'reason'] = 'interval'
        signals, _ = audit.build_signals(frame, cutoffs)
        diagnostics = audit.current_diagnostics(frame, signals)
        nonempty = diagnostics.loc[diagnostics.bucket.eq('lt24')]
        self.assertTrue(nonempty.current_direction.eq('unknown').all())
        self.assertTrue(nonempty.agreement.eq('unknown').all())
        empty = diagnostics.loc[diagnostics.bucket.eq('ge24')]
        self.assertTrue(empty.current_direction.eq('current_empty').all())
        self.assertTrue(empty.agreement.eq('current_empty').all())

    def test_summary_reports_all_combinations_and_independent_empty_and_history_counts(self):
        frame, cutoffs = origins()
        frame = frame.iloc[[6]].copy()
        signals, _ = audit.build_signals(frame, cutoffs)
        current = audit.current_diagnostics(frame, signals)
        leaf = audit.summary_leaf(signals, current)
        self.assertEqual(leaf['combinations'], 6)
        self.assertEqual(leaf['insufficient_history_combinations'], 6)
        self.assertEqual(leaf['empty_prefix_combinations'], 6)
        self.assertEqual(leaf['current_empty_combinations'], 3)
        self.assertEqual(sum(leaf['agreement_counts'].values()), 6)
        self.assertEqual(leaf['signal_coverage'], 0.)
        self.assertEqual(leaf['comparable_combinations'], 0)
        self.assertIsNone(leaf['consistent_fraction_comparable'])

    def test_missing_prediction_fails_instead_of_dropping_an_origin(self):
        frame, cutoffs = origins()
        frame.loc[0, audit.ARMS[0]] = np.nan
        with self.assertRaises(ValueError):
            audit.build_signals(frame, cutoffs)

    def test_all_192_combinations_and_issue_year_survive_empty_current_groups(self):
        frame, _ = origins()
        cutoffs = pd.DataFrame(dict(evaluation_block=np.arange(100, 132),
            cutoff=pd.date_range('2024-12-31', periods=32, freq='h', tz='Asia/Shanghai'),
            evaluation_year=[2025]*32))
        signals, _ = audit.build_signals(frame, cutoffs)
        current = audit.current_diagnostics(frame, signals)
        self.assertEqual(len(signals), 192)
        self.assertEqual(len(current), 192)
        self.assertTrue(signals.evaluation_year.eq(2025).all())
        self.assertTrue(current.current_direction.eq('current_empty').all())
        summary = audit.summarize(signals, current)
        self.assertEqual(summary['all']['combinations'], 192)
        self.assertEqual(summary['by_year']['2025']['combinations'], 192)
        self.assertEqual(summary['by_year']['2024']['combinations'], 0)
        self.assertIsNone(summary['by_year']['2024']['signal_coverage'])
        self.assertEqual(summary['primary']['combinations'], 64)

    def test_current_half_direction_is_unknown_not_a_majority(self):
        frame, cutoffs = origins()
        extra = frame.iloc[[6]].copy()
        extra['origin_index'] = 8
        extra['timestamp'] += pd.Timedelta(hours=1)
        extra[['lower_bound_hours', 'upper_bound_hours', 'exact_hours']] = 5.
        frame = pd.concat([frame, extra], ignore_index=True)
        signals, _ = audit.build_signals(frame, cutoffs)
        current = audit.current_diagnostics(frame, signals)
        nonempty = current.loc[current.bucket.eq('lt24')]
        self.assertTrue(nonempty.row_certain_early.eq(.5).all())
        self.assertTrue(nonempty.row_certain_late.eq(.5).all())
        self.assertTrue(nonempty.current_direction.eq('unknown').all())


class ProvenanceTests(unittest.TestCase):
    def test_parent_chain_must_be_exact_and_independently_passed(self):
        report = {'status': 'pending_independent_verification'}
        verification = dict(status='passed', independent_verification_passed=True,
            report_sha256=audit.PARENT_REPORT_SHA256, verifier_sha256=audit.PARENT_VERIFIER_SHA256)
        audit.validate_parent(report, verification, audit.PARENT_REPORT_SHA256)
        for field, value in [('status', 'failed'), ('independent_verification_passed', False),
                ('report_sha256', 'wrong'), ('verifier_sha256', 'wrong')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate_parent(report, dict(verification, **{field: value}), audit.PARENT_REPORT_SHA256)

    def test_conflicting_locks_are_not_silently_overwritten(self):
        self.assertEqual(audit.merge_hashes({'a': 'x'}, {'a': 'x', 'b': 'y'}), {'a': 'x', 'b': 'y'})
        with self.assertRaises(ValueError):
            audit.merge_hashes({'a': 'x'}, {'a': 'changed'})

    def test_parent_inventory_and_bytes_are_locked_before_any_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            for name in ('report.json', 'verification.json', 'predictions.parquet'):
                (parent / name).write_text(name, encoding='utf-8')
            report = dict(output_sha256={'predictions.parquet': audit.sha(parent / 'predictions.parquet')})
            locked, inventory = audit.parent_artifact_manifest(parent, report)
            manifests = dict(frozen_sha256=locked, parent_run_dir=str(parent), parent_artifact_inventory=inventory)
            audit.assert_locks(manifests)
            (parent / 'predictions.parquet').write_text('changed current labels', encoding='utf-8')
            with self.assertRaises(ValueError):
                audit.assert_locks(manifests)
            (parent / 'unexpected').write_text('extra', encoding='utf-8')
            with self.assertRaises(ValueError):
                audit.parent_artifact_manifest(parent, report)

    def test_unique_utc_output_directory_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            stamp = datetime(2026, 10, 3, 0, 0, 0, 1, tzinfo=timezone.utc)
            created = audit.new_run_directory(Path(directory), stamp)
            self.assertEqual(created.name, '20261003T000000000001Z')
            with self.assertRaises(FileExistsError):
                audit.new_run_directory(Path(directory), stamp)

    def test_protected_directories_require_nonempty_exact_file_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            protected = Path(directory) / 'protected'
            with self.assertRaises(ValueError):
                audit.assert_directory_inventory([protected], {})
            protected.mkdir()
            with self.assertRaises(ValueError):
                audit.assert_directory_inventory([protected], {})
            member = protected / 'member'
            member.write_text('fixed', encoding='utf-8')
            locked = {str(member.resolve()): audit.sha(member)}
            audit.assert_directory_inventory([protected], locked)
            (protected / 'unexpected').write_text('new file', encoding='utf-8')
            with self.assertRaises(ValueError):
                audit.assert_directory_inventory([protected], locked)

    def test_report_writer_refuses_overwrite_or_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            audit.write_json(path, {'status': 'pending'})
            with self.assertRaises(FileExistsError):
                audit.write_json(path, {'status': 'changed'})
            with self.assertRaises(ValueError):
                audit.write_json(Path(directory) / 'invalid.json', {'rate': np.nan})


if __name__ == '__main__':
    unittest.main()
