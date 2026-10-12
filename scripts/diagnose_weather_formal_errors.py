"""Read-only, paired event diagnosis of the frozen 2023 validation outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(
        'artifacts/task01_recovery_head/future_weather_formal_v1'))
    parser.add_argument('--output', type=Path, default=Path(
        'artifacts/task01_recovery_head/future_weather_error_diagnosis_v1'))
    args = parser.parse_args()
    arms = ('base', 'masked_values', 'weather')
    files = [args.run / arm / 'validation_2023_recovery.parquet' for arm in arms]
    files += [args.run / name for name in ('windows.parquet', 'report.json',
                                          'independent_verification.json')]
    before = {str(path): digest(path) for path in files}
    verified = json.loads(files[-1].read_text(encoding='utf-8'))
    assert verified['verification'] == 'PASS' and verified['checkpoint_replay_verified']
    report = json.loads((args.run / 'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'completed_train_validation_only'
    frames = {arm: pd.read_parquet(files[i]) for i, arm in enumerate(arms)}
    identity = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
    reference = frames['base']
    assert reference.timestamp.is_unique
    keep = reference.label_verified_in_scoring_window & reference.actual_hours.notna()
    paired = reference.loc[keep, identity].copy()
    for arm, frame in frames.items():
        pd.testing.assert_frame_equal(reference[identity], frame[identity])
        predictions = frame.loc[keep, 'predicted_hours'].to_numpy()
        assert np.isfinite(predictions).all() and (predictions >= 1).all()
        paired[arm + '_prediction'] = predictions
        paired[arm + '_error'] = np.abs(np.minimum(paired.actual_hours, 68)
                                       - np.minimum(predictions, 68))
        actual = report['arms'][arm]['validation_2023']
        np.testing.assert_allclose(paired[arm + '_error'].mean(), actual['row_mae'])
        np.testing.assert_allclose(paired.groupby('event_id')[arm + '_error'].mean().mean(),
                                   actual['event_equal_mae'])
    paired['weather_minus_base'] = paired.weather_error - paired.base_error
    paired['weather_minus_control'] = paired.weather_error - paired.masked_values_error
    paired['tail'] = paired.actual_hours.ge(68)
    errors = [arm + '_error' for arm in arms]
    deltas = ['weather_minus_base', 'weather_minus_control']
    grouped = paired.groupby('event_id')
    events = grouped.agg(start=('timestamp', 'min'), rows=('timestamp', 'size'),
                         actual_min=('actual_hours', 'min'), actual_max=('actual_hours', 'max'))
    events = events.join(grouped[errors + deltas].mean())
    for delta in deltas:
        events[delta + '_sum'] = grouped[delta].sum()
        values = events[delta]
        events[delta + '_leave_one_event_out'] = (values.sum() - values) / (len(values) - 1)
    strata = paired.groupby('tail')[errors + deltas].agg(['mean', 'sum', 'count'])
    windows = pd.read_parquet(args.run / 'windows.parquet')
    coverage = windows.groupby(['partition', 'survival_eligible', 'future_weather_available']).size()
    summary = {
        'status': 'completed_descriptive_diagnosis',
        'rows': len(paired), 'events': len(events), 'input_sha256': before,
        'scoring': '2023 selection validation only; both durations capped at 68 hours',
        'comparisons': {},
        'limitations': ['not_independent_test', 'not_causal_attribution',
                        'single_seed', 'two_tail_events',
                        'eligible_coverage_is_not_informative_risk_coverage'],
    }
    for delta in deltas:
        summary['comparisons'][delta] = {
            'row_mean_delta': float(paired[delta].mean()),
            'event_mean_delta': float(events[delta].mean()),
            'row_error_sum_delta': float(paired[delta].sum()),
            'events_better_tied_worse': [int((events[delta] < 0).sum()),
                                       int((events[delta] == 0).sum()),
                                       int((events[delta] > 0).sum())],
            'leave_one_event_out_range': [float(events[delta + '_leave_one_event_out'].min()),
                                          float(events[delta + '_leave_one_event_out'].max())],
        }
    assert before == {str(path): digest(path) for path in files}
    args.output.mkdir(parents=True, exist_ok=False)
    paired.to_csv(args.output / 'paired_rows.csv', index=False)
    events.to_csv(args.output / 'event_errors.csv')
    strata.to_csv(args.output / 'short_tail_errors.csv')
    coverage.rename('windows').to_csv(args.output / 'coverage.csv')
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
