"""Rebuild supervision coverage without fitting or evaluating any model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.labels import label_city_data
from haze_forecast.recovery_survival import (
    causal_recovery_eligibility,
    recovery_hazard_labels,
    validate_hourly_grid,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(
        'artifacts/task01_recovery_head/future_weather_formal_v1'))
    parser.add_argument('--output', type=Path, default=Path(
        'artifacts/task01_recovery_head/future_weather_supervision_audit_v1'))
    args = parser.parse_args()
    report_path = args.run / 'report.json'
    windows_path = args.run / 'windows.parquet'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status'] == 'completed_train_validation_only'
    city_path = Path(report['configuration']['data']) / 'city_hourly.parquet'
    before = {str(p): sha(p) for p in (report_path, windows_path, city_path)}
    assert before[str(city_path)] == report['inputs_before']['city_hourly.parquet']
    frame = pd.read_parquet(windows_path)
    assert frame.timestamp.is_unique and set(frame.partition) == {'train', 'validation'}
    city = pd.read_parquet(city_path)
    validate_hourly_grid(city.timestamp)
    labels = label_city_data(city)
    pd.testing.assert_series_equal(labels.timestamp, city.timestamp)
    times = pd.DatetimeIndex(city.timestamp)
    positions = times.get_indexer(frame.timestamp)
    assert (positions >= 0).all() and (positions + 72 < len(city)).all()
    values = city.aqi.to_numpy()
    eligible = causal_recovery_eligibility(values, labels.particle_primary.to_numpy())
    np.testing.assert_array_equal(eligible[positions], frame.survival_eligible)
    target, risk = recovery_hazard_labels(values[positions[:, None] + np.arange(1, 73)])
    frame['informative'] = frame.survival_eligible.to_numpy() & risk.any(axis=1)
    frame['exact'] = frame.informative & target.any(axis=1)
    frame['censored'] = frame.informative & ~target.any(axis=1)
    frame['full_horizon_no_recovery'] = frame.censored & risk.all(axis=1)
    frame['partial_prefix_censored'] = frame.censored & ~risk.all(axis=1)
    frame['risk_bins'] = risk.sum(axis=1) * frame.survival_eligible.to_numpy()
    frame['event_id'] = labels.event_id.to_numpy()[positions]
    frame['year'] = frame.timestamp.dt.year
    frame['month'] = frame.timestamp.dt.month
    frame['season'] = frame.month.map({12: 'winter', 1: 'winter', 2: 'winter',
                                      3: 'spring', 4: 'spring', 5: 'spring',
                                      6: 'summer', 7: 'summer', 8: 'summer',
                                      9: 'autumn', 10: 'autumn', 11: 'autumn'})
    for partition, key in [('train', 'training_label_audit'), ('validation', 'validation_label_audit')]:
        subset = frame.loc[frame.partition.eq(partition)]
        expected = report[key]
        assert len(subset) == expected['windows']
        for col, old in [('survival_eligible', 'eligible'), ('informative', 'informative_eligible'),
                         ('exact', 'exact_recoveries'), ('censored', 'censored_informative')]:
            assert int(subset[col].sum()) == expected[old], (partition, col)
    counts = ['survival_eligible', 'informative', 'exact', 'censored',
              'full_horizon_no_recovery', 'partial_prefix_censored', 'risk_bins']
    summary = {'status': 'PASS', 'input_sha256': before, 'coverage': {},
               'limitations': ['censoring_is_not_exact_duration',
                               'event_id_zero_is_not_a_distinct_pollution_event',
                               'risk_bin_count_is_not_gradient_magnitude',
                               'candidate_selection_uses_causal_eligibility_not_future_targets']}
    for partition in ('train', 'validation'):
        subset = frame.loc[frame.partition.eq(partition)]
        available = subset.loc[subset.future_weather_available]
        summary['coverage'][partition] = {
            'total': {col: int(subset[col].sum()) for col in counts},
            'with_weather': {col: int(available[col].sum()) for col in counts},
            'weather_positive_event_ids': sorted(int(v) for v in available.loc[
                available.informative & available.event_id.gt(0), 'event_id'].unique()),
            'weather_informative_unassigned_origins': int((available.informative & available.event_id.eq(0)).sum()),
        }
    assert before == {str(p): sha(p) for p in (report_path, windows_path, city_path)}
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(args.output / 'supervision_origins.parquet', index=False)
    for name, keys in [('year', ['partition', 'year', 'future_weather_available']),
                       ('season', ['partition', 'season', 'future_weather_available']),
                       ('event', ['partition', 'event_id', 'future_weather_available'])]:
        frame.groupby(keys)[counts].sum().to_csv(args.output / f'{name}_coverage.csv')
    candidates = frame.loc[frame.partition.eq('train') & frame.survival_eligible
                           & ~frame.future_weather_available, ['timestamp', 'year']]
    candidates.to_csv(args.output / 'causal_training_candidates.csv', index=False)
    summary['uncollected_causal_training_origins'] = len(candidates)
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
