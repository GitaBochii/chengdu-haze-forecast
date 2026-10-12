"""Frozen equal-probability ensemble and all-seed recovery stability summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensemble_probabilities(arrays):
    if not arrays or arrays[0].ndim != 2:
        raise ValueError('expected nonempty [origin,category] distributions')
    for array in arrays:
        if (array.shape != arrays[0].shape or not np.isfinite(array).all()
                or (array < 0).any() or not np.allclose(array.sum(axis=1), 1, atol=2e-6)):
            raise ValueError('distributions must align and contain normalized nonnegative probabilities')
    probability = np.mean(np.stack(arrays).astype(np.float64), axis=0)
    median = np.minimum((probability.cumsum(axis=1) < .5).sum(axis=1) + 1, probability.shape[1])
    return probability, median


def validate_protocols(reports):
    seeds = [report['configuration']['seed'] for report in reports]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('need at least two distinct seeds, without duplicates')
    configurations = [{key: value for key, value in report['configuration'].items()
                        if key not in ('seed', 'output')} for report in reports]
    if any(configuration != configurations[0] for configuration in configurations):
        raise ValueError('only seed and output may differ')
    for key in ('inputs_before', 'scaler', 'nodes', 'feature_names', 'partitions',
                'coordinates_lon_lat', 'reference_before', 'deployment_before'):
        if any(report[key] != reports[0][key] for report in reports):
            raise ValueError(f'run protocol differs: {key}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(records, probability=None):
    selected = records.loc[records.label_verified_in_scoring_window & records.actual_hours.notna()].copy()
    actual = selected.actual_hours.to_numpy()
    predicted = np.nan_to_num(selected.predicted_hours.to_numpy(), nan=68)
    selected['error'] = np.abs(np.minimum(actual, 68) - np.minimum(predicted, 68))
    recovered, forecast = actual < 68, predicted < 68
    result = {'rows': len(selected), 'events': int(selected.event_id.nunique()),
              'row_mae': float(selected.error.mean()),
              'event_equal_mae': float(selected.groupby('event_id').error.mean().mean()),
              'tail_rows': int((~recovered).sum()),
              'false_recovery_rows': int((~recovered & forecast).sum()),
              'missed_recovery_rows': int((recovered & ~forecast).sum())}
    if probability is not None:
        p = 1 - probability[selected.index, -1]
        result['brier_67h'] = float(np.mean((p - recovered.astype(float))**2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    reports = [json.loads((run / 'report.json').read_text('utf-8')) for run in args.runs]
    validate_protocols(reports)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = {}
    for run, report in zip(args.runs, reports, strict=True):
        for path in run.glob('*'):
            if path.is_file():
                snapshot[str(path)] = sha(path)
        assert sha(run / 'recovery_head_state.pt') == report['checkpoint_sha256']
        assert sha(run / 'selection.json') == report['selection_sha256']
        for directory, mapping in ((root / report['configuration']['data'], report['inputs_before']),
                                    (root / report['configuration']['reference'], report['reference_before']),
                                    (root / 'artifacts/cnemc_2025_deployment_candidate', report['deployment_before'])):
            for filename, expected in mapping.items():
                assert sha(directory / filename) == expected
    result = {'seeds': [r['configuration']['seed'] for r in reports], 'ensemble_rule': 'equal_probability_mean_then_median',
              'source_sha256': snapshot, 'years': {},
              'limitation': 'candidate multi-seed vs fixed seed42 reference; previously explored retrospective years'}
    tail_records = []
    for year in ('2025', '2026'):
        frames = [pd.read_parquet(run / f'recovery_head_{year}_recovery.parquet') for run in args.runs]
        arrays = [np.load(run / f'recovery_head_{year}_distribution.npz') for run in args.runs]
        for frame, array in zip(frames, arrays, strict=True):
            for key in ('timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window'):
                pd.testing.assert_series_equal(frames[0][key], frame[key])
            np.testing.assert_array_equal(pd.DatetimeIndex(frame.timestamp).asi8, array['origin_ns'])
            for key in ('target', 'risk_mask', 'eligible'):
                np.testing.assert_array_equal(arrays[0][key], array[key])
            _, median = ensemble_probabilities([array['probability']])
            np.testing.assert_array_equal(frame.predicted_hours, median)
        probability, median = ensemble_probabilities([array['probability'] for array in arrays])
        ensemble = frames[0].copy()
        ensemble['predicted_hours'] = median
        ensemble['model'] = 'equal_probability_ensemble_42_43_44'
        ensemble['probability_recovered_by_67h'] = 1 - probability[:, -1]
        ensemble.to_parquet(args.output / f'ensemble_{year}_recovery.parquet', index=False)
        np.savez_compressed(args.output / f'ensemble_{year}_distribution.npz',
                            origin_ns=arrays[0]['origin_ns'], probability=probability)
        individual = {str(report['configuration']['seed']): score(frame, array['probability'])
                      for report, frame, array in zip(reports, frames, arrays, strict=True)}
        statistics = {}
        for metric in ('row_mae', 'event_equal_mae', 'brier_67h'):
            values = np.array([row[metric] for row in individual.values()])
            statistics[metric] = {'mean': float(values.mean()), 'sample_std': float(values.std(ddof=1)),
                                  'min': float(values.min()), 'max': float(values.max())}
        ref = root / reports[0]['configuration']['reference']
        baseline = pd.read_parquet(ref / f'pm25_gnn_history_{year}_recovery.parquet')
        aft = pd.read_parquet(ref / f'matched_aft_{year}_recovery.parquet')
        result['years'][year] = {'seeds': individual, 'statistics': statistics,
                                 'ensemble': score(ensemble, probability),
                                 'fixed_history_gnn_reference': score(baseline), 'matched_aft': score(aft)}
        for name, frame in [*zip(individual, frames, strict=True), ('ensemble', ensemble)]:
            tail = frame.loc[frame.actual_hours.ge(68) & frame.label_verified_in_scoring_window].copy()
            tail['seed_or_ensemble'] = name
            tail['year'] = year
            tail_records.append(tail)
        for array in arrays:
            array.close()
    for path, expected in snapshot.items():
        assert sha(Path(path)) == expected
    result['verified'] = True
    result['decision'] = 'research_only_no_deployment'
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.concat(tail_records).to_csv(args.output / 'tail_predictions.csv', index=False)
    print(json.dumps({key: value for key, value in result.items() if key != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
