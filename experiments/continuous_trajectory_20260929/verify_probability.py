"""Independent saved-artifact audit; never imports experiment implementation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(aqi, task):
    if task == 'onset':
        return (aqi > 100).any(axis=1)
    return np.logical_or.reduce([(aqi[:, start:start + 6] <= 100).all(axis=1)
                                 for start in range(19)])


def metrics(y, probability, threshold):
    prediction = probability >= threshold
    tp = int((y & prediction).sum())
    fp = int((~y & prediction).sum())
    fn = int((y & ~prediction).sum())
    tn = int((~y & ~prediction).sum())
    divide = lambda a, b: a / b if b else None
    # Average precision integrates precision only where recall increases,
    # grouping tied scores before evaluating their common threshold.
    order = np.argsort(-probability, kind='stable')
    sorted_y, sorted_p = y[order], probability[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(y) - 1]
    cumulative_tp = np.cumsum(sorted_y)[ends]
    delta_tp = np.diff(np.r_[0, cumulative_tp])
    ap = float(np.sum(delta_tp * cumulative_tp / (ends + 1)) / y.sum()) if y.any() else 0.0
    return {'threshold': float(threshold), 'n': len(y), 'tp': tp, 'fp': fp,
            'fn': fn, 'tn': tn, 'precision': divide(tp, tp + fp),
            'recall': divide(tp, tp + fn), 'f1': divide(2 * tp, 2 * tp + fp + fn),
            'f2': divide(5 * tp, 5 * tp + fp + 4 * fn),
            'false_positive_rate': divide(fp, fp + tn),
            'brier': float(np.mean((probability.astype(float) - y) ** 2)),
            'average_precision': ap}


def compare(actual, expected, where):
    for key, value in actual.items():
        reference = expected[key]
        if value is None or isinstance(value, str):
            assert value == reference, (where, key, value, reference)
        else:
            assert np.isclose(value, reference, atol=1e-12, rtol=1e-10), (where, key, value, reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', nargs='?', type=Path,
                        default=Path(__file__).parent / 'probability_runs/20260929T094914341676Z')
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    hash_results = {}
    for filename, expected in report['source_sha256'].items():
        path = Path(filename)
        actual = sha(path)
        assert actual == expected, ('source hash', filename)
        hash_results[str(path)] = actual
    for filename, expected in report['artifacts'].items():
        actual = sha(run / filename)
        assert actual == expected, ('artifact hash', filename)
        hash_results[filename] = actual
    previous = next(Path(p).parent for p in report['source_sha256'] if Path(p).name == 'report.json')
    previous_report = json.loads((previous / 'report.json').read_text(encoding='utf-8'))
    for filename, expected in previous_report['artifact_sha256'].items():
        assert sha(previous / filename) == expected, ('prior artifact hash', filename)
    meta = pd.read_parquet(previous / 'evaluation_origins.parquet')
    targets = np.load(previous / 'targets.npy', allow_pickle=False)
    assert targets.shape == (len(meta), 24, 3)
    assert previous_report['trajectory_target_order'] == ['pm2_5', 'pm10', 'aqi']
    assert meta.source_index.is_unique
    all_meta = pd.read_parquet(previous / 'all_eligible_origins.parquet')
    source_path = next(Path(p) for p in report['source_sha256'] if p.endswith('city_hourly.parquet'))
    raw = pd.read_parquet(source_path)
    all_indices = all_meta.source_index.to_numpy()
    assert np.array_equal(raw.timestamp.iloc[all_indices].to_numpy(), all_meta.timestamp.to_numpy())
    future = raw.aqi.to_numpy()[all_indices[:, None] + np.arange(1, 25)].astype('float32')
    assert np.array_equal(raw.aqi.to_numpy()[all_indices].astype('float32'), all_meta.origin_aqi.to_numpy())
    assert np.array_equal(future[all_meta.partition != 'train'], targets[:, :, 2])
    audit = {'status': 'passed', 'report_sha256': sha(run / 'report.json'),
             'verifier_sha256': sha(Path(__file__)), 'hashes_checked': hash_results,
             'prior_artifact_hashes_checked': len(previous_report['artifact_sha256']),
             'independence': 'No probability.py, trajectory.py or event_audit.py imports; manual metrics and AP',
             'tasks': {}}
    for task, task_report in report['tasks'].items():
        task_meta = pd.read_parquet(run / f'{task}_origins.parquet')
        population = meta.origin_aqi <= 100 if task == 'onset' else meta.origin_aqi > 100
        expected_meta = meta.loc[population].reset_index(drop=True)
        for key in ('source_index', 'timestamp', 'partition'):
            assert np.array_equal(task_meta[key].to_numpy(), expected_meta[key].to_numpy()), (task, key)
        y = labels(targets[population, :, 2], task)
        assert np.array_equal(y, task_meta.label.to_numpy()), (task, 'labels')
        train_population = all_meta.origin_aqi <= 100 if task == 'onset' else all_meta.origin_aqi > 100
        train_mask = (all_meta.partition == 'train') & train_population
        train_y = labels(future[train_mask], task)
        assert len(train_y) == task_report['train_n']
        assert int(train_y.sum()) == task_report['train_positive']
        train_rate = float(train_y.mean())
        probabilities = {}
        grid = []
        dev = (task_meta.partition == 'dev').to_numpy()
        for candidate in ('unweighted', 'balanced'):
            probability = np.load(run / f'{task}_{candidate}.npy', allow_pickle=False)
            assert probability.shape == y.shape
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            probabilities[candidate] = probability
            for threshold in np.arange(1, 20) / 20:
                row = {'candidate': candidate, **metrics(y[dev], probability[dev], threshold)}
                reference = next(r for r in task_report['development_grid']
                                 if r['candidate'] == candidate and np.isclose(r['threshold'], threshold))
                compare(row, reference, (task, candidate, threshold))
                grid.append(row)
        assert len(task_report['development_grid']) == 38
        chosen = max(grid, key=lambda r: (r['f2'], r['precision'] or 0,
                                         r['threshold'], r['candidate'] == 'unweighted'))
        compare(chosen, task_report['chosen'], (task, 'chosen'))
        for name in ('all_conditions', 'daily_repeat', 'pollution_only', 'persistence'):
            curve = np.load(previous / f'{name}.npy', allow_pickle=False)
            probabilities[f'curve_{name}'] = labels(curve[population, :, 2], task).astype(float)
        comparisons = {}
        for partition in ('dev', 'test', 'recent_audit'):
            keep = (task_meta.partition == partition).to_numpy()
            for candidate, probability in probabilities.items():
                reference = task_report['metrics'][candidate][partition]
                compare(metrics(y[keep], probability[keep], reference['threshold']), reference,
                        (task, candidate, partition))
            selected = metrics(y[keep], probabilities[chosen['candidate']][keep], chosen['threshold'])
            positive_baseline = metrics(y[keep], np.ones(int(keep.sum())), 0.5)
            constant_brier = float(np.mean((train_rate - y[keep]) ** 2))
            comparisons[partition] = {
                'n': int(keep.sum()), 'positive': int(y[keep].sum()),
                'selected': selected, 'always_positive_f2': positive_baseline['f2'],
                'selected_minus_always_positive_f2': selected['f2'] - positive_baseline['f2'],
                'train_prevalence_constant_brier': constant_brier,
                'selected_minus_constant_brier': selected['brier'] - constant_brier,
                'worse_than_always_positive_f2': selected['f2'] < positive_baseline['f2'],
                'worse_than_train_constant_brier': selected['brier'] > constant_brier}
        audit['tasks'][task] = {'labels_verified': len(y), 'grid_rows_verified': len(grid),
                              'training_labels_verified': len(train_y), 'train_positive_rate': train_rate,
                              'chosen': chosen, 'comparisons': comparisons}
    output = run / 'probability_independent_audit.json'
    output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(audit['tasks'], indent=2))
    print(f'PASS: {output}')


if __name__ == '__main__':
    main()
