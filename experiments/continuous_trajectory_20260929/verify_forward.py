"""Audit forward artifacts directly from source; no experiment implementation imports."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from verify_probability import labels, metrics, sha


def score(y, p, threshold):
    result = metrics(y, p, threshold)
    result.pop('threshold')
    if not y.any():
        result['average_precision'] = None
    if result['f2'] is None:
        result['f2'] = 0.0
    return result


def check(actual, expected, where):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), (where, actual.keys(), expected.keys())
        for key in actual:
            check(actual[key], expected[key], f'{where}/{key}')
    elif isinstance(actual, list):
        assert len(actual) == len(expected), where
        for i, value in enumerate(actual):
            check(value, expected[i], f'{where}/{i}')
    elif actual is None or isinstance(actual, (str, bool)):
        assert actual == expected, (where, actual, expected)
    else:
        assert np.isclose(actual, expected, atol=1e-12, rtol=1e-10), (where, actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', nargs='?', type=Path, default=Path(__file__).parent / 'forward_runs/20260929T100914338806Z')
    run = parser.parse_args().run.resolve()
    report = json.loads((run / 'report.json').read_text('utf-8'))
    assert report['status'] == 'completed' and report['deployed'] is False
    verified_hashes = {}
    for category in ('source_sha256', 'artifact_sha256'):
        for filename, expected in report[category].items():
            path = Path(filename) if category == 'source_sha256' else run / filename
            assert sha(path) == expected, ('hash mismatch', path)
            verified_hashes[str(path)] = expected
    source = next(Path(p) for p in report['source_sha256'] if Path(p).name == 'city_hourly.parquet')
    raw = pd.read_parquet(source)
    ts = pd.DatetimeIndex(raw.timestamp)
    assert ts.tz is not None and ts.is_unique and not ts.hasnans
    assert (np.diff(ts.asi8) == 3600 * 10**9).all()
    ts = ts.tz_convert('Asia/Shanghai')
    observations = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    valid = np.isfinite(observations).all(axis=1) & (observations >= 0).all(axis=1)
    indices = np.arange(23, len(raw) - 24)
    # Independent sliding-window completeness check over all 48 observations.
    complete = np.lib.stride_tricks.sliding_window_view(valid, 48).all(axis=1)
    indices = indices[complete]
    indices = indices[ts[indices + 24].year <= 2023]
    origin_aqi = observations[indices, 2].astype('float32')
    future = observations[indices[:, None] + np.arange(1, 25), 2].astype('float32')
    seasons = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
               6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
    levels = {'season': ['DJF', 'MAM', 'JJA', 'SON'],
              'intensity': ['<=50', '50-100', '100-150', '>150'],
              'stations': ['<=6', '7-8', '>=9', 'missing']}
    groups = {
        'season': np.array([seasons[m] for m in ts[indices].month]),
        'intensity': np.array(['<=50' if a <= 50 else '50-100' if a <= 100 else '100-150' if a <= 150 else '>150' for a in origin_aqi]),
        'stations': np.array(['missing' if not np.isfinite(s) else '<=6' if s <= 6 else '7-8' if s <= 8 else '>=9'
                              for s in raw.aqi_station_count.to_numpy()[indices]])}
    expected_folds = {f'{year}_{task}' for year in (2021, 2022, 2023) for task in ('onset', 'recovery')}
    assert set(report['folds']) == expected_folds
    audit = {'status': 'passed', 'report_sha256': sha(run / 'report.json'),
             'verifier_sha256': sha(Path(__file__)),
             'independent_metrics_sha256': sha(Path(__file__).with_name('verify_probability.py')),
             'hashes_checked': verified_hashes,
             'independence': 'Direct source reconstruction; imports only independent labels/metrics/hash helpers, not forward or its implementation dependencies.',
             'limitations': 'Historical overlapping hourly origins, not new blind holdout or independent events. Subgroup associations are not causal. Models are not retrained by this audit.',
             'folds': {}}
    for key, record in report['folds'].items():
        year_text, task = key.split('_')
        year = int(year_text)
        cutoff = year - 2
        assert record['train_end_year'] == cutoff
        population = origin_aqi <= 100 if task == 'onset' else origin_aqi > 100
        y_all = labels(future, task)
        start, end = ts[indices - 23], ts[indices + 24]
        train = population & (end < pd.Timestamp(f'{cutoff+1}-01-01', tz=ts.tz))
        dev = population & (start >= pd.Timestamp(f'{year-1}-01-01', tz=ts.tz)) & (end < pd.Timestamp(f'{year}-01-01', tz=ts.tz))
        test = population & (start >= pd.Timestamp(f'{year}-01-01', tz=ts.tz)) & (end < pd.Timestamp(f'{year+1}-01-01', tz=ts.tz))
        assert not (train & dev).any() and not (train & test).any() and not (dev & test).any()
        assert train.sum() == record['train_n'] and y_all[train].sum() == record['train_positive']
        ev = dev | test
        meta = pd.read_parquet(run / f'{key}_origins.parquet')
        assert meta.source_index.is_unique
        expected = {'source_index': indices[ev], 'timestamp': ts[indices[ev]].to_numpy(),
                    'partition': np.where(dev[ev], 'dev', 'test'), 'label': y_all[ev],
                    'origin_aqi': origin_aqi[ev], **{name: values[ev] for name, values in groups.items()}}
        assert set(meta.columns) == set(expected)
        for name, values in expected.items():
            assert np.array_equal(meta[name].to_numpy(), values), (key, name)
        y = y_all[ev]
        dev_ev = dev[ev]
        selected_indices = indices[ev]
        history = observations[selected_indices[:, None] + np.arange(-23, 1)].astype('float32')
        ev_ts = ts[selected_indices]
        calendar = np.column_stack((np.sin(2*np.pi*ev_ts.hour/24), np.cos(2*np.pi*ev_ts.hour/24),
                                    np.sin(2*np.pi*ev_ts.dayofyear/366), np.cos(2*np.pi*ev_ts.dayofyear/366)))
        features = np.column_stack((history.reshape(len(history), 72), calendar)).astype('float32')
        probabilities, grid = {}, []
        for candidate in ('unweighted', 'balanced'):
            p = np.load(run / f'{key}_{candidate}.npy', allow_pickle=False)
            assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            model = lgb.Booster(model_str=(run / f'{key}_{candidate}.txt').read_text('utf-8'))
            assert model.num_feature() == 76
            np.testing.assert_array_equal(model.predict(features, num_threads=4), p)
            probabilities[candidate] = p
            for threshold in (i / 100 for i in range(1, 102)):
                grid.append({'candidate': candidate, 'threshold': threshold, **score(y[dev_ev], p[dev_ev], threshold)})
        check(grid, record['development_grid'], f'{key}/grid')
        budget = .1 if task == 'onset' else .05
        assert record['budget'] == budget
        feasible = [row for row in grid if row['false_positive_rate'] <= budget]
        chosen = sorted(feasible, key=lambda r: (r['recall'] or 0, r['precision'] or 0,
                        -r['false_positive_rate'], r['threshold'], r['candidate'] == 'unweighted'))[-1]
        check(chosen, record['chosen'], f'{key}/chosen')
        p = probabilities[chosen['candidate']]
        prevalence = float(y_all[train].mean())
        reconstructed = {}
        for part in ('dev', 'test'):
            mask = expected['partition'] == part
            part_y, part_p = y[mask], p[mask]
            bins = []
            for i in range(5):
                bin_mask = (part_p >= i/5) & (part_p < (i+1)/5 if i < 4 else part_p <= 1)
                bins.append({'lower': i/5, 'upper': (i+1)/5, 'n': int(bin_mask.sum()),
                             'mean_probability': float(part_p[bin_mask].mean()) if bin_mask.any() else None,
                             'observed_rate': float(part_y[bin_mask].mean()) if bin_mask.any() else None})
            grouped = {}
            for name, categories in levels.items():
                grouped[name] = {}
                for level in categories:
                    group_mask = mask & (expected[name] == level)
                    grouped[name][level] = score(y[group_mask], p[group_mask], chosen['threshold']) if group_mask.any() else None
            reconstructed[part] = {'overall': score(part_y, part_p, chosen['threshold']), 'calibration': bins,
                                   'constant_train_brier': float(np.mean((prevalence - part_y)**2)), 'groups': grouped}
            check(reconstructed[part], record['diagnostics'][part], f'{key}/{part}')
        overall = reconstructed['test']['overall']
        audit['folds'][key] = {'train_n': int(train.sum()), 'train_positive': int(y_all[train].sum()),
                              'evaluation_n': len(meta), 'grid_rows_verified': len(grid),
                              'serialized_models_verified': 2, 'all_metadata_labels_groups_verified': True,
                              'all_48h_windows_within_partition': True, 'chosen': chosen,
                              'test': reconstructed['test'],
                              'test_fpr_exceeds_dev_budget': overall['false_positive_rate'] > budget,
                              'test_brier_minus_train_constant': overall['brier'] - reconstructed['test']['constant_train_brier']}
        print(json.dumps({'fold': key, 'test': overall, 'budget': budget}, allow_nan=False))
    output = run / 'independent_audit.json'
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'PASS: {output}')


if __name__ == '__main__':
    main()
