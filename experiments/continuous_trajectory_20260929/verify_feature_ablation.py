"""Independent feature-ablation artifact, source alignment and gate audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_probability import labels, sha
from verify_forward import check, score


def diagnostic(y, p, threshold, meta, prevalence):
    bins = []
    for i in range(5):
        m = (p >= i/5) & ((p < (i+1)/5) if i < 4 else p <= 1)
        bins.append(dict(lower=i/5, upper=(i+1)/5, n=int(m.sum()),
                         mean_probability=float(p[m].mean()) if m.any() else None,
                         observed_rate=float(y[m].mean()) if m.any() else None))
    grouped = {}
    for name, levels in {'season': ['DJF', 'MAM', 'JJA', 'SON'],
                         'intensity': ['<=50', '50-100', '100-150', '>150'],
                         'stations': ['<=6', '7-8', '>=9', 'missing']}.items():
        grouped[name] = {}
        for level in levels:
            m = meta[name].eq(level).to_numpy()
            grouped[name][level] = score(y[m], p[m], threshold) if m.any() else None
    return dict(overall=score(y, p, threshold), calibration=bins,
                constant_train_brier=float(np.mean((prevalence-y)**2)), groups=grouped)


def selection(y, probabilities, budget, expected, where):
    grid = [dict(candidate=name, threshold=i/100, **score(y, p, i/100))
            for name, p in probabilities.items() for i in range(1, 102)]
    check(grid, expected['development_grid'], where + '/grid')
    feasible = [row for row in grid if row['false_positive_rate'] <= budget]
    objective = lambda r: (r['recall'] or 0, r['precision'] or 0,
                            -r['false_positive_rate'], r['threshold'], r['candidate'] == 'unweighted')
    best = max(feasible, key=objective)
    check(best, expected['chosen'], where + '/chosen')
    ties = [r['candidate'] for r in feasible if objective(r) == objective(best)]
    return best, {'top_tied_candidates': ties, 'insertion_order_affects_choice': len(set(ties)) > 1,
                  'combined_suffix_unweighted_tie_preference_effective': any(n == 'unweighted' for n in probabilities)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', nargs='?', type=Path, default=Path(__file__).parent / 'feature_runs/20260929T102202428649Z')
    run = parser.parse_args().run.resolve()
    report = json.loads((run/'report.json').read_text('utf-8'))
    assert report['status'] == 'completed' and report['deployed'] is False
    hashes = {}
    for category in ('source_sha256', 'artifact_sha256'):
        for name, digest in report[category].items():
            path = Path(name) if category == 'source_sha256' else run/name
            assert sha(path) == digest, ('hash', path)
            hashes[str(path)] = digest
    base = next(Path(p).parent for p in report['source_sha256'] if Path(p).name == 'report.json')
    baseline = json.loads((base/'report.json').read_text('utf-8'))
    for name, digest in baseline['artifact_sha256'].items():
        assert sha(base/name) == digest
    source = next(Path(p) for p in report['source_sha256'] if Path(p).name == 'city_hourly.parquet')
    raw = pd.read_parquet(source)
    ts = pd.DatetimeIndex(raw.timestamp)
    assert ts.tz is not None and ts.is_unique and not ts.hasnans
    assert (np.diff(ts.asi8) == 3600*10**9).all()
    ts = ts.tz_convert('Asia/Shanghai')
    data = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    valid = np.isfinite(data).all(axis=1) & (data >= 0).all(axis=1)
    indices = np.arange(23, len(raw)-24)[np.lib.stride_tricks.sliding_window_view(valid, 48).all(axis=1)]
    indices = indices[ts[indices+24].year <= 2023]
    aqi = data[indices, 2].astype('float32')
    future = data[indices[:, None]+np.arange(1, 25), 2].astype('float32')
    seasons = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
               6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
    groups = {'season': np.array([seasons[m] for m in ts[indices].month]),
              'intensity': np.array(['<=50' if a <= 50 else '50-100' if a <= 100 else '100-150' if a <= 150 else '>150' for a in aqi]),
              'stations': np.array(['missing' if not np.isfinite(s) else '<=6' if s <= 6 else '7-8' if s <= 8 else '>=9'
                                    for s in raw.aqi_station_count.to_numpy()[indices]])}
    check(report['feature_dimensions'], {'baseline': 76, 'trend': 139, 'chemistry': 323}, 'dimensions')
    assert set(report['folds']) == set(baseline['folds'])
    audit = dict(status='passed', report_sha256=sha(run/'report.json'), verifier_sha256=sha(Path(__file__)),
                 helper_hashes={n: sha(Path(__file__).with_name(n)) for n in ('verify_forward.py', 'verify_probability.py')},
                 hashes_checked=hashes, source_alignment='Independently reconstructed complete 48h windows, partitions, labels and groups',
                 limitations='No model inference or retraining in this audit. Historical overlapping origins are not independent events or a fresh blind holdout.',
                 folds={}, pooled={})
    verified_scores = {}
    for key, record in report['folds'].items():
        year_text, task = key.split('_')
        year = int(year_text)
        population = aqi <= 100 if task == 'onset' else aqi > 100
        y_all = labels(future, task)
        start, end = ts[indices-23], ts[indices+24]
        train = population & (end < pd.Timestamp(f'{year-1}-01-01', tz=ts.tz))
        dev = population & (start >= pd.Timestamp(f'{year-1}-01-01', tz=ts.tz)) & (end < pd.Timestamp(f'{year}-01-01', tz=ts.tz))
        test = population & (start >= pd.Timestamp(f'{year}-01-01', tz=ts.tz)) & (end < pd.Timestamp(f'{year+1}-01-01', tz=ts.tz))
        ev = dev | test
        assert int(train.sum()) == record['train_n'] == baseline['folds'][key]['train_n']
        assert int(y_all[train].sum()) == baseline['folds'][key]['train_positive']
        meta = pd.read_parquet(run/f'{key}_origins.parquet')
        pd.testing.assert_frame_equal(meta, pd.read_parquet(base/f'{key}_origins.parquet'))
        expected = dict(source_index=indices[ev], timestamp=ts[indices[ev]].to_numpy(),
                        partition=np.where(dev[ev], 'dev', 'test'), label=y_all[ev], origin_aqi=aqi[ev],
                        **{n: values[ev] for n, values in groups.items()})
        assert meta.source_index.is_unique and set(meta.columns) == set(expected)
        for name, values in expected.items():
            assert np.array_equal(meta[name].to_numpy(), values), (key, name)
        y = y_all[ev]
        dev_mask = dev[ev]
        test_mask = test[ev]
        prevalence = float(y_all[train].mean())
        budget = .1 if task == 'onset' else .05
        assert record['budget'] == budget
        candidates, fold_audit, fold_scores = {}, {}, {}
        assert list(record['arms']) == ['baseline', 'trend', 'chemistry']
        for arm, arm_record in record['arms'].items():
            probabilities = {}
            for name in ('unweighted', 'balanced'):
                p = np.load(run/f'{key}_{arm}_{name}.npy', allow_pickle=False)
                assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                if arm == 'baseline' and report['seed'] == 42:
                    np.testing.assert_array_equal(p, np.load(base/f'{key}_{name}.npy', allow_pickle=False))
                probabilities[name] = p
                candidates[f'{arm}_{name}'] = p
            chosen, tie = selection(y[dev_mask], {n:p[dev_mask] for n,p in probabilities.items()}, budget, arm_record, f'{key}/{arm}')
            p = probabilities[chosen['candidate']]
            for part, mask in [('dev', dev_mask), ('test', test_mask)]:
                computed = diagnostic(y[mask], p[mask], chosen['threshold'], meta.loc[mask], prevalence)
                check(computed, arm_record['diagnostics'][part], f'{key}/{arm}/{part}')
                if part == 'test':
                    fold_scores[arm] = computed['overall']
            fold_audit[arm] = dict(chosen=chosen, ties=tie)
        selected, tie = selection(y[dev_mask], {n:p[dev_mask] for n,p in candidates.items()}, budget, record['selected'], f'{key}/selected')
        computed = diagnostic(y[test_mask], candidates[selected['candidate']][test_mask], selected['threshold'], meta.loc[test_mask], prevalence)
        check(computed, record['selected']['diagnostics'], f'{key}/selected/test')
        fold_scores['selected'] = computed['overall']
        fold_audit['selected'] = dict(chosen=selected, ties=tie)
        audit['folds'][key] = fold_audit
        verified_scores[key] = fold_scores
    for task in ('onset', 'recovery'):
        scores = [verified_scores[f'{year}_{task}'] for year in (2021, 2022, 2023)]
        totals = {}
        for arm in ('baseline', 'trend', 'chemistry', 'selected'):
            tp, fn, fp, tn = [sum(r[arm][k] for r in scores) for k in ('tp', 'fn', 'fp', 'tn')]
            totals[arm] = dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=tp/(tp+fn), false_positive_rate=fp/(fp+tn),
                               precision=tp/(tp+fp) if tp+fp else None, f1=2*tp/(2*tp+fp+fn))
        for arm, total in totals.items():
            improved = sum(r[arm]['recall'] > r['baseline']['recall'] for r in scores)
            total['years_recall_improved'] = improved
            total['meaningful_gate'] = bool(total['recall'] >= totals['baseline']['recall']+.05
                and total['false_positive_rate'] <= totals['baseline']['false_positive_rate']
                and total['false_positive_rate'] <= (.1 if task == 'onset' else .05) and improved >= 2)
        check(totals, report['pooled'][task], f'pooled/{task}')
        audit['pooled'][task] = totals
    audit['combined_tie_affected_folds'] = [k for k,v in audit['folds'].items() if v['selected']['ties']['insertion_order_affects_choice']]
    audit['verified_grid_rows'] = 6 * (3*202 + 606)
    (run/'independent_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'pooled': audit['pooled'], 'combined_tie_affected_folds': audit['combined_tie_affected_folds']}, indent=2))
    print(f'PASS: {run / "independent_audit.json"}')


if __name__ == '__main__':
    main()
