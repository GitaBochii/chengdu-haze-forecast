"""Independent probability-artifact audit of fixed follow-up screens."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_probability import sha
from verify_forward import check
from verify_feature_ablation import diagnostic, selection


HERE = Path(__file__).resolve().parent
BASE = HERE/'forward_runs/20260929T100914338806Z'
DEFAULTS = ['spatial_runs/20260929T102734925552Z', 'recency_runs/20260929T103503095182Z',
            'regularized_runs/20260929T103713504212Z', 'ensemble_runs/20260929T103249126668Z']
CANDIDATES = {'spatial_runs': ['unweighted', 'balanced'],
              'recency_runs': ['half2_unweighted', 'half2_balanced', 'half4_unweighted', 'half4_balanced'],
              'regularized_runs': ['unweighted', 'balanced'],
              'ensemble_runs': ['mean_unweighted', 'median_unweighted', 'mean_balanced', 'median_balanced']}


def pooled(rows):
    tp, fn, fp, tn = [sum(row[k] for row in rows) for k in ('tp', 'fn', 'fp', 'tn')]
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=tp/(tp+fn),
                false_positive_rate=fp/(fp+tn), precision=tp/(tp+fp) if tp+fp else None,
                f1=2*tp/(2*tp+fp+fn))


def verify_hashes(path, checked, visited):
    if path in visited:
        return
    visited.add(path)
    report = json.loads(path.read_text('utf-8'))
    for kind in ('source_sha256', 'hashes', 'artifact_sha256'):
        for name, expected in report.get(kind, {}).items():
            item = path.parent/name if kind == 'artifact_sha256' else Path(name)
            assert sha(item) == expected, ('hash', item)
            checked[str(item)] = expected
            if item.name == 'report.json':
                verify_hashes(item, checked, visited)


def audit_run(run):
    report = json.loads((run/'report.json').read_text('utf-8'))
    assert report['status'] == 'completed' and report['deployed'] is False
    baseline = json.loads((BASE/'report.json').read_text('utf-8'))
    prior_audit = json.loads((BASE/'independent_audit.json').read_text('utf-8'))
    assert prior_audit['status'] == 'passed' and prior_audit['report_sha256'] == sha(BASE/'report.json')
    checked = {}
    verify_hashes(run/'report.json', checked, set())
    verify_hashes(BASE/'report.json', checked, set())
    assert set(report['folds']) == set(baseline['folds'])
    kind = run.parent.name
    names = CANDIDATES[kind]
    audit = dict(status='passed', report_sha256=sha(run/'report.json'), verifier_sha256=sha(Path(__file__)),
                 helper_hashes={n: sha(HERE/n) for n in ('verify_probability.py', 'verify_forward.py', 'verify_feature_ablation.py')},
                 hashes_checked=checked, baseline_independent_audit_sha256=sha(BASE/'independent_audit.json'),
                 scope='Reuses independently source-verified baseline labels/metadata. Recomputes every candidate dev grid, selection, dev/test diagnostics and pooled gate. No model inference or retraining.',
                 limitations='Historical overlapping origins are not independent events. Retrospective screening is not a blind holdout or deployment evidence.',
                 folds={}, pooled={})
    recomputed = {}
    for key, record in report['folds'].items():
        b = baseline['folds'][key]
        meta = pd.read_parquet(BASE/f'{key}_origins.parquet')
        if kind != 'ensemble_runs':
            pd.testing.assert_frame_equal(meta, pd.read_parquet(run/f'{key}_origins.parquet'))
            assert record['train_n'] == b['train_n'] and record['train_positive'] == b['train_positive']
            assert record['budget'] == b['budget']
        budget = .1 if key.endswith('_onset') else .05
        assert b['budget'] == budget
        y = meta.label.to_numpy(bool)
        dev = meta.partition.eq('dev').to_numpy()
        probabilities = {}
        for candidate in names:
            p = np.load(run/f'{key}_{candidate}.npy', allow_pickle=False)
            assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            probabilities[candidate] = p
        if kind == 'ensemble_runs':
            dependencies = {Path(p).parent.parent.name: Path(p).parent for p in report['hashes'] if Path(p).name == 'report.json'}
            for weight in ('unweighted', 'balanced'):
                arrays = [np.load(BASE/f'{key}_{weight}.npy', allow_pickle=False),
                          np.load(dependencies['feature_runs']/f'{key}_chemistry_{weight}.npy', allow_pickle=False),
                          np.load(dependencies['spatial_runs']/f'{key}_{weight}.npy', allow_pickle=False)]
                np.testing.assert_array_equal(np.mean(arrays, axis=0), probabilities['mean_'+weight])
                np.testing.assert_array_equal(np.median(arrays, axis=0), probabilities['median_'+weight])
        chosen, ties = selection(y[dev], {n: p[dev] for n,p in probabilities.items()}, budget, record, key)
        p = probabilities[chosen['candidate']]
        diagnostics = {}
        for part in ('dev', 'test'):
            m = meta.partition.eq(part).to_numpy()
            result = diagnostic(y[m], p[m], chosen['threshold'], meta.loc[m], b['train_positive']/b['train_n'])
            check(result, record['diagnostics'][part], f'{key}/{part}')
            diagnostics[part] = result
        recomputed[key] = diagnostics['test']['overall']
        audit['folds'][key] = dict(chosen=chosen, tie_audit=ties, grid_rows_verified=101*len(names),
                                  test=diagnostics['test'], source_aligned_metadata_rows=len(meta))
    for task in ('onset', 'recovery'):
        keys = [f'{year}_{task}' for year in (2021, 2022, 2023)]
        base = pooled([prior_audit['folds'][key]['test']['overall'] for key in keys])
        new = pooled([recomputed[key] for key in keys])
        years = sum(recomputed[key]['recall'] > prior_audit['folds'][key]['test']['overall']['recall'] for key in keys)
        gate = bool(new['recall'] >= base['recall']+.05 and new['false_positive_rate'] <= base['false_positive_rate']
                    and new['false_positive_rate'] <= (.1 if task == 'onset' else .05) and years >= 2)
        new_name = 'spatial' if kind == 'spatial_runs' else 'new'
        result = dict(baseline=base, **{new_name: new}, years_recall_improved=years, meaningful_gate=gate)
        check(result, report['pooled'][task], f'pooled/{task}')
        audit['pooled'][task] = result
    audit['insertion_order_affected_folds'] = [key for key,value in audit['folds'].items() if value['tie_audit']['insertion_order_affects_choice']]
    output = run/'independent_audit.json'
    output.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(run=str(run), pooled=audit['pooled'], insertion_order_affected_folds=audit['insertion_order_affected_folds'])))
    print(f'PASS: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='*', type=Path)
    args = parser.parse_args()
    for run in args.runs or [HERE/name for name in DEFAULTS]:
        audit_run(run.resolve())


if __name__ == '__main__':
    main()
