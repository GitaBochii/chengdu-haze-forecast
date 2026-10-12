"""Separate event likelihoods; selection is restricted to 2024."""
from datetime import datetime, timezone
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from trajectory import windows, features, SOURCE, SOURCE_SHA, sha, save_booster, load_booster
from event_audit import transition_times, confusion

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE/'runs/20260929T091246660694Z'


def binary_metrics(y, p, threshold):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    if y.shape != p.shape or not len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Nonempty aligned binary probabilities required')
    m = confusion(y, p >= threshold)
    tp, fn, fp, tn = (m[k] for k in ('tp', 'fn', 'fp', 'tn'))
    m.update(f2=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,
             false_positive_rate=fp/(fp+tn) if fp+tn else None,
             brier=float(brier_score_loss(y, p)),
             average_precision=float(average_precision_score(y, p)) if y.any() else None)
    return m


def select_candidate(y, probabilities):
    grid = []
    for name, p in probabilities.items():
        for i in range(1, 20):
            threshold = i/20
            grid.append(dict(candidate=name, threshold=threshold, **binary_metrics(y, p, threshold)))
    best = max(grid, key=lambda x: (x['f2'], x['precision'] or 0, x['threshold'], x['candidate'] == 'unweighted'))
    return best, grid


def main():
    assert sha(SOURCE) == SOURCE_SHA
    paths = [SOURCE, Path(__file__), HERE/'PROBABILITY_PROTOCOL.md', HERE/'trajectory.py', HERE/'event_audit.py', PREVIOUS/'report.json']
    identities = {str(p): sha(p) for p in paths}
    old_report = json.loads((PREVIOUS/'report.json').read_text('utf-8'))
    for name, digest in old_report['artifact_sha256'].items():
        assert sha(PREVIOUS/name) == digest
    w = windows(pd.read_parquet(SOURCE))
    x = features(w['history'], w['timestamps'])
    evaluation = w['partition'] != 'train'
    old_meta = pd.read_parquet(PREVIOUS/'evaluation_origins.parquet')
    assert np.array_equal(old_meta.source_index, w['indices'][evaluation])
    assert np.array_equal(np.load(PREVIOUS/'targets.npy'), w['target'][evaluation])
    output = HERE/'probability_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report = dict(status='running', source_sha256=identities, selection='2024 F2 only', tasks={}, deployed=False)
    on, rec = transition_times(w['target'][:, :, 2])
    for task, eligible, label in [('onset', ~w['polluted'], on < 25), ('recovery', w['polluted'], rec < 20)]:
        train = eligible & (w['partition'] == 'train')
        ev = eligible & evaluation
        meta = pd.DataFrame(dict(source_index=w['indices'][ev], timestamp=w['timestamps'][ev], partition=w['partition'][ev], label=label[ev]))
        meta.to_parquet(output/f'{task}_origins.parquet', index=False)
        probabilities = {}
        for name, weight in [('unweighted', None), ('balanced', 'balanced')]:
            model = lgb.LGBMClassifier(objective='binary', class_weight=weight, n_estimators=300,
                learning_rate=.04, num_leaves=24, min_child_samples=30, random_state=42,
                n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(x[train], label[train])
            p = model.predict_proba(x[ev])[:, 1]
            path = output/f'{task}_{name}.txt'
            save_booster(model.booster_, path)
            np.testing.assert_array_equal(p, load_booster(path).predict(x[ev], num_threads=4))
            np.save(output/f'{task}_{name}.npy', p, allow_pickle=False)
            probabilities[name] = p
        dev = meta.partition.eq('dev').to_numpy()
        chosen, grid = select_candidate(meta.label.to_numpy()[dev], {k: v[dev] for k,v in probabilities.items()})
        record = dict(train_n=int(train.sum()), train_positive=int(label[train].sum()), chosen=chosen, development_grid=grid, metrics={})
        candidates = dict(probabilities)
        for arm in ('all_conditions', 'daily_repeat', 'pollution_only', 'persistence'):
            a, b = transition_times(np.load(PREVIOUS/f'{arm}.npy')[:, :, 2])
            candidates['curve_'+arm] = ((a < 25) if task == 'onset' else (b < 20))[eligible[evaluation]].astype(float)
        for name, p in candidates.items():
            threshold = chosen['threshold'] if name == chosen['candidate'] else .5
            record['metrics'][name] = {}
            for part in ('dev','test','recent_audit'):
                m = meta.partition.eq(part).to_numpy()
                record['metrics'][name][part] = dict(threshold=threshold, **binary_metrics(meta.label.to_numpy()[m], p[m], threshold))
        report['tasks'][task] = record
        print(json.dumps({'task':task, 'chosen':chosen, 'test':record['metrics'][chosen['candidate']]['test']}), flush=True)
    assert all(sha(Path(p)) == h for p,h in identities.items())
    report['artifacts'] = {p.name:sha(p) for p in output.iterdir()}
    report['status'] = 'completed'
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n','utf-8')
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
