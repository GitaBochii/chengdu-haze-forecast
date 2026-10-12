"""Development-only threshold selection with task-specific empirical FPR limits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from probability import binary_metrics
from trajectory import sha

HERE = Path(__file__).resolve().parent
RUN = HERE/'probability_runs/20260929T094914341676Z'


def choose(y, probabilities, budget):
    if not 0 <= budget <= 1 or not (~np.asarray(y, bool)).any():
        raise ValueError('FPR requires negatives and a valid budget')
    grid = [dict(candidate=name, threshold=i/100, **binary_metrics(y,p,i/100))
            for name,p in probabilities.items() for i in range(1,102)]
    feasible = [r for r in grid if r['false_positive_rate'] <= budget]
    best = max(feasible, key=lambda r:(r['recall'] or 0,r['precision'] or 0,
                -r['false_positive_rate'],r['threshold'],r['candidate']=='unweighted'))
    return best,grid


def main():
    old = json.loads((RUN/'report.json').read_text('utf-8'))
    for name,digest in old['artifacts'].items():
        assert sha(RUN/name) == digest
    dependencies = [Path(__file__), HERE/'PROBABILITY_PROTOCOL.md', HERE/'CONSTRAINED_PROTOCOL.md', HERE/'probability.py', HERE/'event_audit.py', HERE/'trajectory.py',RUN/'report.json']
    hashes = {str(p):sha(p) for p in dependencies}
    out = dict(status='completed', hashes=hashes, tasks={}, deployed=False)
    for task,budget in [('onset',.10),('recovery',.05)]:
        meta = pd.read_parquet(RUN/f'{task}_origins.parquet')
        y = meta.label.to_numpy(bool)
        probabilities = {n:np.load(RUN/f'{task}_{n}.npy') for n in ('unweighted','balanced')}
        dev = meta.partition.eq('dev').to_numpy()
        best,grid = choose(y[dev],{n:p[dev] for n,p in probabilities.items()},budget)
        record = dict(fpr_budget=budget, chosen=best, development_grid=grid,
                      informative_on_dev=best['tp']>0, evaluation={})
        previous = old['tasks'][task]['chosen']
        for part in ('dev','test','recent_audit'):
            m = meta.partition.eq(part).to_numpy()
            new = binary_metrics(y[m],probabilities[best['candidate']][m],best['threshold'])
            old_score = binary_metrics(y[m],probabilities[previous['candidate']][m],previous['threshold'])
            record['evaluation'][part] = dict(constrained=new,previous_f2=old_score,
                empirical_budget_holds=new['false_positive_rate']<=budget,
                has_true_detections=new['tp']>0)
        out['tasks'][task] = record
        print(json.dumps(dict(task=task,chosen=best,evaluation=record['evaluation']),indent=2))
    assert all(sha(Path(p)) == h for p,h in hashes.items())
    (RUN/'constrained_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n','utf-8')


if __name__ == '__main__':
    main()
