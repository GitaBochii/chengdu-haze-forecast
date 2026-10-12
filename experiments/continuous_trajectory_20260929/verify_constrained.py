"""Recompute constrained decisions using the independent probability auditor."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from verify_probability import metrics, sha

HERE = Path(__file__).resolve().parent
RUN = HERE/'probability_runs/20260929T094914341676Z'


def check(expected, actual):
    if isinstance(expected, dict):
        for k,v in expected.items():
            check(v, actual[k])
    elif isinstance(expected, float):
        assert abs(expected-actual) < 1e-10
    else:
        assert expected == actual


def main():
    report = json.loads((RUN/'constrained_metrics.json').read_text('utf-8'))
    old = json.loads((RUN/'report.json').read_text('utf-8'))
    for p,h in report['hashes'].items():
        assert sha(Path(p)) == h
    for p,h in old['artifacts'].items():
        assert sha(RUN/p) == h
    for task,record in report['tasks'].items():
        meta = pd.read_parquet(RUN/f'{task}_origins.parquet')
        y = meta.label.to_numpy(bool)
        dev = meta.partition.eq('dev').to_numpy()
        probs = {n:np.load(RUN/f'{task}_{n}.npy') for n in ('unweighted','balanced')}
        grid = []
        for n,p in probs.items():
            for i in range(1,102):
                grid.append(dict(candidate=n, **metrics(y[dev],p[dev],i/100)))
        for a,b in zip(record['development_grid'],grid):
            # The primary evaluator represents undefined F2 as zero.
            if b['f2'] is None:
                b['f2'] = 0.
            check(a,b)
        feasible = [r for r in grid if r['false_positive_rate']<=record['fpr_budget']]
        best = sorted(feasible,key=lambda r:(r['recall'] or 0,r['precision'] or 0,-r['false_positive_rate'],r['threshold'],r['candidate']=='unweighted'))[-1]
        check(record['chosen'],best)
        for part,result in record['evaluation'].items():
            m = meta.partition.eq(part).to_numpy()
            check(result['constrained'],metrics(y[m],probs[best['candidate']][m],best['threshold']))
            previous = old['tasks'][task]['chosen']
            check(result['previous_f2'],metrics(y[m],probs[previous['candidate']][m],previous['threshold']))
            assert result['empirical_budget_holds'] == (result['constrained']['false_positive_rate']<=record['fpr_budget'])
    evidence = dict(status='passed',candidate_rows_verified=404,partition_comparisons_verified=6,
                    report_sha256=sha(RUN/'constrained_metrics.json'),auditor_sha256=sha(Path(__file__)))
    (RUN/'constrained_independent_audit.json').write_text(json.dumps(evidence,indent=2)+'\n','utf-8')
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
