"""Reload each member, reconstruct sampling weights, and independently score medians."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'process_bagging_2023_20260918'
    c = json.loads((out / 'protocol.json').read_bytes())
    r = json.loads((out / 'report.json').read_bytes())
    for p, h in c['input_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    for p, h in r['output_sha256'].items():
        assert hashlib.sha256((out / p).read_bytes()).hexdigest() == h
    weights = pd.read_parquet(out / 'weights.parquet')
    counts = pd.read_parquet(out / 'multiplicities.parquet')
    v = pd.read_parquet(b / 'gefs_2021_extension_v1/inference_rows.parquet')
    pred = pd.read_parquet(out / 'predictions.parquet')
    scored = pd.read_parquet(out / 'scored.parquet')
    assert pred.timestamp.equals(v.timestamp)
    matrix = xgb.DMatrix(v[c['features']].astype(float))
    for seed in c['seeds']:
        ids = np.array(c['blocks'])
        choices = np.random.default_rng(seed).integers(0, len(ids), size=len(ids))
        multiplicity = np.bincount(choices, minlength=len(ids))
        np.testing.assert_array_equal(multiplicity, counts[f'count_{seed}'])
        mapping = dict(zip(ids, multiplicity))
        raw = np.array([mapping[x] for x in weights.continuity_block], dtype=float)
        np.testing.assert_array_equal(raw * (len(raw)/raw.sum()), weights[f'w_{seed}'])
        for arm in ('control', 'bootstrap'):
            key = f'{arm}_{seed}'
            m = xgb.Booster()
            m.load_model(out / f'{key}.ubj')
            np.testing.assert_array_equal(m.predict(matrix), pred[key])
    for arm in ('control', 'bootstrap'):
        key = arm+'_median'
        independent = np.sort(pred[[f'{arm}_{s}' for s in c['seeds']]].to_numpy(), axis=1)[:, 2]
        np.testing.assert_array_equal(independent, pred[key])
        np.testing.assert_array_equal(pd.Series(independent, index=pred.timestamp).loc[scored.timestamp], scored[key])
    for key in ('control_42', 'control_median', 'bootstrap_median'):
        e = scored.loc[scored.exact_hours.notna()].copy()
        e['error'] = [abs(float(a)-float(y)) for a,y in zip(e[key],e.exact_hours)]
        actual = [e.error.mean(),e.groupby('continuity_block').error.mean().mean(),
                  e.loc[e.exact_hours>=24].groupby('continuity_block').error.mean().mean()]
        expected = [r['scores'][key][k] for k in ('exact_row_mae','exact_block_mae','tail24_block_mae')]
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    for key in r['paired']:
        bounds = []
        for row in scored.itertuples():
            old, new = getattr(row,key), row.bootstrap_median
            lo, hi = row.lower_bound_hours, row.upper_bound_hours
            knots = [lo,hi,np.clip(old,lo,hi),np.clip(new,lo,hi)]
            delta = [abs(new-t)-abs(old-t) for t in knots]
            bounds.append([min(delta),max(delta)])
        row_means = np.mean(bounds,axis=0)
        block_means = pd.DataFrame(bounds).groupby(scored.continuity_block.to_numpy()).mean().mean()
        for unit, values in [('row',row_means),('block',block_means)]:
            np.testing.assert_allclose(values,[r['paired'][key][f'all_{unit}_delta_{side}'] for side in ('lower','upper')],atol=1e-12)
    print('Verified 5 multiplicity/weight vectors, 5190 predictions, 1038 medians, 9 metrics, paired row/block bounds and hashes')


if __name__ == '__main__':
    main()
