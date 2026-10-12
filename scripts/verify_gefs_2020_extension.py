"""Replay models, rebuild changed cells and independently score interval bounds."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT/'artifacts/task01_recovery_head'
    out = b/'gefs_2020_extension_2023_20260918'
    c = json.loads((out/'protocol.json').read_bytes())
    r = json.loads((out/'report.json').read_bytes())
    for p,h in c['input_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    for p,h in r['output_sha256'].items():
        assert hashlib.sha256((out/p).read_bytes()).hexdigest() == h
    old = pd.read_parquet(b/'gefs_2021_extension_v1/training_rows.parquet')
    new = pd.read_parquet(out/'training.parquet')
    features = pd.read_parquet(b/'gefs_2020_support_audit_20260918/features.parquet').set_index('timestamp')
    expected = old.copy()
    for i,row in old.iterrows():
        if row.timestamp in features.index:
            for n in c['forecast_fields']:
                assert pd.isna(row[n])
                expected.at[i,n] = features.at[row.timestamp,n]
    pd.testing.assert_frame_equal(new, expected)
    infer = pd.read_parquet(b/'gefs_2021_extension_v1/inference_rows.parquet')
    p = pd.read_parquet(out/'predictions.parquet')
    v = pd.read_parquet(out/'scored.parquet')
    assert new.timestamp.lt(pd.Timestamp('2023-01-01',tz='Asia/Shanghai')).all()
    assert new.label_available_at.lt(pd.Timestamp('2023-01-01',tz='Asia/Shanghai')).all()
    assert not set(new.continuity_block)&set(v.continuity_block)
    model = xgb.Booster()
    model.load_model(out/'candidate.ubj')
    np.testing.assert_array_equal(model.predict(xgb.DMatrix(infer[c['features']].astype(float))),p.candidate)
    for label,path in c['baselines'].items():
        np.testing.assert_array_equal(joblib.load(path).predict(infer[c['features']]),p[label])
    groups = v.continuity_block.to_numpy()
    exact = v.exact_hours.notna().to_numpy()
    tail = v.exact_hours.ge(24).to_numpy()
    for label in ('candidate',*c['baselines']):
        np.testing.assert_array_equal(p.set_index('timestamp').loc[v.timestamp,label],v[label])
        errors = np.abs(v[label].to_numpy()-v.exact_hours.to_numpy())
        values = [errors[exact].mean(), np.mean([errors[(groups==g)&exact].mean() for g in set(groups[exact])]),
                  np.mean([errors[(groups==g)&tail].mean() for g in set(groups[tail])])]
        keys = ['exact_row_mae','exact_block_mae','tail24_block_mae']
        np.testing.assert_allclose(values,[r['scores'][label][k] for k in keys],atol=1e-12)
    for label in c['baselines']:
        bounds=[]
        for a,z,lo,hi in zip(v.candidate,v[label],v.lower_bound_hours,v.upper_bound_hours):
            points=[lo,hi]+[float(q) for q in (a,z) if lo<=q<=hi]
            differences=[abs(float(a)-t)-abs(float(z)-t) for t in points]
            bounds.append([min(differences),max(differences)])
        bounds=np.asarray(bounds)
        for i,suffix in enumerate(('lower','upper')):
            vals=[bounds[:,i].mean(), np.mean([bounds[groups==g,i].mean() for g in set(groups)])]
            np.testing.assert_allclose(vals,[r['paired'][label][f'all_row_delta_{suffix}'],
                                            r['paired'][label][f'all_block_delta_{suffix}']],atol=1e-12)
    print('Allowed cells rebuilt; 1557 predictions replayed; 9 main metrics and two paired interval comparisons passed; hashes and isolation passed')


if __name__ == '__main__':
    main()
