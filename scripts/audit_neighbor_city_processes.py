"""Process sensitivity and tree usage audit, without fitting or cohort changes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'artifacts/task01_recovery_head/neighbor_city_preflight_20260918'
    out = root/'artifacts/task01_recovery_head/neighbor_city_process_audit_20260918'
    if out.exists():
        raise FileExistsError(out)
    files = [source/'report.json', Path(__file__).resolve()]
    files += [source/f'{y}_predictions.parquet' for y in [2021,2022]]
    files += [source/f'{y}_{a}.ubj' for y in [2021,2022] for a in ['local','regional']]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    previous = json.loads((source/'report.json').read_text(encoding='utf-8'))
    rows = []; results = {}; usage = {}
    for year in [2021,2022]:
        d = pd.read_parquet(source/f'{year}_predictions.parquet')
        exact = d[d.exact_hours.notna()]
        results[str(year)] = {'all_rows':len(d), 'exact_rows':len(exact), 'controls':{}}
        for control in ['native','masked','local']:
            values = []
            for block,g in exact.groupby('continuity_block'):
                delta = abs(g.prediction_regional-g.exact_hours)-abs(g[f'prediction_{control}']-g.exact_hours)
                scalar = [abs(float(p)-float(t))-abs(float(q)-float(t)) for p,q,t in zip(g.prediction_regional,g[f'prediction_{control}'],g.exact_hours)]
                np.testing.assert_allclose(delta.to_numpy(),scalar,rtol=0,atol=1e-12)
                row = {'year':year,'control':control,'block':int(block),'exact_rows':len(g),
                       'start':g.timestamp.min().isoformat(),'end':g.timestamp.max().isoformat(),
                       'delta':float(delta.mean()),
                       'native_or_control_mae':float(abs(g[f'prediction_{control}']-g.exact_hours).mean()),
                       'regional_mae':float(abs(g.prediction_regional-g.exact_hours).mean())}
                rows.append(row); values.append(row)
            delta = np.array([x['delta'] for x in values]); weights = np.array([x['exact_rows'] for x in values])
            expected = previous['paired'][str(year)][control]
            np.testing.assert_allclose([np.average(delta,weights=weights),delta.mean()],
                                       [expected['exact_row_delta'],expected['exact_block_delta']],atol=1e-12)
            leave_block = (delta.sum()-delta)/(len(delta)-1)
            leave_row = ((delta*weights).sum()-delta*weights)/(weights.sum()-weights)
            results[str(year)]['controls'][control] = {
                'blocks':len(delta),'improved':int((delta<0).sum()),'worse':int((delta>0).sum()),
                'ties':int((delta==0).sum()),'median_delta':float(np.median(delta)),
                'row_delta':float(np.average(delta,weights=weights)), 'block_delta':float(delta.mean()),
                'leave_one_process_out_block_negative':int((leave_block<0).sum()),
                'leave_one_process_out_row_negative':int((leave_row<0).sum()),
                'leave_one_process_out_block_range':[float(leave_block.min()),float(leave_block.max())],
                'leave_one_process_out_row_range':[float(leave_row.min()),float(leave_row.max())],
                'largest_improvement':min(values,key=lambda x:x['delta']),
                'largest_regression':max(values,key=lambda x:x['delta'])}
        for arm in ['local','regional']:
            model=xgb.Booster(); model.load_model(source/f'{year}_{arm}.ubj')
            counts=model.get_score(importance_type='weight')
            frame=model.trees_to_dataframe(); splits=frame[frame.Feature.ne('Leaf')]
            assert counts == {k:float(v) for k,v in splits.Feature.value_counts().items()}
            cities={f'city48_{i:02d}' for i in range(3,24)}
            usage[f'{year}_{arm}']={'all_splits':int(sum(counts.values())),
                'neighbor_splits':int(sum(n for k,n in counts.items() if k in cities)),
                'neighbor_features_used':sum(k in cities for k in counts),
                'per_feature_splits':{k:int(v) for k,v in counts.items() if k.startswith('city48_')}}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    out.mkdir()
    pd.DataFrame(rows).to_csv(out/'processes.csv',index=False)
    report={'input_sha256':hashes,'processes':results,'tree_usage':usage,
            'limits':'Exact-label process sensitivity only; no samples removed. Split counts are not causal importance or independent observations.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'processes':results,'tree_usage':usage},indent=2))


if __name__=='__main__':
    main()
