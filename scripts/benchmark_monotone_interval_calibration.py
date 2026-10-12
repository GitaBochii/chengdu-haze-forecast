"""Historical forward-only nonlinear calibration preflight, two fixed folds."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.monotone_interval_calibration import fit_map,predict_map
from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT=Path(__file__).resolve().parents[1]


def main():
    b=ROOT/'artifacts/task01_recovery_head'
    source=b/'aft_temporal_search_20260918'
    out=b/'monotone_interval_calibration_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths=[source/f'{y}_predictions.parquet' for y in (2020,2021,2022)]
    paths += [Path(__file__).resolve(),ROOT/'src/haze_forecast/monotone_interval_calibration.py',
              ROOT/'monotone_interval_calibration_plan_20260918.md',ROOT/'src/haze_forecast/cohort_scorecard.py',
              ROOT/'src/haze_forecast/paired_interval_score.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    columns=['timestamp','continuity_block','lower_bound_hours','upper_bound_hours','exact_hours','label_available_at','candidate_8']
    data={y:pd.read_parquet(source/f'{y}_predictions.parquet',columns=columns).rename(columns={'candidate_8':'native'}) for y in (2020,2021,2022)}
    out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'input_sha256':hashes,'folds':[[2020,2021],[2020,2021,2022]]},indent=2),encoding='utf-8')
    reports={}
    for year in (2021,2022):
        t=pd.concat([data[y] for y in range(2020,year)],ignore_index=True)
        v=data[year].copy()
        cut=pd.Timestamp(f'{year}-01-01',tz='Asia/Shanghai')
        assert t.timestamp.lt(cut).all() and t.label_available_at.lt(cut).all()
        assert not set(t.continuity_block)&set(v.continuity_block)
        weights=1/t.groupby('continuity_block').timestamp.transform('size')
        model=fit_map(t.native,t.lower_bound_hours,t.upper_bound_hours,weights)
        (out/f'{year}_map.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
        v['calibrated']=predict_map(model,v.native)
        restored=json.loads((out/f'{year}_map.json').read_bytes())
        np.testing.assert_array_equal(predict_map(restored,v.native),v.calibrated)
        scores={n:score_predictions(v,v[n]) for n in ('native','calibrated')}
        paired=paired_error_summary(v,v.native,v.calibrated)
        a,z=scores['calibrated'],scores['native']
        passed=a['exact_row_mae']<z['exact_row_mae'] and a['exact_block_mae']<z['exact_block_mae'] and a['tail24_block_mae']<=z['tail24_block_mae'] and paired['all_row_delta_upper']<=0 and paired['all_block_delta_upper']<=0
        for n in scores:
            exact=v.loc[v.exact_hours.notna()]; e=(exact[n]-exact.exact_hours).abs(); tail=exact.exact_hours>=24
            np.testing.assert_allclose([e.mean(),e.groupby(exact.continuity_block).mean().mean(),e[tail].groupby(exact.loc[tail,'continuity_block']).mean().mean()],
                [scores[n][k] for k in ('exact_row_mae','exact_block_mae','tail24_block_mae')],atol=1e-12)
        t.to_parquet(out/f'{year}_fit.parquet',index=False); v.to_parquet(out/f'{year}_predictions.parquet',index=False)
        reports[str(year)]={'scores':scores,'paired':paired,'passed':passed,'map':model}
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    report={'years':reports,'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({y:{'metrics':{n:[s[k] for k in ('exact_row_mae','exact_block_mae','tail24_block_mae')] for n,s in r['scores'].items()},'passed':r['passed']} for y,r in reports.items()},indent=2))


if __name__=='__main__':
    main()
