"""Historical forward teacher check before attempting privileged-input distillation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT=Path(__file__).resolve().parents[1]


def main():
    b=ROOT/'artifacts/task01_recovery_head'
    out=b/'offline_teacher_preflight_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths=[b/'gefs_2021_extension_v1/f024_training_rows.parquet',
           b/'interval_weather_triplet_v1/masked_training.parquet',
           b/'grouped_forecast_aft_2023_20260916/protocol.json',
           b/'interval_weather_triplet_v1/protocol.json', Path(__file__).resolve(),
           ROOT/'offline_teacher_preflight_plan_20260918.md']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original,teacher=[pd.read_parquet(p) for p in paths[:2]]
    c=json.loads(paths[2].read_bytes()); weather=json.loads(paths[3].read_bytes())
    names=c['features']; fields=weather['replaced_fields']
    pd.testing.assert_frame_equal(original[[n for n in original if n not in fields]],teacher[[n for n in original if n not in fields]])
    pd.testing.assert_frame_equal(original[fields].isna(),teacher[fields].isna())
    out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'input_sha256':hashes,'features':names,'params':c['params'],'rounds':500,'years':[2020,2021,2022]},indent=2),encoding='utf-8')
    results={}
    for year in (2020,2021,2022):
        cut=pd.Timestamp(f'{year}-01-01',tz='Asia/Shanghai'); end=pd.Timestamp(f'{year+1}-01-01',tz='Asia/Shanghai')
        validation=original.timestamp.ge(cut)&original.timestamp.lt(end)&original.label_available_at.lt(end)
        fit=original.timestamp.lt(cut)&original.label_available_at.lt(cut)&~original.continuity_block.isin(original.loc[validation,'continuity_block'])
        scored=original.loc[validation].copy(); scores={}
        original.loc[fit,['timestamp','continuity_block','label_available_at']].to_parquet(out/f'{year}_training_members.parquet',index=False)
        for name,frame in [('original',original),('teacher',teacher)]:
            t=frame.loc[fit]; v=frame.loc[validation]
            matrix=xgb.DMatrix(t[names].astype(float)); matrix.set_float_info('label_lower_bound',t.lower_bound_hours); matrix.set_float_info('label_upper_bound',t.upper_bound_hours)
            model=xgb.train(c['params'],matrix,num_boost_round=500)
            scored[name]=model.predict(xgb.DMatrix(v[names].astype(float)))
            model.save_model(out/f'{year}_{name}.ubj')
            scores[name]=score_predictions(scored,scored[name].to_numpy())
        pair=paired_error_summary(scored,scored.original,scored.teacher)
        passed=all(scores['teacher'][k]<scores['original'][k] for k in ('exact_row_mae','exact_block_mae')) and scores['teacher']['tail24_block_mae']<=scores['original']['tail24_block_mae'] and pair['all_row_delta_upper']<=0 and pair['all_block_delta_upper']<=0
        results[str(year)]={'training_rows':int(fit.sum()),'scores':scores,'paired':pair,'pass':bool(passed)}
        scored.to_parquet(out/f'{year}_scored.parquet',index=False)
        print(year,{n:[s[k] for k in ('exact_row_mae','exact_block_mae','tail24_block_mae')] for n,s in scores.items()},'pass',passed,flush=True)
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    (out/'report.json').write_text(json.dumps({'years':results,'pass':all(v['pass'] for v in results.values()),
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}},indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
