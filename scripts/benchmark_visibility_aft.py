"""One fixed saturated-visibility addition, with matched missingness control."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.visibility_features import COLUMNS, visibility_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'visibility_aft_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [b/'gefs_2021_extension_v1/f024_training_rows.parquet',
             b/'gefs_2021_extension_v1/inference_rows.parquet',
             b/'grouped_forecast_aft_2023_20260916/expanded.parquet',
             b/'grouped_forecast_aft_2023_20260916/protocol.json',
             b/'visibility_source_20260918/hourly.parquet',
             b/'reusable_gefs_aft_v1/model.joblib', Path(__file__).resolve(),
             ROOT/'src/haze_forecast/visibility_features.py', ROOT/'src/haze_forecast/altimeter_features.py',
             ROOT/'visibility_aft_plan_20260918.md']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    t,v,scored = [pd.read_parquet(p) for p in paths[:3]]
    old = json.loads(paths[3].read_bytes())
    for p,h in old['input_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    names = old['features']
    hourly = pd.read_parquet(paths[4])
    cut = pd.Timestamp('2023-01-01',tz='Asia/Shanghai')
    assert t.timestamp.lt(cut).all() and t.label_available_at.lt(cut).all()
    assert not set(t.continuity_block)&set(scored.continuity_block)
    for frame in (t,v):
        frame[COLUMNS] = visibility_features(hourly,frame.timestamp).to_numpy()
    np.testing.assert_allclose(visibility_features(hourly.loc[hourly.timestamp<cut],t.timestamp),t[COLUMNS],equal_nan=True)
    baseline = joblib.load(paths[5]).predict(v[names])
    predictions = v[['timestamp',*COLUMNS]].copy()
    out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'input_sha256':hashes,'features':names,'added_features':COLUMNS,
        'params':old['params'],'rounds':500},indent=2),encoding='utf-8')
    scores = {}
    for label in ('native','missingness','visibility'):
        cols = names if label=='native' else names+COLUMNS
        x,xv = t[cols].astype(float).copy(),v[cols].astype(float).copy()
        if label=='missingness':
            x[COLUMNS]=x[COLUMNS].where(x[COLUMNS].isna(),0.)
            xv[COLUMNS]=xv[COLUMNS].where(xv[COLUMNS].isna(),0.)
        matrix=xgb.DMatrix(x)
        matrix.set_float_info('label_lower_bound',t.lower_bound_hours)
        matrix.set_float_info('label_upper_bound',t.upper_bound_hours)
        model=xgb.train(old['params'],matrix,num_boost_round=500)
        p=model.predict(xgb.DMatrix(xv))
        if label=='native':
            np.testing.assert_array_equal(p,baseline)
        model.save_model(out/f'{label}.ubj')
        predictions[label]=p
        scored[label]=pd.Series(p,index=v.timestamp).loc[scored.timestamp].to_numpy()
        scores[label]=score_predictions(scored,scored[label].to_numpy())
        print(label,'completed',flush=True)
    paired={n:paired_error_summary(scored,scored[n],scored.visibility) for n in ('native','missingness')}
    gates={n:all(scores['visibility'][k]<scores[n][k] for k in ('exact_row_mae','exact_block_mae'))
        and scores['visibility']['tail24_block_mae']<=scores[n]['tail24_block_mae']
        and paired[n]['all_row_delta_upper']<=0 and paired[n]['all_block_delta_upper']<=0 for n in paired}
    for file,frame in [('training',t),('inference',v),('predictions',predictions),('scored',scored)]:
        frame.to_parquet(out/f'{file}.parquet',index=False)
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    result={'scores':scores,'paired':paired,'gates':gates,'pass':all(gates.values()),
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
