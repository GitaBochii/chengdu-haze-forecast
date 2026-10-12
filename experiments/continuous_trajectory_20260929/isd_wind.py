"""Archived ISD weather references; delay is an assumption, not release evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from trajectory import windows,SOURCE,SOURCE_SHA,sha,save_booster,load_booster
from feature_ablation import build_features,summaries,pooled,BASE
from forward import fold_parts,diagnostic
from event_audit import transition_times
from constrained import choose

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from haze_forecast.noaa_isd import prepare_noaa_hourly
from haze_forecast.cnemc_weather import add_noaa_weather


def delayed_weather(grid,timestamps,indices,cutoff,lag):
    ts=pd.DatetimeIndex(timestamps)
    positions=indices[:,None]+np.arange(-23,1)-lag
    safe=positions.clip(0,len(grid)-1)
    origin_year=ts[indices].year.to_numpy()
    source_year=ts.year.to_numpy()[safe]
    origin_part=np.where(origin_year<=cutoff,cutoff,origin_year)
    source_part=np.where(source_year<=cutoff,cutoff,source_year)
    valid=(positions>=0)&(positions<len(grid))&(source_part==origin_part[:,None])
    h=grid[safe].copy()
    h[~valid]=np.nan
    return h


PREVIOUS=HERE/'isd_runs'/'20260929T124655970592Z'

def corrected_grid(frame):
    grid=frame[['temperature_c','relative_humidity_pct','wind_u_ms','wind_v_ms','noaa_wind_speed_ms']].to_numpy(float).copy()
    grid[grid[:,4]==0,2:4]=0
    grid[~np.isfinite(grid)]=np.nan
    return grid


def main():
    assert sha(SOURCE)==SOURCE_SHA
    csvs=[ROOT/f'data/raw/noaa_isd/{y}/56294099999.csv' for y in range(2014,2024)]
    deps=[SOURCE,BASE/'report.json',PREVIOUS/'report.json',PREVIOUS/'parsed_hourly.parquet',*csvs,ROOT/'src/haze_forecast/noaa_isd.py',ROOT/'src/haze_forecast/cnemc_weather.py',*[HERE/n for n in ('isd_wind.py','ISD_WIND_PROTOCOL.md','isd_proxy.py','ISD_PROTOCOL.md','feature_ablation.py','forward.py','trajectory.py','constrained.py','probability.py','event_audit.py')]]
    hashes={str(p):sha(p) for p in deps}
    old=json.loads((BASE/'report.json').read_text('utf-8'))
    for n,h in old['artifact_sha256'].items():
        assert sha(BASE/n)==h
    raw=pd.read_parquet(SOURCE)
    previous=json.loads((PREVIOUS/'report.json').read_text('utf-8'))
    assert sha(PREVIOUS/'parsed_hourly.parquet')==previous['artifact_sha256']['parsed_hourly.parquet']
    hourly=pd.read_parquet(PREVIOUS/'parsed_hourly.parquet')
    weather=add_noaa_weather(raw[['timestamp']],hourly)
    grid=corrected_grid(weather)
    w=windows(raw)
    chemistry=build_features(raw,w['indices'],w['history'],w['timestamps'])['chemistry']
    on,rec=transition_times(w['target'][:,:,2])
    output=HERE/'isd_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    hourly.to_parquet(output/'parsed_hourly.parquet',index=False)
    out=dict(status='completed',scope='Lag0 offline observations; lag72 assumed delayed observations, neither proven historical release-time availability',source_sha256=hashes,arms={},deployed=False)
    for lag in (0,72):
        folds={}
        for cutoff in (2019,2020,2021):
            h=delayed_weather(grid,raw.timestamp,w['indices'],cutoff,lag)
            x=np.column_stack([chemistry,h.reshape(len(h),-1),summaries(h),np.isfinite(h).mean(axis=1)]).astype('float32')
            parts=fold_parts(w['timestamps'],cutoff)
            for task,eligible,label,budget in [('onset',~w['polluted'],on<25,.1),('recovery',w['polluted'],rec<20,.05)]:
                key=f'{cutoff+2}_{task}'
                train=eligible&(parts=='train');ev=eligible&np.isin(parts,['dev','test'])
                meta=pd.read_parquet(BASE/f'{key}_origins.parquet')
                np.testing.assert_array_equal(meta.source_index,w['indices'][ev])
                np.testing.assert_array_equal(meta.label,label[ev])
                meta.to_parquet(output/f'lag{lag}_{key}_origins.parquet',index=False)
                candidates={}
                for name,weight in [('unweighted',None),('balanced','balanced')]:
                    model=lgb.LGBMClassifier(objective='binary',class_weight=weight,n_estimators=300,learning_rate=.04,
                        num_leaves=24,min_child_samples=30,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
                    model.fit(x[train],label[train])
                    p=model.predict_proba(x[ev])[:,1]
                    path=output/f'lag{lag}_{key}_{name}.txt'
                    save_booster(model.booster_,path)
                    np.testing.assert_array_equal(p,load_booster(path).predict(x[ev],num_threads=4))
                    np.save(output/f'lag{lag}_{key}_{name}.npy',p,allow_pickle=False)
                    candidates[name]=p
                dev=meta.partition.eq('dev').to_numpy()
                best,selection=choose(label[ev][dev],{n:p[dev] for n,p in candidates.items()},budget)
                record=dict(train_n=int(train.sum()),train_positive=int(label[train].sum()),budget=budget,chosen=best,development_grid=selection,diagnostics={},coverage={})
                for part in ('dev','test'):
                    m=meta.partition.eq(part).to_numpy()
                    record['diagnostics'][part]=diagnostic(meta.loc[m],candidates[best['candidate']][m],best['threshold'],float(label[train].mean()))
                    finite=np.isfinite(h[ev][m])
                    record['coverage'][part]=dict(observed_fraction_by_channel=finite.mean(axis=(0,1)).tolist(),all_channels_complete_window_fraction=float(finite.all(axis=(1,2)).mean()))
                folds[key]=record
                print(json.dumps(dict(lag=lag,fold=key,test=record['diagnostics']['test']['overall'])),flush=True)
        totals={}
        for task,budget in [('onset',.1),('recovery',.05)]:
            keys=[k for k in folds if k.endswith('_'+task)]
            b=pooled([old['folds'][k]['diagnostics']['test']['overall'] for k in keys])
            new=pooled([folds[k]['diagnostics']['test']['overall'] for k in keys])
            years=sum(folds[k]['diagnostics']['test']['overall']['recall']>old['folds'][k]['diagnostics']['test']['overall']['recall'] for k in keys)
            totals[task]=dict(baseline=b,new=new,years_recall_improved=years,
                meaningful_gate=bool(new['recall']>=b['recall']+.05 and new['false_positive_rate']<=min(budget,b['false_positive_rate']) and years>=2))
        out['arms'][f'lag{lag}']=dict(folds=folds,pooled=totals)
        print(json.dumps(dict(lag=lag,pooled=totals),indent=2),flush=True)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    out['artifact_sha256']={p.name:sha(p) for p in output.iterdir()}
    (output/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n','utf-8')
    print(str(output),flush=True)


if __name__=='__main__':
    main()
