"""Frozen train-2014..2019 / development-2020 empirical complete paths."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/continuous_trajectory_20260929'
sys.path.insert(0,str(OLD))
from delayed_chemistry import build as chemistry_build
from trajectory import save_booster,load_booster
SOURCE=ROOT/'data/processed/regional_air_models_full_history/city_hourly.parquet'
SOURCE_SHA='b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'
PROTOCOL=ROOT/'docs/iteration-20261002/NEXT_EXPERIMENT_PROTOCOL.md'
CHANNELS=['pm2_5','pm10','aqi']
HOUR=3600*10**9

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def recovery(a):
    a=np.asarray(a)
    if a.ndim!=2 or a.shape[1]!=24 or not np.isfinite(a).all(): raise ValueError('Finite 24h AQI required')
    s=np.lib.stride_tricks.sliding_window_view(a<=100,6,axis=1).all(axis=2)
    return np.where(s.any(axis=1),s.argmax(axis=1)+1,20)

def groups(ts,aqi,valid):
    ts=pd.DatetimeIndex(ts); part=np.where(ts.year<=2019,'train','dev')
    epoch=pd.Timestamp('2014-01-01',tz='Asia/Shanghai').value
    result=[]; segment=-1; episode=0; active=False; clean=0
    for i,(a,v) in enumerate(zip(aqi,valid)):
        reset=i==0 or not valid[i-1] or ts.asi8[i]-ts.asi8[i-1]!=HOUR or part[i]!=part[i-1]
        if reset: segment+=1; active=False; clean=0
        if not v: result.append('invalid'); active=False; clean=0; continue
        if not active and a>100: episode+=1; active=True; clean=0
        if active:
            result.append(f'{part[i]}:s{segment}:p{episode}')
            clean=clean+1 if a<=100 else 0
            if clean==6: active=False; clean=0
        else:
            block=(ts.asi8[i]-epoch)//(72*HOUR)
            result.append(f'{part[i]}:s{segment}:c{block}')
    return np.array(result,dtype=str)

def prepare(frame):
    frame=frame.sort_values('timestamp').reset_index(drop=True)
    ts=pd.DatetimeIndex(frame.timestamp)
    if ts.tz is None or not ts.is_unique or ts.hasnans: raise ValueError('Unique aware timestamps required')
    ts=ts.tz_convert('Asia/Shanghai')
    if (ts.year<2014).any() or (ts.year>2020).any(): raise ValueError('Forbidden partition')
    a=frame[CHANNELS].to_numpy(float)
    valid=np.isfinite(a).all(axis=1)
    gp=groups(ts,a[:,2],np.isfinite(a[:,2]))
    indices=[]; exclusions=[]
    part=np.where(ts.year<=2019,'train','dev')
    for i in range(len(ts)):
        reason=None
        if i<24 or i+24>=len(ts): reason='edge'
        elif part[i-24]!=part[i] or part[i+24]!=part[i]: reason='partition_boundary'
        elif not np.all(np.diff(ts.asi8[i-24:i+25])==HOUR): reason='hour_gap'
        elif not valid[i-24:i].all() or not valid[i+1:i+25].all() or not np.isfinite(a[i,2]): reason='nonfinite'
        if reason: exclusions.append((i,ts[i],part[i],reason))
        else: indices.append(i)
    ix=np.array(indices,int); hist=a[ix[:,None]+np.arange(-24,0)]; paths=a[ix[:,None]+np.arange(1,25)]
    issue=ts[ix]
    calendar=np.stack([np.sin(2*np.pi*issue.hour/24),np.cos(2*np.pi*issue.hour/24),
        np.sin(2*np.pi*issue.month/12),np.cos(2*np.pi*issue.month/12)],axis=1)
    w=dict(indices=ix,timestamps=issue.asi8,history=hist,paths=paths,
        features=np.column_stack([hist.reshape(len(ix),72),calendar]),groups=gp[ix],
        origin_aqi=a[ix,2],labels=recovery(paths[:,:,2]),partition=part[ix])
    return w,pd.DataFrame(exclusions,columns=['source_index','timestamp','partition','reason'])

def neighbors(query,z,times,g):
    distance=np.mean((z-query)**2,axis=1)
    order=np.lexsort((times,distance)); chosen=[]; used=set(); skips=[]
    for j in order:
        if g[j] in used: skips.append((int(j),'same_group')); continue
        if any(abs(int(times[j])-int(times[k]))<49*HOUR for k in chosen): skips.append((int(j),'overlap_49h')); continue
        chosen.append(int(j)); used.add(g[j])
        if len(chosen)==32: break
    return np.array(chosen if len(chosen)==32 else [],int),skips

def distribution(labels):
    pmf=np.bincount(labels,minlength=21)[1:21]/len(labels)
    return pmf,np.cumsum(pmf)[:19]

def decision(cdf,theta):
    b=cdf>=theta
    return np.where(b.any(axis=1),b.argmax(axis=1)+1,20)

def metrics(t,p,g,_groups=True):
    t=np.asarray(t); p=np.asarray(p); yes=t<20; hat=p<20
    tp,fn,fp,tn=[int(x.sum()) for x in (yes&hat,yes&~hat,~yes&hat,~yes&~hat)]
    n=len(t); error=np.abs(t-p); early=(p<t)&hat
    ratio=lambda x,y:float(x/y) if y else None
    result=dict(n=n,actual_right_censored=int((~yes).sum()),predicted_recovery=int(hat.sum()),
        mae=float(error.mean()) if n else None,group_equal_mae=float(pd.Series(error).groupby(g).mean().mean()) if n else None,
        groups=len(set(g)),confusion=dict(tp=tp,fn=fn,fp=fp,tn=tn),recall=ratio(tp,tp+fn),precision=ratio(tp,tp+fp),
        fpr=ratio(fp,fp+tn),early_fraction=ratio(early.sum(),n),early_count=int(early.sum()))
    if _groups:
        grouped=[metrics(t[g==v],p[g==v],g[g==v],False) for v in np.unique(g)]
        result['group_equal']={k:dict(value=float(np.mean([r[k] for r in grouped if r[k] is not None])) if any(r[k] is not None for r in grouped) else None,
            valid_groups=sum(r[k] is not None for r in grouped),total_groups=len(grouped)) for k in ('mae','recall','precision','fpr','early_fraction')}
    return result

def select(rows,max_early):
    valid=[r for r in rows if r['fpr'] is not None and r['fpr']<=.05 and r['early_fraction']<=max_early and r['confusion']['tp']>0]
    return min(valid,key=lambda r:(r['mae'],-(r['recall'] or 0),r['fpr'],-r['theta'])) if valid else None

def probability_metrics(cdf,t,g):
    outcome=t[:,None]<=np.arange(1,20)
    e=(cdf-outcome)**2
    return dict(n=len(t),brier_6=float(e[:,5].mean()),brier_12=float(e[:,11].mean()),brier_19=float(e[:,18].mean()),
        brier_mean_1_19=float(e.mean()),group_equal_brier_6=float(pd.Series(e[:,5]).groupby(g).mean().mean()),
        group_equal_brier_12=float(pd.Series(e[:,11]).groupby(g).mean().mean()),group_equal_brier_19=float(pd.Series(e[:,18]).groupby(g).mean().mean()))

def main():
    start=time.perf_counter()
    if sha(SOURCE)!=SOURCE_SHA: raise ValueError('Source hash mismatch')
    run=HERE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    dependencies=[PROTOCOL,HERE/'EXECUTION_CLARIFICATIONS.md',Path(__file__),HERE/'test_experiment.py',*[OLD/n for n in ('trajectory.py','full_curve_recovery.py','FULL_CURVE_PROTOCOL.md','delayed_chemistry.py','issue_delay_features.py','feature_ablation.py','recovery_calibration.py','issue_delay.py','forward.py')]]
    hashes={str(p):sha(p) for p in [SOURCE,*dependencies]}
    report=dict(status='running',run_dir=str(run),frozen_sha256=hashes,versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__),
        k=32,train_years=list(range(2014,2020)),dev_year=2020,selected_on_development=True,blind_test=False,deployed=False,models={})
    write_json(run/'report.json',report)
    print(json.dumps(dict(run_dir=str(run))),flush=True)
    try:
        # Predicate filtering never loads forbidden years' values for modeling or selection.
        raw=pd.read_parquet(SOURCE,filters=[('timestamp','>=',pd.Timestamp('2014-01-01',tz='Asia/Shanghai')),('timestamp','<',pd.Timestamp('2021-01-01',tz='Asia/Shanghai'))]).sort_values('timestamp').reset_index(drop=True)
        w,excluded=prepare(raw); excluded.to_parquet(run/'exclusions.parquet',index=False)
        train=w['partition']=='train'; dev=~train
        a={k:v[train] for k,v in w.items() if k!='partition'}; b={k:v[dev] for k,v in w.items() if k!='partition'}
        np.savez_compressed(run/'train.npz',**a); np.savez_compressed(run/'dev.npz',**b)
        pd.DataFrame(dict(source_index=w['indices'],timestamp=pd.to_datetime(w['timestamps'],utc=True).tz_convert('Asia/Shanghai'),partition=w['partition'],group=w['groups'],origin_aqi=w['origin_aqi'],true_start=w['labels'])).to_parquet(run/'eligible.parquet',index=False)
        mean=a['features'].mean(axis=0); std=a['features'].std(axis=0); std[std==0]=1
        z=(a['features']-mean)/std; q=(b['features']-mean)/std
        np.savez_compressed(run/'standardization.npz',mean=mean,std=std,z_train=z,z_dev=q)
        n=len(q); nn=np.full((n,32),-1,int); pmf=np.full((n,20),np.nan); cdf=np.full((n,19),np.nan); onset=np.full(n,np.nan); path_mean=np.full((n,24,3),np.nan); med=np.full(n,-1,int)
        skips=[]
        for i in range(n):
            keep,skip=neighbors(q[i],z,a['timestamps'],a['groups']); skips.extend((i,j,r) for j,r in skip)
            if len(keep):
                nn[i]=keep; pmf[i],cdf[i]=distribution(a['labels'][keep]); onset[i]=np.mean((a['paths'][keep,:,2]>100).any(axis=1))
                path_mean[i]=a['paths'][keep].mean(axis=0); med[i]=np.searchsorted(np.cumsum(pmf[i]),.5)+1
            if i%1000==0: print(json.dumps(dict(stage='neighbors',done=i,total=n)),flush=True)
        pd.DataFrame(skips,columns=['dev_row','train_row','reason']).to_parquet(run/'neighbor_skips.parquet',index=False)
        np.save(run/'neighbor_paths.npy',np.where(nn[:,:,None,None]>=0,a['paths'][np.maximum(nn,0)],np.nan),allow_pickle=False)
        # Fixed chemistry configuration; the old window population cannot justify reuse.
        x=chemistry_build(raw,dict(indices=w['indices'],timestamps=pd.to_datetime(w['timestamps'],utc=True).tz_convert('Asia/Shanghai')),2019,1)
        if x.shape!=(len(w['indices']),323): raise ValueError('Chemistry feature dimension mismatch')
        np.save(run/'chemistry_features.npy',x,allow_pickle=False)
        raw_curve=np.empty((n,24),np.float32)
        for j in range(24):
            model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=300,learning_rate=.04,num_leaves=24,min_child_samples=30,random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
            model.fit(x[train],a['paths'][:,j,2]); pred=model.predict(x[dev]).clip(min=0).astype('float32')
            p=run/f'chemistry_aqi_h{j+1:02}.txt'; save_booster(model.booster_,p)
            np.testing.assert_array_equal(pred,load_booster(p).predict(x[dev],num_threads=4).clip(min=0).astype('float32'))
            raw_curve[:,j]=pred; report['models'][p.name]=sha(p); model.booster_.free_dataset(); del model
            write_json(run/'report.json',report); print(json.dumps(dict(stage='chemistry',lead=j+1)),flush=True)
        covered=nn[:,0]>=0; polluted=b['origin_aqi']>100; matched=covered&polluted; t=b['labels']; g=b['groups']
        raw_metrics=metrics(t[polluted],recovery(raw_curve[polluted]),g[polluted])
        buffer_rows=[dict(delta=d,**metrics(t[polluted],recovery(raw_curve[polluted]+d),g[polluted])) for d in range(102)]
        feasible=[r for r in buffer_rows if r['fpr'] is not None and r['fpr']<=.05 and r['early_fraction']<=raw_metrics['early_fraction']]
        buffer=min(feasible,key=lambda r:(r['mae'],-(r['recall'] or 0),r['fpr'],-r['delta'])) if feasible else None
        if buffer is None: raise ValueError('No conservative buffer')
        conservative=raw_curve.astype(float)+buffer['delta']; reference=metrics(t[matched],recovery(conservative[matched]),g[matched])
        thresholds=[round(i*.05,2) for i in range(1,21)]+[1.000001]
        matched_raw=metrics(t[matched],recovery(raw_curve[matched]),g[matched])
        rows=[dict(theta=h,**metrics(t[matched],decision(cdf[matched],h),g[matched])) for h in thresholds]
        selected=select(rows,matched_raw['early_fraction']); theta=selected['theta'] if selected else 1.000001
        prediction=decision(cdf,theta); prediction[~covered]=-1
        ucdf=np.mean(a['labels'][:,None]<=np.arange(1,20),axis=0); uonset=float(np.mean((a['paths'][:,:,2]>100).any(axis=1)))
        np.savez_compressed(run/'prediction.npz',neighbors=nn,selected_group_count=np.where(covered,32,0),pmf=pmf,cdf=cdf,pred_start=prediction,raw_curve=raw_curve,conservative_curve=conservative,path_mean=path_mean,onset_probability=onset,median_start=med,unconditional_cdf=ucdf,unconditional_onset=np.array(uonset))
        write_json(run/'grid.json',rows); write_json(run/'buffer_grid.json',buffer_rows)
        new=metrics(t[matched],prediction[matched],g[matched]); prob=probability_metrics(cdf[matched],t[matched],g[matched]); unconditional=probability_metrics(np.broadcast_to(ucdf,cdf[matched].shape),t[matched],g[matched])
        clean=covered&~polluted; truth_onset=(b['paths'][:,:,2]>100).any(axis=1); clean_negative=clean&~truth_onset
        clean_brier=float(np.mean((onset[clean]-truth_onset[clean])**2)); clean_unconditional=float(np.mean((uonset-truth_onset[clean])**2))
        clean_fpr=float(np.mean(onset[clean_negative]>=.5)) if clean_negative.any() else None
        clean_rates=[]
        for group in np.unique(g[clean]):
            m=clean&(g==group); neg=m&~truth_onset; pos=m&truth_onset; hat=onset>=.5
            clean_rates.append(dict(fpr=float(hat[neg].mean()) if neg.any() else None,recall=float(hat[pos].mean()) if pos.any() else None,precision=float(truth_onset[m&hat].mean()) if (m&hat).any() else None))
        clean_equal={k:dict(value=float(np.mean([r[k] for r in clean_rates if r[k] is not None])) if any(r[k] is not None for r in clean_rates) else None,valid_groups=sum(r[k] is not None for r in clean_rates),total_groups=len(clean_rates)) for k in ('fpr','recall','precision')}
        clean_metrics=dict(n=int(clean.sum()),negative_n=int(clean_negative.sum()),fpr=clean_fpr,brier=clean_brier,unconditional_brier=clean_unconditional,group_equal=clean_equal,
            group_equal_brier=float(pd.Series((onset[clean]-truth_onset[clean])**2).groupby(g[clean]).mean().mean()),recovery=metrics(t[clean],prediction[clean],g[clean]))
        concentration={}
        for condition,mask in [('all',covered),('polluted',matched),('clean',clean)]:
            error=np.abs(path_mean[mask]-b['paths'][mask])
            concentration[condition]={name:dict(n=int(mask.sum()),hour_cells=int(mask.sum()*24),mae=float(error[:,:,j].mean()),lead_mae=error[:,:,j].mean(axis=0).tolist(),group_equal_mae=float(pd.Series(error[:,:,j].mean(axis=1)).groupby(g[mask]).mean().mean()),units='AQI' if name=='aqi' else 'ug/m3') for j,name in enumerate(CHANNELS)}
        gates=dict(coverage_100=bool(covered.all()),actual_detection=selected is not None,mae_10pct=bool(new['mae']<=reference['mae']*.9),recall_5pp=bool(new['recall'] is not None and reference['recall'] is not None and new['recall']>=reference['recall']+.05),fpr_5pct=bool(new['fpr'] is not None and new['fpr']<=.05),year_mae_not_worse=bool(new['mae']<=reference['mae']),process_mae_not_worse=bool(new['group_equal_mae']<=reference['group_equal_mae']),brier_6=bool(prob['brier_6']<=unconditional['brier_6']),brier_12=bool(prob['brier_12']<=unconditional['brier_12']),brier_19=bool(prob['brier_19']<=unconditional['brier_19']),clean_fpr_5pct=bool(clean_fpr is not None and clean_fpr<=.05),clean_brier=bool(clean_brier<=clean_unconditional))
        row=pd.DataFrame(dict(dev_row=np.arange(n),timestamp=pd.to_datetime(b['timestamps'],utc=True),group=g,polluted=polluted,covered=covered,true_start=t,pred_start=prediction,raw_start=recovery(raw_curve),conservative_start=recovery(conservative),onset_probability=onset,true_onset=truth_onset))
        row['absolute_error']=np.where(covered,np.abs(prediction-t),np.nan); row['reference_absolute_error']=np.abs(row.conservative_start-t)
        row.to_parquet(run/'row_metrics.parquet',index=False)
        row.groupby(['polluted','group'],as_index=False).agg(n=('dev_row','size'),mae=('absolute_error','mean'),reference_mae=('reference_absolute_error','mean')).to_parquet(run/'group_metrics.parquet',index=False)
        reliability=[]
        for s in (6,12,19):
            p=cdf[matched,s-1]; truth=t[matched]<=s
            for j in range(10):
                mask=(p>=j/10)&((p<(j+1)/10) if j<9 else (p<=1))
                reliability.append(dict(horizon=s,bin=j,n=int(mask.sum()),mean_probability=float(p[mask].mean()) if mask.any() else None,observed_fraction=float(truth[mask].mean()) if mask.any() else None))
        pd.DataFrame(reliability).to_parquet(run/'reliability.parquet',index=False)
        if any(sha(p)!=h for p,h in hashes.items()): raise ValueError('Frozen source/dependency changed')
        report.update(status='completed_pending_verification',preliminary_development_pass=all(gates.values()),counts=dict(train=len(z),dev=n,covered=int(covered.sum()),no_prediction=int((~covered).sum()),polluted=int(polluted.sum()),clean=int((~polluted).sum()),exclusions=len(excluded)),selected_theta=selected['theta'] if selected else None,selected_buffer=buffer['delta'],metrics=dict(raw=raw_metrics,raw_matched=matched_raw,conservative_all_polluted=metrics(t[polluted],recovery(conservative[polluted]),g[polluted]),conservative_matched=reference,empirical_matched=new,probability=prob,unconditional_probability=unconditional,clean=clean_metrics,concentration=concentration),gates=gates,failure_reasons=[k for k,v in gates.items() if not v],elapsed_seconds=time.perf_counter()-start)
    except BaseException as exc:
        report.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc()); raise
    finally:
        report['artifact_sha256']={p.name:sha(p) for p in run.iterdir() if p.is_file() and p.name!='report.json'}
        write_json(run/'report.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
