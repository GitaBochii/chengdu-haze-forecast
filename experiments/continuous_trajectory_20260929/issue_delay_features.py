"""Pollution histories end before issue time; labels and calendar do not shift."""
import numpy as np
import pandas as pd
from feature_ablation import summaries


def build(raw,w,cutoff,lag,trend):
    if not isinstance(lag,int) or lag<0:
        raise ValueError('Nonnegative integer observation delay required')
    ts=pd.DatetimeIndex(raw.timestamp)
    if ts.tz is None or not ts.is_unique or ts.hasnans or not (np.diff(ts.asi8)==3600*10**9).all():
        raise ValueError('Unique consecutive timezone-aware hourly source required')
    idx=np.asarray(w['indices'],int)
    if (idx<0).any() or (idx>=len(ts)).any():
        raise ValueError('Issue index outside source')
    origin=ts[idx].tz_convert('Asia/Shanghai')
    if not np.array_equal(origin.asi8,pd.DatetimeIndex(w['timestamps']).asi8):
        raise ValueError('Issue timestamps do not match source indices')
    pos=idx[:,None]+np.arange(-23,1)-lag
    safe=pos.clip(0,len(ts)-1)
    source_year=ts.tz_convert('Asia/Shanghai').year.to_numpy()[safe]
    origin_year=origin.year.to_numpy()
    valid=(pos>=0)&(pos<len(ts))
    valid &= np.where(origin_year[:,None]<=cutoff,source_year<=cutoff,source_year==origin_year[:,None])
    values=raw[['pm2_5','pm10','aqi']].to_numpy(float)
    values[~np.isfinite(values)|(values<0)]=np.nan
    history=values[safe].copy()
    history[~valid]=np.nan
    calendar=np.stack([np.sin(2*np.pi*origin.hour/24),np.cos(2*np.pi*origin.hour/24),
                       np.sin(2*np.pi*origin.dayofyear/366),np.cos(2*np.pi*origin.dayofyear/366)],axis=1)
    base=np.column_stack([history.astype('float32').reshape(len(idx),72),calendar]).astype('float32')
    return np.column_stack([base,summaries(history)]).astype('float32') if trend else base
