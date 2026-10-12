"""Frozen observational classifier adapter, independent of the deployed website."""
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_features(frame, origin):
    origin=pd.Timestamp(origin)
    ts=pd.DatetimeIndex(frame.timestamp)
    if origin.tzinfo is None or ts.tz is None or origin != origin.floor('h'):
        raise ValueError('Timezone-aware hourly origin and observations required')
    expected=pd.date_range(end=origin, periods=24, freq='h')
    if len(ts)!=24 or not np.array_equal(ts.asi8, expected.asi8):
        raise ValueError('Exactly 24 ordered hours ending at origin required')
    history=frame[['pm2_5','pm10','aqi']].to_numpy(dtype=float)
    if not np.isfinite(history).all() or (history<0).any():
        raise ValueError('Finite nonnegative pollution measurements required')
    if (history>np.finfo(np.float32).max).any():
        raise ValueError('Pollution values exceed model numeric range')
    local=origin.tz_convert('Asia/Shanghai')
    calendar=[np.sin(2*np.pi*local.hour/24),np.cos(2*np.pi*local.hour/24),
              np.sin(2*np.pi*local.dayofyear/366),np.cos(2*np.pi*local.dayofyear/366)]
    return np.r_[history.astype('float32').ravel(),calendar].astype('float32')[None,:]


def baseline_manifest(project):
    run=Path(project)/'experiments/continuous_trajectory_20260929/probability_runs/20260929T094914341676Z'
    report=json.loads((run/'report.json').read_text('utf-8'))
    selection=json.loads((run/'constrained_metrics.json').read_text('utf-8'))
    for name,expected in report['artifacts'].items():
        if digest(run/name)!=expected:
            raise ValueError('Baseline artifact changed: '+name)
    for path,expected in selection['hashes'].items():
        if digest(path)!=expected:
            raise ValueError('Selection dependency changed: '+path)
    paths=[Path(__file__).resolve(),Path(__file__).with_name('prospective_ledger.py').resolve(),
           Path(project)/'scripts/prospective_validation.py',run/'report.json',run/'constrained_metrics.json']
    tasks={}
    for task in ('onset','recovery'):
        chosen=selection['tasks'][task]['chosen']
        path=run/f'{task}_{chosen["candidate"]}.txt'
        paths.append(path)
        tasks[task]={'model_path':str(path.resolve()),'threshold':chosen['threshold']}
    return {'kind':'pollution_only_76_classifier','tasks':tasks,
            'files':{str(p.resolve()):digest(p) for p in paths},
            'trained_years':'2014-2023','selected_year':'2024',
            'feature_order':['pm2_5','pm10','aqi'], 'horizon_hours':24,
            'scope':'uncalibrated onset/recovery classifier; not dissipation-time regression',
            'lightgbm_version':lgb.__version__, 'numpy_version':np.__version__,
            'pandas_version':pd.__version__, 'selection_path':str((run/'constrained_metrics.json').resolve()),
            'deployed':False}


def predict(manifest, frame, origin):
    if (manifest.get('kind')!='pollution_only_76_classifier'
            or manifest.get('feature_order')!=['pm2_5','pm10','aqi']
            or manifest.get('horizon_hours')!=24
            or set(manifest.get('tasks',{}))!={'onset','recovery'}):
        raise ValueError('Frozen feature/task contract mismatch')
    if manifest['lightgbm_version']!=lgb.__version__:
        raise ValueError('Frozen inference runtime changed')
    if manifest['numpy_version']!=np.__version__ or manifest['pandas_version']!=pd.__version__:
        raise ValueError('Frozen feature runtime changed')
    selection_path=manifest['selection_path']
    if selection_path not in manifest['files'] or str(Path(__file__).resolve()) not in manifest['files']:
        raise ValueError('Unbound selection or adapter')
    for path,expected in manifest['files'].items():
        if digest(path)!=expected:
            raise ValueError('Frozen input/model/adapter changed: '+path)
    selection=json.loads(Path(selection_path).read_text('utf-8'))
    for task,config in manifest['tasks'].items():
        chosen=selection['tasks'][task]['chosen']
        expected=Path(selection_path).parent/f'{task}_{chosen["candidate"]}.txt'
        if (config['model_path'] not in manifest['files']
                or Path(config['model_path']).resolve()!=expected.resolve()
                or config['threshold']!=chosen['threshold']):
            raise ValueError('Model or threshold does not match frozen selection')
    x=history_features(frame,origin)
    out={'origin_aqi':float(frame.aqi.iloc[-1])}
    for task,config in manifest['tasks'].items():
        booster=lgb.Booster(model_str=Path(config['model_path']).read_text('utf-8'))
        if booster.num_feature()!=76:
            raise ValueError('Model does not match 76-feature contract')
        probability=float(booster.predict(x,num_threads=2)[0])
        out[task+'_probability']=probability
        out[task+'_threshold']=config['threshold']
        out[task+'_decision']=bool(probability>=config['threshold'])
    return out
