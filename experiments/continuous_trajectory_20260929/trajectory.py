"""Observed-history-only, direct 24-hour pollution trajectory baseline."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FEATURES = ('pm2_5', 'pm10', 'aqi')
SOURCE = ROOT / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
SOURCE_SHA = 'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_booster(booster, path):
    # Avoid the Windows C++ narrow-path API for a Unicode project directory.
    path.write_text(booster.model_to_string(), encoding='utf-8')


def load_booster(path):
    return lgb.Booster(model_str=path.read_text(encoding='utf-8'))


def windows(frame):
    ts = pd.DatetimeIndex(frame.timestamp)
    if ts.tz is None or not ts.is_unique or ts.hasnans or not np.all(np.diff(ts.asi8) == 3600*10**9):
        raise ValueError('Unique consecutive timezone-aware hourly grid required')
    ts = ts.tz_convert('Asia/Shanghai')
    raw = frame[list(FEATURES)].to_numpy(dtype='float64')
    valid = np.isfinite(raw).all(axis=1) & (raw >= 0).all(axis=1)
    cumulative = np.r_[0, np.cumsum(~valid)]
    part = np.where(ts.year <= 2023, 'train', np.where(ts.year == 2024, 'dev',
                    np.where(ts.year == 2025, 'test', 'recent_audit')))
    origins = np.arange(23, len(ts)-24)
    complete = cumulative[origins+25] == cumulative[origins-23]
    same = (part[origins-23] == part[origins]) & (part[origins+24] == part[origins])
    selected = origins[complete & same]
    history = raw[selected[:, None] + np.arange(-23, 1)].astype('float32')
    target = raw[selected[:, None] + np.arange(1, 25)].astype('float32')
    return {'indices': selected, 'timestamps': ts[selected], 'partition': part[selected],
            'history': history, 'target': target, 'polluted': history[:, -1, 2] > 100,
            'eligibility': {'source_rows': len(ts), 'edge_rows': len(ts)-len(origins),
                'invalid_window_rows': int((~complete).sum()),
                'split_boundary_rows_with_complete_data': int((complete & ~same).sum()),
                'eligible': len(selected)}}


def baselines(history):
    return {'persistence': np.repeat(history[:, -1:, :], 24, axis=1),
            'daily_repeat': history.copy()}


def features(history, timestamps):
    if history.shape[1:] != (24, 3) or not np.isfinite(history).all():
        raise ValueError('Complete 24-hour three-channel histories required')
    ts = pd.DatetimeIndex(timestamps)
    calendar = np.stack([np.sin(2*np.pi*ts.hour/24), np.cos(2*np.pi*ts.hour/24),
                         np.sin(2*np.pi*ts.dayofyear/366), np.cos(2*np.pi*ts.dayofyear/366)], axis=1)
    return np.concatenate([history.reshape(len(history), -1), calendar], axis=1).astype('float32')


def score(target, prediction, days):
    if target.shape != prediction.shape or target.ndim != 3 or target.shape[1:] != (24, 3):
        raise ValueError('Matched 24-step trajectories required')
    if not np.isfinite(prediction).all() or (prediction < 0).any() or not np.isfinite(target).all():
        raise ValueError('Finite nonnegative predictions and observed targets required')
    days = np.asarray(days)
    if len(days) != len(target) or not len(days):
        raise ValueError('Nonempty aligned days required')
    error = prediction.astype('float64') - target.astype('float64')
    result = {}
    for j, name in enumerate(FEATURES):
        absolute = np.abs(error[:, :, j])
        row_errors = absolute.mean(axis=1)
        result[name] = {'mae': float(absolute.mean()), 'rmse': float(np.sqrt(np.mean(error[:, :, j]**2))),
                        'day_equal_mae': float(pd.Series(row_errors).groupby(days).mean().mean()),
                        **{f'mae_h{h}': float(absolute[:, h-1].mean()) for h in (1,6,12,24)}}
    return result


def main():
    started = time.perf_counter()
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen source changed')
    identities = {str(path): sha(path) for path in (SOURCE, Path(__file__), HERE/'PROTOCOL.md')}
    w = windows(pd.read_parquet(SOURCE))
    X = features(w['history'], w['timestamps'])
    train = w['partition'] == 'train'
    evaluation = ~train
    if not train.any() or not (train & w['polluted']).any():
        raise ValueError('Both training conditions required')
    output = HERE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    meta = pd.DataFrame({'timestamp': w['timestamps'], 'partition': w['partition'],
                         'origin_aqi': w['history'][:, -1, 2], 'polluted': w['polluted'],
                         'source_index': w['indices']})
    meta.to_parquet(output/'all_eligible_origins.parquet', index=False)
    meta.loc[evaluation].to_parquet(output/'evaluation_origins.parquet', index=False)
    y = w['target'][evaluation]
    predictions = baselines(w['history'][evaluation])
    np.save(output/'targets.npy', y, allow_pickle=False)
    counts = meta.groupby(['partition', 'polluted']).size().to_dict()
    report = {'status': 'running', 'source_sha256': identities, 'eligibility': w['eligibility'],
              'counts': {f'{part}/{condition}': int(n) for (part, condition), n in counts.items()},
              'features': '72 observed lag values + origin calendar sin/cos',
              'models': {}, 'metrics': {}, 'lightgbm_version': lgb.__version__,
              'trajectory_target_order': list(FEATURES), 'forecast_leads': list(range(1,25)),
              'dissipation_time_evaluated': False, 'deployed': False}
    def save_report():
        temp = output/'report.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), 'utf-8')
        temp.replace(output/'report.json')
    save_report()
    print(json.dumps({'run_dir': str(output), 'counts': report['counts']}), flush=True)
    for arm in ('pollution_only', 'all_conditions'):
        keep = train & w['polluted'] if arm == 'pollution_only' else train
        predictions[arm] = np.empty_like(y)
        for j, name in enumerate(FEATURES):
            for h in range(24):
                model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=300,
                    learning_rate=.04, num_leaves=24, min_child_samples=30, random_state=42,
                    n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
                model.fit(X[keep], w['target'][keep, h, j])
                pred = model.predict(X[evaluation]).clip(min=0).astype('float32')
                if not np.isfinite(pred).all():
                    raise ValueError('Invalid forecast')
                predictions[arm][:, h, j] = pred
                path = output/f'{arm}_{name}_h{h+1:02}.txt'
                save_booster(model.booster_, path)
                restored = load_booster(path)
                again = restored.predict(X[evaluation], num_threads=4).clip(min=0).astype('float32')
                if not np.array_equal(pred, again):
                    raise ValueError('Model serialization changed predictions')
                report['models'][path.name] = sha(path)
                model.booster_.free_dataset()
                if (h+1) % 6 == 0:
                    save_report()
                    print(json.dumps({'arm': arm, 'channel': name, 'completed_lead': h+1}), flush=True)
                del restored, model
        np.save(output/f'{arm}.npy', predictions[arm], allow_pickle=False)
    eval_meta = meta.loc[evaluation].reset_index(drop=True)
    days = eval_meta.timestamp.dt.strftime('%Y-%m-%d').to_numpy()
    for arm, pred in predictions.items():
        np.save(output/f'{arm}.npy', pred, allow_pickle=False)
        report['metrics'][arm] = {}
        for part in ('dev', 'test', 'recent_audit'):
            for condition in ('all', 'clean_origin', 'polluted_origin'):
                keep = eval_meta.partition.eq(part).to_numpy()
                if condition != 'all':
                    keep &= eval_meta.polluted.to_numpy() == (condition == 'polluted_origin')
                if not keep.any():
                    raise ValueError('Empty evaluation category')
                report['metrics'][arm][part+'/'+condition] = {'origins': int(keep.sum()),
                    'days': int(len(set(days[keep]))), 'scores': score(y[keep], pred[keep], days[keep])}
    if any(sha(Path(path)) != digest for path, digest in identities.items()):
        raise ValueError('Source or frozen protocol changed')
    report['artifact_sha256'] = {path.name: sha(path) for path in output.iterdir() if path.suffix in ('.npy', '.parquet')}
    report.update(status='completed_fixed_protocol', elapsed_seconds=time.perf_counter()-started)
    save_report()
    print(json.dumps({'status': report['status'], 'run_dir': str(output), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
