"""Fixed issuance-delay historical AQI regression stress experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

from trajectory import SOURCE, SOURCE_SHA, windows, sha, save_booster, load_booster
from forward import fold_parts
from issue_delay_features import build

HERE = Path(__file__).resolve().parent
ARMS = (('lag0_base', 0, False), ('lag1_base', 1, False),
        ('lag1_trend', 1, True), ('lag3_base', 3, False), ('lag3_trend', 3, True))
LEADS = (1, 6, 24)


def metrics(target, prediction):
    valid = np.isfinite(prediction).all(axis=1)
    y, p = target[valid].astype('float64'), prediction[valid].astype('float64')
    result = dict(origins=len(target), scored_origins=int(valid.sum()),
                  coverage=float(valid.mean()) if len(valid) else None)
    if not len(y):
        return {**result, 'mae': None, 'rmse': None, 'leads': {}}
    error = p-y
    result.update(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                  leads={f'h{lead}': dict(mae=float(np.abs(error[:, j]).mean()),
                         rmse=float(np.sqrt(np.square(error[:, j]).mean()))) for j, lead in enumerate(LEADS)})
    return result


def diagnostics(target, prediction, origin_aqi):
    return {name: metrics(target[keep], prediction[keep]) for name, keep in (
        ('all', np.ones(len(target), dtype=bool)), ('clean_origin', origin_aqi <= 100),
        ('polluted_origin', origin_aqi > 100))}


def main():
    started = time.perf_counter()
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen source changed')
    deps = [SOURCE, *[HERE/name for name in ('ISSUE_DELAY_PROTOCOL.md', 'issue_delay.py',
        'issue_delay_features.py', 'trajectory.py', 'forward.py', 'feature_ablation.py')]]
    hashes = {str(path): sha(path) for path in deps}
    raw = pd.read_parquet(SOURCE)
    w = windows(raw)
    y = w['target'][:, np.array(LEADS)-1, 2]
    output = HERE/'issue_delay_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', source_sha256=hashes, folds={}, pooled_test={}, models={},
        target='available-station-median AQI', units='AQI', leads=list(LEADS),
        historical_stress_test=True, blind_test=False, prospective=False, deployed=False,
        versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__),
        eligibility=w['eligibility'])
    pooled = {name: [] for name, _, _ in ARMS}
    pooled.update({f'persistence_lag{lag}': [] for lag in (0, 1, 3)})

    def save_report():
        temporary = output/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(output/'report.json')

    save_report()
    print(json.dumps(dict(run_dir=str(output))), flush=True)
    try:
        for cutoff in (2019, 2020, 2021):
            parts = fold_parts(w['timestamps'], cutoff)
            train, ev = parts == 'train', np.isin(parts, ['dev', 'test'])
            key = str(cutoff+2)
            meta = pd.DataFrame(dict(source_index=w['indices'][ev], timestamp=w['timestamps'][ev],
                partition=parts[ev], origin_aqi=w['history'][ev, -1, 2]))
            meta.to_parquet(output/f'{key}_origins.parquet', index=False)
            target = y[ev]
            np.save(output/f'{key}_targets.npy', target, allow_pickle=False)
            report['folds'][key] = dict(train_end_year=cutoff, train_n=int(train.sum()),
                train_clean=int((train & ~w['polluted']).sum()),
                train_polluted=int((train & w['polluted']).sum()), arms={})
            for name, lag, trend in ARMS:
                x = build(raw, w, cutoff, lag, trend)
                expected = 139 if trend else 76
                if x.shape != (len(y), expected) or x.dtype != np.float32 or np.isinf(x).any():
                    raise ValueError('Invalid feature matrix')
                prediction = np.empty_like(target)
                for j, lead in enumerate(LEADS):
                    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=300,
                        learning_rate=.04, num_leaves=24, min_child_samples=30, random_state=42,
                        n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
                    model.fit(x[train], y[train, j])
                    pred = model.predict(x[ev]).clip(min=0).astype('float32')
                    path = output/f'{key}_{name}_h{lead}.txt'
                    save_booster(model.booster_, path)
                    restored = load_booster(path).predict(x[ev], num_threads=4).clip(min=0).astype('float32')
                    np.testing.assert_array_equal(pred, restored)
                    if not np.isfinite(pred).all():
                        raise ValueError('Nonfinite prediction')
                    prediction[:, j] = pred
                    report['models'][path.name] = sha(path)
                    model.booster_.free_dataset()
                    del model
                np.save(output/f'{key}_{name}.npy', prediction, allow_pickle=False)
                predictions = {name: prediction}
                if not trend:
                    persistence = np.repeat(x[ev, 71:72], len(LEADS), axis=1)
                    persistence_name = f'persistence_lag{lag}'
                    np.save(output/f'{key}_{persistence_name}.npy', persistence, allow_pickle=False)
                    predictions[persistence_name] = persistence
                for candidate, p in predictions.items():
                    record = {}
                    for partition in ('dev', 'test'):
                        m = meta.partition.eq(partition).to_numpy()
                        origin = meta.origin_aqi.to_numpy()[m]
                        record[partition] = diagnostics(target[m], p[m], origin)
                        if partition == 'test':
                            pooled[candidate].append((target[m], p[m], origin))
                    report['folds'][key]['arms'][candidate] = record
                save_report()
                print(json.dumps(dict(fold=key, arm=name,
                    test=report['folds'][key]['arms'][name]['test']['all'])), flush=True)
                del x
        for name, items in pooled.items():
            yy, pp, oo = [np.concatenate([item[j] for item in items]) for j in range(3)]
            report['pooled_test'][name] = diagnostics(yy, pp, oo)
        report['paired_comparisons'] = {}
        for lag in (1, 3):
            base = report['pooled_test'][f'lag{lag}_base']['all']
            trend = report['pooled_test'][f'lag{lag}_trend']['all']
            report['paired_comparisons'][f'lag{lag}_trend_vs_base'] = dict(
                mae_difference=trend['mae']-base['mae'],
                relative_mae_reduction=(base['mae']-trend['mae'])/base['mae'],
                years_improved=sum(report['folds'][str(year)]['arms'][f'lag{lag}_trend']['test']['all']['mae'] <
                    report['folds'][str(year)]['arms'][f'lag{lag}_base']['test']['all']['mae']
                    for year in (2021, 2022, 2023)))
        if any(sha(Path(path)) != digest for path, digest in hashes.items()):
            raise ValueError('Frozen source or protocol changed during run')
        report.update(status='completed_fixed_protocol', elapsed_seconds=time.perf_counter()-started)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                      elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        report['artifact_sha256'] = {path.name: sha(path) for path in output.iterdir()
                                    if path.suffix in ('.npy', '.txt', '.parquet')}
        save_report()
    print(json.dumps(dict(run_dir=str(output), status=report['status'],
        paired_comparisons=report['paired_comparisons'], elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
