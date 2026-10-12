"""Prior archived weather added to fixed six-hour AQI maximum regressors."""
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
from delayed_chemistry import build
from feature_ablation import summaries
from full_curve_recovery import recovery_start, recovery_metrics
from recovery_windows import start_metrics, BASE, BUFFER
from six_hour_max import maximum_targets, maximum_start, select_buffer, max_diagnostics
from isd_wind import corrected_grid, delayed_weather, add_noaa_weather

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEATHER = HERE / 'isd_runs' / '20260929T124655970592Z'
NO_WEATHER = HERE / 'six_hour_max_runs' / '20260930T130143964861Z'
PARAMS = dict(objective='regression_l1', n_estimators=300, learning_rate=.04,
    num_leaves=24, min_child_samples=30, random_state=42, n_jobs=4,
    verbosity=-1, deterministic=True, force_col_wise=True)


def weather_history(grid, timestamps, indices, cutoff):
    g = np.asarray(grid, dtype=float)
    ts = pd.DatetimeIndex(timestamps)
    idx = np.asarray(indices)
    if g.ndim != 2 or g.shape != (len(ts), 5) or np.isinf(g).any():
        raise ValueError('Five weather channels aligned to hourly timestamps required')
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or ((idx < 0) | (idx >= len(ts))).any():
        raise ValueError('Valid integer issue positions required')
    if ts.has_duplicates or not ts.is_monotonic_increasing or (np.diff(ts.asi8) != 3600_000_000_000).any():
        raise ValueError('Exact consecutive hourly timestamps required')
    return delayed_weather(g, ts, idx, cutoff, 1)


def old_dependencies(reference):
    result = {}
    for key, expected in reference['source_sha256'].items():
        normalized = key.replace('\\', '/')
        matches = [normalized.index('/' + name + '/') for name in ('data', 'src', 'experiments')
                   if '/' + name + '/' in normalized]
        if len(matches) != 1:
            raise ValueError(f'Cannot identify archived dependency suffix: {key}')
        path = ROOT / normalized[matches[0] + 1:]
        if sha(path) != expected:
            raise ValueError(f'Archived weather dependency changed: {path}')
        result[str(path)] = expected
    for year in range(2014, 2024):
        path = ROOT / f'data/raw/noaa_isd/{year}/56294099999.csv'
        if str(path) not in result:
            raise ValueError(f'Missing archived raw-weather identity: {year}')
    return result


def coverage(history, mask):
    finite = np.isfinite(history[mask])
    return dict(origins=int(mask.sum()),
        observed_fraction_by_channel=finite.mean(axis=(0, 1)).tolist(),
        all_channels_complete_window_fraction=float(finite.all(axis=(1, 2)).mean()))


def main():
    started = time.perf_counter()
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen source changed')
    ref = json.loads((BASE / 'report.json').read_text('utf-8'))
    weather_ref = json.loads((WEATHER / 'report.json').read_text('utf-8'))
    no_weather_ref = json.loads((NO_WEATHER / 'report.json').read_text('utf-8'))
    selected = json.loads((BUFFER / 'selected.json').read_text('utf-8'))
    inputs = old_dependencies(weather_ref)
    inputs[str(SOURCE)] = SOURCE_SHA
    for n in ('trajectory.py', 'forward.py', 'delayed_chemistry.py', 'issue_delay_features.py',
              'feature_ablation.py', 'full_curve_recovery.py', 'recovery_windows.py', 'six_hour_max.py', 'isd_wind.py'):
        inputs[str(HERE / n)] = sha(HERE / n)
    for base in (BASE, BUFFER, NO_WEATHER):
        audit = json.loads((base / 'independent_audit.json').read_text('utf-8'))
        if audit['status'] != 'passed' or sha(base / 'report.json') != audit['report_sha256']:
            raise ValueError('Control is not frozen and verified')
        for n in ('report.json', 'independent_audit.json'):
            inputs[str(base / n)] = sha(base / n)
    inputs[str(BUFFER / 'selected.json')] = sha(BUFFER / 'selected.json')
    inputs[str(WEATHER / 'report.json')] = sha(WEATHER / 'report.json')
    inputs[str(WEATHER / 'parsed_hourly.parquet')] = weather_ref['artifact_sha256']['parsed_hourly.parquet']
    for base, baseline in ((BASE, ref), (NO_WEATHER, no_weather_ref)):
        for n, h in baseline['artifact_sha256'].items():
            if n.endswith(('.npy', '.parquet')):
                inputs[str(base / n)] = h
    for p, h in inputs.items():
        if sha(Path(p)) != h:
            raise ValueError(f'Frozen dependency changed: {p}')
    raw = pd.read_parquet(SOURCE)
    hourly = pd.read_parquet(WEATHER / 'parsed_hourly.parquet')
    aligned = add_noaa_weather(raw[['timestamp']], hourly)
    pd.testing.assert_series_equal(aligned.timestamp.reset_index(drop=True), raw.timestamp.reset_index(drop=True))
    grid = corrected_grid(aligned)
    w = windows(raw)
    y = w['target'][:, :, 2]
    labels = maximum_targets(y)
    output = HERE / 'weather_max_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report = dict(status='running', source_run=str(BASE), buffer_run=str(BUFFER),
        weather_run=str(WEATHER), no_weather_run=str(NO_WEATHER),
        input_sha256=inputs, code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(HERE / 'WEATHER_MAX_PROTOCOL.md'),
        params=PARAMS, lightgbm_version=lgb.__version__, lag=1,
        base_feature_dimension=323, weather_feature_dimension=230, feature_dimension=553,
        weather_channels=['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms', 'noaa_wind_speed_ms'],
        weather_actual_receipt_time_verified=False, weather_actual_revision_time_verified=False,
        folds={}, models={}, pooled_test={}, historical_stress_test=True,
        blind_test=False, prospective=False, deployed=False, concentration_predictions_changed=False)
    pooled = {k: [] for k in ('truth', 'weather', 'no_weather', 'raw', 'buffered')}
    diagnosis = {k: [] for k in ('target', 'prediction', 'curve', 'origin')}

    def save():
        tmp = output / 'report.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False), 'utf-8')
        tmp.replace(output / 'report.json')

    save()
    print(output, flush=True)
    try:
        np.save(output / 'weather_grid.npy', grid, allow_pickle=False)
        raw[['timestamp']].to_parquet(output / 'weather_timestamps.parquet', index=False)
        for cutoff in (2019, 2020, 2021):
            year = str(cutoff + 2)
            parts = fold_parts(w['timestamps'], cutoff)
            train, ev = parts == 'train', np.isin(parts, ['dev', 'test'])
            meta = pd.DataFrame(dict(source_index=w['indices'][ev], timestamp=w['timestamps'][ev],
                partition=parts[ev], origin_aqi=w['history'][ev, -1, 2]))
            for control in (BASE, NO_WEATHER):
                pd.testing.assert_frame_equal(meta, pd.read_parquet(control / f'{year}_origins.parquet'))
                np.testing.assert_array_equal(y[ev], np.load(control / f'{year}_targets.npy'))
            np.testing.assert_array_equal(labels[train], np.load(NO_WEATHER / f'{year}_train_maximum_targets.npy'))
            h = weather_history(grid, raw.timestamp, w['indices'], cutoff)
            x = np.column_stack([build(raw, w, cutoff, 1), h.reshape(len(h), -1),
                                 summaries(h), np.isfinite(h).mean(axis=1)]).astype('float32')
            if x.shape != (len(y), 553):
                raise ValueError('Feature dimension changed')
            pred = np.empty((int(ev.sum()), 19), dtype='float32')
            np.save(output / f'{year}_train_maximum_targets.npy', labels[train], allow_pickle=False)
            pd.DataFrame(dict(source_index=w['indices'][train], timestamp=w['timestamps'][train])).to_parquet(
                output / f'{year}_train_origins.parquet', index=False)
            for j in range(19):
                model = lgb.LGBMRegressor(**PARAMS)
                model.fit(x[train], labels[train, j])
                p = model.predict(x[ev]).clip(min=0).astype('float32')
                path = output / f'{year}_window{j + 1}.txt'
                save_booster(model.booster_, path)
                np.testing.assert_array_equal(p, load_booster(path).predict(x[ev], num_threads=4).clip(min=0).astype('float32'))
                pred[:, j] = p
                report['models'][path.name] = sha(path)
                print(json.dumps(dict(year=year, window=j + 1)), flush=True)
                save()
            meta.to_parquet(output / f'{year}_origins.parquet', index=False)
            target = y[ev]
            for name, values in dict(targets=target, maximum_targets=labels[ev], maximum_predictions=pred).items():
                np.save(output / f'{year}_{name}.npy', values, allow_pickle=False)
            curve = np.load(BASE / f'{year}_lag1_chemistry.npy')
            buffered = curve.astype(float) + selected[year]['lag1_chemistry']
            no_weather_pred = np.load(NO_WEATHER / f'{year}_maximum_predictions.npy')
            no_weather_start = maximum_start(no_weather_pred, no_weather_ref['folds'][year]['delta'])
            origin = meta.origin_aqi.to_numpy()
            dev = (meta.partition.to_numpy() == 'dev') & (origin > 100)
            test = (meta.partition.to_numpy() == 'test') & (origin > 100)
            raw_dev = recovery_metrics(target[dev], curve[dev])
            delta, devgrid = select_buffer(target[dev], pred[dev], raw_dev['premature_start_fraction_all'])
            truth, predicted = recovery_start(target), maximum_start(pred, delta)
            report['folds'][year] = dict(train_end_year=cutoff, train_n=int(train.sum()),
                train_clean=int((train & ~w['polluted']).sum()), train_polluted=int((train & w['polluted']).sum()),
                delta=delta, development_grid=devgrid, dev=start_metrics(truth[dev], predicted[dev]),
                test=start_metrics(truth[test], predicted[test]), raw_test=recovery_metrics(target[test], curve[test]),
                buffered_test=recovery_metrics(target[test], buffered[test]),
                no_weather_test=start_metrics(truth[test], no_weather_start[test]), coverage={}, max_diagnostics={})
            for part in ('dev', 'test'):
                k = meta.partition.to_numpy() == part
                report['folds'][year]['coverage'][part] = coverage(h[ev], k)
                report['folds'][year]['max_diagnostics'][part] = max_diagnostics(target[k], pred[k], curve[k], origin[k])
            for name, values in dict(truth=truth, weather=predicted, no_weather=no_weather_start,
                                     raw=recovery_start(curve), buffered=recovery_start(buffered)).items():
                pooled[name].append(values[test])
            k = meta.partition.to_numpy() == 'test'
            for name, values in dict(target=target, prediction=pred, curve=curve, origin=origin).items():
                diagnosis[name].append(values[k])
            save()
            del x, h
        a = {k: np.concatenate(v) for k, v in pooled.items()}
        report['pooled_test'] = {k: start_metrics(a['truth'], a[k]) for k in ('weather', 'no_weather', 'raw', 'buffered')}
        q = {k: np.concatenate(v) for k, v in diagnosis.items()}
        report['pooled_max_diagnostics'] = max_diagnostics(q['target'], q['prediction'], q['curve'], q['origin'])
        c, b = report['pooled_test']['weather'], report['pooled_test']['buffered']
        gate = dict(timing_at_least_10pct_better=c['capped_min_T_20_mae_hours'] <= .9 * b['capped_min_T_20_mae_hours'],
            recall_at_least_5pp_better=c['recall'] >= b['recall'] + .05, pooled_fpr_at_most_5pct=c['fpr'] <= .05,
            all_years_fpr_at_most_5pct=all(f['test']['fpr'] <= .05 for f in report['folds'].values()),
            no_year_timing_worse=all(f['test']['capped_min_T_20_mae_hours'] <= f['buffered_test']['capped_min_T_20_mae_hours'] for f in report['folds'].values()))
        gate['passed'] = all(gate.values())
        report['clear_improvement_gate'] = gate
        if any(sha(Path(p)) != expected for p, expected in inputs.items()):
            raise ValueError('Frozen input changed during experiment')
        if sha(Path(__file__)) != report['code_sha256'] or sha(HERE / 'WEATHER_MAX_PROTOCOL.md') != report['protocol_sha256']:
            raise ValueError('Protocol or implementation changed during experiment')
        report.update(status='completed_fixed_protocol', elapsed_seconds=time.perf_counter() - started)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['artifact_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.suffix in ('.txt', '.npy', '.parquet')}
        save()
    print(json.dumps(dict(metrics=report['pooled_test'], gate=report['clear_improvement_gate']), indent=2), flush=True)


if __name__ == '__main__':
    main()
