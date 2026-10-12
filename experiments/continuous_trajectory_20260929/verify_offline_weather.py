"""Independent frozen offline-weather audit, without training-code imports."""
import json
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from verify_forward import check, score
from verify_probability import labels, sha


HERE = Path(__file__).resolve().parent
RUN = HERE / 'offline_weather_runs/20260929T103927867552Z'


def statistics(a):
    a = np.asarray(a, dtype=np.float64)
    pieces = []
    for lag in (1, 3, 6, 12, 23):
        pieces.append(a[:, 23] - a[:, 23 - lag])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for length in (3, 6, 12, 24):
            window = a[:, 24 - length:24]
            mean = np.nanmean(window, axis=1)
            pieces.extend((mean, np.nanstd(window, axis=1),
                           np.nanmax(window, axis=1) - np.nanmin(window, axis=1), a[:, 23] - mean))
    return np.column_stack(pieces).astype(np.float32)


def reconstruct(raw, weather, indices):
    positions = indices[:, None] - np.arange(23, -1, -1)
    assert (positions <= indices[:, None]).all()
    assert (positions[:, 0] == indices - 23).all()
    t = pd.DatetimeIndex(raw.timestamp.iloc[indices])
    h = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)[positions].astype(np.float32)
    assert np.isfinite(h).all()
    calendar = np.column_stack((np.sin(t.hour * (2*np.pi) / 24), np.cos(t.hour * (2*np.pi) / 24),
                                np.sin(t.dayofyear * (2*np.pi) / 366), np.cos(t.dayofyear * (2*np.pi) / 366)))
    baseline = np.column_stack((h.reshape(len(h), 72), calendar)).astype(np.float32)
    aux = raw[['so2', 'no2', 'co', 'o3_8h']].to_numpy(float)[positions]
    aux[(~np.isfinite(aux)) | (aux < 0)] = np.nan
    chemistry = np.column_stack((baseline, statistics(h), aux.reshape(len(h), 96),
                                 statistics(aux), np.isfinite(aux).sum(axis=1) / 24)).astype(np.float32)
    met = weather[['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']].to_numpy(float)
    met[~np.isfinite(met)] = np.nan
    speed = np.hypot(met[:, 2], met[:, 3])
    wh = np.column_stack((met, speed))[positions]
    all_features = np.column_stack((chemistry, wh.reshape(len(h), 120), statistics(wh))).astype(np.float32)
    assert chemistry.shape[1] == 323 and all_features.shape[1] == 548
    return all_features


def combine(rows):
    r = {key: sum(x[key] for x in rows) for key in ('tp', 'fn', 'fp', 'tn')}
    tp, fn, fp, tn = (r[k] for k in ('tp', 'fn', 'fp', 'tn'))
    r.update(recall=tp / (tp + fn), false_positive_rate=fp / (fp + tn),
             precision=tp / (tp + fp) if tp + fp else None, f1=2*tp / (2*tp + fp + fn))
    return r


def diagnose(meta, y, p, threshold, prevalence):
    bins = []
    for i in range(5):
        m = (p >= i/5) & ((p < (i+1)/5) if i != 4 else (p <= 1))
        bins.append(dict(lower=i/5, upper=(i+1)/5, n=int(m.sum()),
                         mean_probability=float(p[m].mean()) if m.any() else None,
                         observed_rate=float(y[m].mean()) if m.any() else None))
    grouped = {}
    for name, levels in {'season': ['DJF', 'MAM', 'JJA', 'SON'],
                         'intensity': ['<=50', '50-100', '100-150', '>150'],
                         'stations': ['<=6', '7-8', '>=9', 'missing']}.items():
        grouped[name] = {}
        for level in levels:
            m = meta[name].eq(level).to_numpy()
            grouped[name][level] = score(y[m], p[m], threshold) if m.any() else None
    return dict(overall=score(y, p, threshold), calibration=bins,
                constant_train_brier=float(np.mean((prevalence-y)**2)), groups=grouped)


def main():
    report = json.loads((RUN / 'report.json').read_text('utf-8'))
    assert report['status'] == 'completed' and report['deployed'] is False
    assert report['feature_dimension'] == 548 and 'OFFLINE' in report['scope']
    checked = {}
    for category in ('source_sha256', 'artifact_sha256'):
        for filename, digest in report[category].items():
            path = Path(filename) if category == 'source_sha256' else RUN / filename
            assert sha(path) == digest, path
            checked[str(path)] = digest
    source = next(Path(p) for p in report['source_sha256'] if Path(p).name == 'city_hourly.parquet')
    weather_path = next(Path(p) for p in report['source_sha256'] if Path(p).name == 'weather_proxy.parquet')
    base_path = next(Path(p) for p in report['source_sha256'] if Path(p).name == 'report.json')
    base = json.loads(base_path.read_text('utf-8'))
    for filename, digest in base['artifact_sha256'].items():
        assert sha(base_path.parent / filename) == digest
    raw, weather = pd.read_parquet(source), pd.read_parquet(weather_path)
    pd.testing.assert_series_equal(raw.timestamp, weather.timestamp)
    ts = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    assert ts.is_unique and not ts.hasnans and (np.diff(ts.asi8) == 3600*10**9).all()
    observed = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    valid = (np.isfinite(observed) & (observed >= 0)).all(axis=1)
    indices = np.arange(23, len(ts)-24)[np.lib.stride_tricks.sliding_window_view(valid, 48).all(axis=1)]
    indices = indices[ts[indices+24].year <= 2023]
    future = observed[indices[:, None] + np.arange(1, 25), 2].astype(np.float32)
    aqi = observed[indices, 2].astype(np.float32)
    audit = dict(status='passed', report_sha256=sha(RUN / 'report.json'), verifier_sha256=sha(Path(__file__)),
                 helper_sha256={n: sha(HERE / n) for n in ('verify_probability.py', 'verify_forward.py')},
                 scope='OFFLINE ERA5 REANALYSIS ONLY. Past valid times do not prove historical publication or availability. Not deployable evidence.',
                 independence='No offline_weather, feature_ablation, trajectory, forward, constrained, probability or event_audit imports. Rebuilt all 548 features, source windows, labels, selection, subgroup metrics and pooled gate.',
                 hashes_checked=checked, deployed=False, folds={}, pooled={})
    expected_keys = {f'{year}_{task}' for year in (2021, 2022, 2023) for task in ('onset', 'recovery')}
    assert set(report['folds']) == expected_keys
    baseline_scores = {}
    for key, record in report['folds'].items():
        year_text, task = key.split('_')
        year = int(year_text)
        population = aqi <= 100 if task == 'onset' else aqi > 100
        y_all = labels(future, task)
        start, end = ts[indices-23], ts[indices+24]
        train = population & (end < pd.Timestamp(f'{year-1}-01-01', tz=ts.tz))
        dev = population & (start >= pd.Timestamp(f'{year-1}-01-01', tz=ts.tz)) & (end < pd.Timestamp(f'{year}-01-01', tz=ts.tz))
        test = population & (start >= pd.Timestamp(f'{year}-01-01', tz=ts.tz)) & (end < pd.Timestamp(f'{year+1}-01-01', tz=ts.tz))
        assert not (train & dev).any() and not (train & test).any() and not (dev & test).any()
        assert int(train.sum()) == record['train_n']
        assert int(y_all[train].sum()) == record['train_positive']
        ev = dev | test
        selected = indices[ev]
        y = y_all[ev]
        meta = pd.read_parquet(RUN / f'{key}_origins.parquet')
        pd.testing.assert_frame_equal(meta, pd.read_parquet(base_path.parent / f'{key}_origins.parquet'))
        expected = dict(source_index=selected, timestamp=ts[selected].to_numpy(), label=y,
                        partition=np.where(dev[ev], 'dev', 'test'), origin_aqi=aqi[ev])
        month = ts[selected].month
        expected['season'] = np.array(['DJF' if m in (12, 1, 2) else 'MAM' if m in (3, 4, 5) else 'JJA' if m in (6, 7, 8) else 'SON' for m in month])
        expected['intensity'] = np.array(['<=50' if a <= 50 else '50-100' if a <= 100 else '100-150' if a <= 150 else '>150' for a in aqi[ev]])
        expected['stations'] = np.array(['missing' if not np.isfinite(s) else '<=6' if s <= 6 else '7-8' if s <= 8 else '>=9' for s in raw.aqi_station_count.to_numpy()[selected]])
        for name, values in expected.items():
            np.testing.assert_array_equal(meta[name], values)
        x = reconstruct(raw, weather, selected)
        # Truncating both sources after an origin must preserve all its features.
        probe = int(selected[len(selected)//2])
        np.testing.assert_array_equal(reconstruct(raw.iloc[:probe+1], weather.iloc[:probe+1], np.array([probe])),
                                      reconstruct(raw, weather, np.array([probe])))
        probabilities, grid = {}, []
        for candidate in ('unweighted', 'balanced'):
            p = np.load(RUN / f'{key}_{candidate}.npy', allow_pickle=False)
            assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            model = lgb.Booster(model_str=(RUN / f'{key}_{candidate}.txt').read_text('utf-8'))
            assert model.num_feature() == 548
            np.testing.assert_array_equal(model.predict(x, num_threads=4), p)
            probabilities[candidate] = p
            for i in range(1, 102):
                grid.append(dict(candidate=candidate, threshold=i/100, **score(y[dev[ev]], p[dev[ev]], i/100)))
        check(grid, record['development_grid'], key + '/grid')
        budget = .1 if task == 'onset' else .05
        assert record['budget'] == budget
        chosen = max((r for r in grid if r['false_positive_rate'] <= budget),
                     key=lambda r: (r['recall'] or 0, r['precision'] or 0, -r['false_positive_rate'], r['threshold'], r['candidate'] == 'unweighted'))
        check(chosen, record['chosen'], key + '/chosen')
        diagnostics = {}
        for part in ('dev', 'test'):
            m = expected['partition'] == part
            diagnostics[part] = diagnose(meta.loc[m], y[m], probabilities[chosen['candidate']][m], chosen['threshold'], float(y_all[train].mean()))
            check(diagnostics[part], record['diagnostics'][part], key + '/' + part)
        old = base['folds'][key]
        bp = np.load(base_path.parent / f'{key}_{old["chosen"]["candidate"]}.npy', allow_pickle=False)
        m = test[ev]
        baseline_scores[key] = score(y[m], bp[m], old['chosen']['threshold'])
        check(baseline_scores[key], old['diagnostics']['test']['overall'], key + '/baseline')
        audit['folds'][key] = dict(train_n=int(train.sum()), train_positive=int(y_all[train].sum()), evaluation_n=len(y),
                                  source_alignment_verified=True, labels_groups_partitions_verified=True,
                                  same_baseline_population=True, weather_valid_times_past_only=True,
                                  future_truncation_invariance=True, feature_dimension=548,
                                  reloaded_models_exact=2, grid_rows_verified=len(grid), chosen=chosen,
                                  diagnostics=diagnostics)
        print('PASS ' + key, flush=True)
    for task, budget in (('onset', .1), ('recovery', .05)):
        keys = [k for k in expected_keys if k.endswith('_' + task)]
        b = combine([baseline_scores[k] for k in keys])
        n = combine([audit['folds'][k]['diagnostics']['test']['overall'] for k in keys])
        years = sum(audit['folds'][k]['diagnostics']['test']['overall']['recall'] > baseline_scores[k]['recall'] for k in keys)
        result = dict(baseline=b, new=n, years_recall_improved=years,
                      meaningful_gate=bool(n['recall'] >= b['recall']+.05 and n['false_positive_rate'] <= min(budget, b['false_positive_rate']) and years >= 2))
        check(result, report['pooled'][task], task + '/pooled')
        audit['pooled'][task] = result
    for filename, digest in checked.items():
        assert sha(Path(filename)) == digest
    (RUN / 'independent_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', 'utf-8')
    print(json.dumps(audit['pooled'], indent=2))
    print('PASS: 12 models, 548 independently reconstructed features, selection, all diagnostics and gate; OFFLINE ONLY')


if __name__ == '__main__':
    main()
