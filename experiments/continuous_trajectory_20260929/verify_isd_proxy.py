"""Independently audit saved ISD proxy features and evaluation, never retrain."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from verify_forward import check, score
from verify_probability import labels, sha
from verify_offline_weather import combine, diagnose, reconstruct, statistics


HERE = Path(__file__).resolve().parent


def weather_grid(hourly, timestamps):
    assert hourly.timestamp.is_unique
    aligned = hourly.set_index('timestamp').reindex(timestamps)
    t = aligned.noaa_temperature_c.to_numpy(float)
    d = aligned.noaa_dewpoint_c.to_numpy(float)
    direction = np.deg2rad(aligned.noaa_wind_direction_deg.to_numpy(float))
    speed = aligned.noaa_wind_speed_ms.to_numpy(float)
    rh = np.clip(100 * np.exp(17.625*d/(243.04+d) - 17.625*t/(243.04+t)), 0, 100)
    u, v = -speed * np.sin(direction), -speed * np.cos(direction)
    result = np.column_stack((t, rh, u, v, np.hypot(u, v)))
    result[~np.isfinite(result)] = np.nan
    absent = ~timestamps.isin(pd.DatetimeIndex(hourly.timestamp))
    assert np.isnan(result[absent]).all()
    return result


def delayed(grid, timestamps, indices, cutoff, lag):
    origins = timestamps[indices]
    requested = origins.asi8[:, None] - np.arange(lag+23, lag-1, -1)[None, :] * 3600 * 10**9
    source_times = pd.to_datetime(requested.ravel(), utc=True).tz_convert(timestamps.tz)
    positions = timestamps.get_indexer(source_times).reshape(len(indices), 24)
    history = np.full((len(indices), 24, 5), np.nan)
    found = positions >= 0
    history[found] = grid[positions[found]]
    source_year = source_times.year.to_numpy().reshape(len(indices), 24)
    origin_year = origins.year.to_numpy()
    # Training years form one expanding partition; dev and test are separate years.
    same = np.where(origin_year[:, None] <= cutoff, source_year <= cutoff,
                    source_year == origin_year[:, None])
    history[~same] = np.nan
    assert (source_times.asi8.reshape(len(indices), 24) <= origins.asi8[:, None] - lag*3600*10**9).all()
    assert (np.isnan(history[~same])).all()
    expected_positions = indices[:, None] - np.arange(lag+23, lag-1, -1)
    np.testing.assert_array_equal(positions[found], expected_positions[found])
    return history, int((~same).sum()), int((~found).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', nargs='?', type=Path, default=HERE / 'isd_runs/20260929T124655970592Z')
    run = parser.parse_args().run.resolve()
    report = json.loads((run / 'report.json').read_text('utf-8'))
    assert report['status'] == 'completed' and report['deployed'] is False
    assert set(report['arms']) == {'lag0', 'lag72'}
    checked = {}
    for category in ('source_sha256', 'artifact_sha256'):
        for filename, digest in report[category].items():
            path = Path(filename) if category == 'source_sha256' else run / filename
            assert sha(path) == digest, path
            checked[str(path)] = digest
    source = next(Path(p) for p in report['source_sha256'] if Path(p).name == 'city_hourly.parquet')
    base_path = next(Path(p) for p in report['source_sha256'] if Path(p).name == 'report.json')
    base = json.loads(base_path.read_text('utf-8'))
    for filename, digest in base['artifact_sha256'].items():
        path = base_path.parent / filename
        assert sha(path) == digest
        checked[str(path)] = digest
    raw = pd.read_parquet(source)
    parsed = pd.read_parquet(run / 'parsed_hourly.parquet')
    ts = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    assert ts.is_unique and not ts.hasnans and (np.diff(ts.asi8) == 3600*10**9).all()
    parsed_ts = pd.DatetimeIndex(parsed.timestamp)
    assert parsed_ts.tz is not None and parsed_ts.is_unique
    assert (parsed_ts == parsed_ts.ceil('h')).all()
    grid = weather_grid(parsed, ts)
    observed = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    valid = (np.isfinite(observed) & (observed >= 0)).all(axis=1)
    indices = np.arange(23, len(ts)-24)[np.lib.stride_tricks.sliding_window_view(valid, 48).all(axis=1)]
    indices = indices[ts[indices+24].year <= 2023]
    future = observed[indices[:, None] + np.arange(1, 25), 2].astype(np.float32)
    aqi = observed[indices, 2].astype(np.float32)
    # The independent previous auditor exposes the full chemistry block as its first 323 columns.
    placeholder = pd.DataFrame(np.zeros((len(raw), 4)), columns=['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms'])
    audit = dict(status='passed', report_sha256=sha(run / 'report.json'), verifier_sha256=sha(Path(__file__)),
                 helpers_sha256={n: sha(HERE / n) for n in ('verify_probability.py', 'verify_forward.py', 'verify_offline_weather.py')},
                 scope='Lag0 OFFLINE archived observations; lag72 assumed delay, not historical release-time evidence. No deployment claim.',
                 independence='Independent exact-time weather join, RH/wind transform, lag/partition masks, chemistry features, metrics and selection; no isd_proxy or production feature imports.',
                 parser_boundary='Saved parsed_hourly is hash-verified and inspected for unique hourly timestamps. Existing NOAA QC parser was code-reviewed, not independently reimplemented; source-to-parsed QC is outside numerical reconstruction.',
                 hashes_checked=checked, deployed=False, arms={})
    expected_keys = {f'{year}_{task}' for year in (2021, 2022, 2023) for task in ('onset', 'recovery')}
    for arm, arm_report in report['arms'].items():
        lag = int(arm.removeprefix('lag'))
        assert set(arm_report['folds']) == expected_keys
        audited_folds, baseline_scores = {}, {}
        for key, record in arm_report['folds'].items():
            year_text, task = key.split('_')
            year = int(year_text)
            cutoff = year-2
            population = aqi <= 100 if task == 'onset' else aqi > 100
            y_all = labels(future, task)
            start, end = ts[indices-23], ts[indices+24]
            train = population & (end < pd.Timestamp(f'{year-1}-01-01', tz=ts.tz))
            dev = population & (start >= pd.Timestamp(f'{year-1}-01-01', tz=ts.tz)) & (end < pd.Timestamp(f'{year}-01-01', tz=ts.tz))
            test = population & (start >= pd.Timestamp(f'{year}-01-01', tz=ts.tz)) & (end < pd.Timestamp(f'{year+1}-01-01', tz=ts.tz))
            assert not (train & dev).any() and not (train & test).any() and not (dev & test).any()
            assert int(train.sum()) == record['train_n'] and int(y_all[train].sum()) == record['train_positive']
            ev = dev | test
            selected, y = indices[ev], y_all[ev]
            meta = pd.read_parquet(run / f'{arm}_{key}_origins.parquet')
            pd.testing.assert_frame_equal(meta, pd.read_parquet(base_path.parent / f'{key}_origins.parquet'))
            expected = dict(source_index=selected, timestamp=ts[selected].to_numpy(), label=y,
                            partition=np.where(dev[ev], 'dev', 'test'), origin_aqi=aqi[ev])
            expected['season'] = np.array(['DJF' if m in (12,1,2) else 'MAM' if m in (3,4,5) else 'JJA' if m in (6,7,8) else 'SON' for m in ts[selected].month])
            expected['intensity'] = np.array(['<=50' if a <= 50 else '50-100' if a <= 100 else '100-150' if a <= 150 else '>150' for a in aqi[ev]])
            expected['stations'] = np.array(['missing' if not np.isfinite(s) else '<=6' if s <= 6 else '7-8' if s <= 8 else '>=9' for s in raw.aqi_station_count.to_numpy()[selected]])
            for name, values in expected.items():
                np.testing.assert_array_equal(meta[name], values)
            chemistry = reconstruct(raw, placeholder, selected)[:, :323]
            h, masked, outside = delayed(grid, ts, selected, cutoff, lag)
            x = np.column_stack((chemistry, h.reshape(len(h), 120), statistics(h), np.isfinite(h).sum(axis=1)/24)).astype(np.float32)
            assert x.shape[1] == 553
            probe = int(selected[len(selected)//2])
            truncated, _, _ = delayed(grid[:probe+1], ts[:probe+1], np.array([probe]), cutoff, lag)
            untruncated, _, _ = delayed(grid, ts, np.array([probe]), cutoff, lag)
            np.testing.assert_array_equal(truncated, untruncated)
            probabilities, selection = {}, []
            for candidate in ('unweighted', 'balanced'):
                p = np.load(run / f'{arm}_{key}_{candidate}.npy', allow_pickle=False)
                assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                model = lgb.Booster(model_str=(run / f'{arm}_{key}_{candidate}.txt').read_text('utf-8'))
                assert model.num_feature() == 553
                np.testing.assert_array_equal(model.predict(x, num_threads=4), p)
                probabilities[candidate] = p
                for i in range(1, 102):
                    selection.append(dict(candidate=candidate, threshold=i/100, **score(y[dev[ev]], p[dev[ev]], i/100)))
            check(selection, record['development_grid'], arm+'/'+key+'/grid')
            budget = .1 if task == 'onset' else .05
            assert record['budget'] == budget
            chosen = max((r for r in selection if r['false_positive_rate'] <= budget),
                         key=lambda r: (r['recall'] or 0, r['precision'] or 0, -r['false_positive_rate'], r['threshold'], r['candidate']=='unweighted'))
            check(chosen, record['chosen'], arm+'/'+key+'/chosen')
            diagnostics, coverage = {}, {}
            for part in ('dev', 'test'):
                m = expected['partition'] == part
                diagnostics[part] = diagnose(meta.loc[m], y[m], probabilities[chosen['candidate']][m], chosen['threshold'], float(y_all[train].mean()))
                check(diagnostics[part], record['diagnostics'][part], arm+'/'+key+'/'+part)
                finite = np.isfinite(h[m])
                coverage[part] = dict(observed_fraction_by_channel=finite.sum(axis=(0,1)).astype(float).tolist(),
                                      all_channels_complete_window_fraction=float(finite.all(axis=(1,2)).mean()))
                coverage[part]['observed_fraction_by_channel'] = [n/(int(m.sum())*24) for n in coverage[part]['observed_fraction_by_channel']]
                check(coverage[part], record['coverage'][part], arm+'/'+key+'/'+part+'/coverage')
            old = base['folds'][key]
            bp = np.load(base_path.parent / f'{key}_{old["chosen"]["candidate"]}.npy', allow_pickle=False)
            baseline_scores[key] = score(y[test[ev]], bp[test[ev]], old['chosen']['threshold'])
            check(baseline_scores[key], old['diagnostics']['test']['overall'], key+'/baseline')
            audited_folds[key] = dict(train_n=int(train.sum()), train_positive=int(y_all[train].sum()), evaluation_n=len(y),
                                      same_baseline_population=True, metadata_verified=True, feature_dimension=553,
                                      reloaded_models_exact=2, grid_rows_verified=len(selection), chosen=chosen,
                                      source_time_partition_masked_cells=masked, source_outside_grid_cells=outside,
                                      max_weather_source_lag_hours=lag+23, min_weather_source_lag_hours=lag,
                                      future_truncation_invariance=True, diagnostics=diagnostics, coverage=coverage)
            print('PASS '+arm+' '+key, flush=True)
        pooled = {}
        for task, budget in (('onset', .1), ('recovery', .05)):
            keys = [k for k in expected_keys if k.endswith('_'+task)]
            b = combine([baseline_scores[k] for k in keys])
            n = combine([audited_folds[k]['diagnostics']['test']['overall'] for k in keys])
            years = sum(audited_folds[k]['diagnostics']['test']['overall']['recall'] > baseline_scores[k]['recall'] for k in keys)
            result = dict(baseline=b, new=n, years_recall_improved=years,
                          meaningful_gate=bool(n['recall'] >= b['recall']+.05 and n['false_positive_rate'] <= min(budget,b['false_positive_rate']) and years >= 2))
            check(result, arm_report['pooled'][task], arm+'/'+task+'/pooled')
            pooled[task] = result
        audit['arms'][arm] = dict(folds=audited_folds, pooled=pooled)
    for filename, digest in checked.items():
        assert sha(Path(filename)) == digest
    (run / 'independent_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', 'utf-8')
    print(json.dumps({arm: r['pooled'] for arm,r in audit['arms'].items()}, indent=2))
    print('PASS: all 24 models, independent features/masks/coverage/selection/metrics; NOT real-time availability evidence')


if __name__ == '__main__':
    main()
