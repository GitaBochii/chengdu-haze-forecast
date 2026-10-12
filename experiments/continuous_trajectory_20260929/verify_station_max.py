"""Independent fixed-station identity six-hour maximum regression audit."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from verify_issue_delay import digest, equal
from verify_delayed_chemistry import reconstruct
from verify_recovery_windows import first_hit, statistics, window_labels

HERE = Path(__file__).resolve().parent
SOURCE_RUN = HERE / 'full_curve_runs/20260930T033524196780Z'
BUFFER_RUN = HERE / 'recovery_calibration_runs/20260930T121425927928Z'
MAX_RUN = HERE / 'six_hour_max_runs/20260930T130143964861Z'
ROOT = HERE.parents[1]
STATION_SOURCE = ROOT / 'data/processed/regional_air_models_full_history/station_hourly.parquet'
STATION_SHA = '45ce6dbcf96ce804253e0756361d7303fae0d90f8b357096911f21299a6a8dc1'
CHANNELS = ['pm2_5', 'pm10', 'aqi']


def maxima(curves):
    curves = np.asarray(curves)
    assert curves.ndim == 2 and curves.shape[1] == 24 and np.isfinite(curves).all()
    result = np.empty((len(curves), 19), dtype=np.float32)
    for start in range(19):
        result[:, start] = np.max(curves[:, start:start + 6], axis=1)
    return result


def maximum_diagnostics(target, prediction, derived, origin):
    result = {}
    for name, mask in [('all', np.ones(len(target), bool)),
            ('clean_origin', origin <= 100), ('polluted_origin', origin > 100)]:
        result[name] = dict(origins=int(mask.sum()),
            mae=float(np.abs(target[mask].astype(float) - prediction[mask]).mean()) if mask.any() else None,
            curve_derived_mae=float(np.abs(target[mask].astype(float) - derived[mask]).mean()) if mask.any() else None)
    return result


def station_features(stations, timestamps, indices, cutoff):
    cleaned = stations.copy()
    for channel in CHANNELS:
        a = cleaned[channel].to_numpy(float)
        a[(a < 0) | ~np.isfinite(a)] = np.nan
        cleaned[channel] = a
    station_dates = pd.DatetimeIndex(cleaned.timestamp)
    train_available = (station_dates.year <= cutoff) & cleaned[CHANNELS].notna().any(axis=1).to_numpy()
    station_ids = sorted(cleaned.loc[train_available, 'station_id'].astype(str).unique().tolist())
    assert len(station_ids) == 9
    assert not cleaned.duplicated(['timestamp', 'station_id']).any()
    columns = [(sid, channel) for sid in station_ids for channel in CHANNELS]
    blocks = []
    for sid in station_ids:
        selected = cleaned.loc[cleaned.station_id.astype(str) == sid].set_index('timestamp')
        blocks.append(selected[CHANNELS].reindex(timestamps).to_numpy(float))
    grid = np.column_stack(blocks)
    positions = indices[:, None] - np.arange(24, 0, -1)[None, :]
    inside = (positions >= 0) & (positions < len(timestamps))
    safe = positions.clip(0, len(timestamps) - 1)
    history = grid[safe].copy()
    source_year = timestamps.year.to_numpy()[safe]
    issue_year = timestamps.year.to_numpy()[indices]
    allowed = inside & np.where(issue_year[:, None] <= cutoff, source_year <= cutoff,
        source_year == issue_year[:, None])
    history[~allowed] = np.nan
    source_times = timestamps.asi8[indices, None] - np.arange(24, 0, -1)[None, :] * 3600 * 10**9
    assert (source_times < timestamps.asi8[indices, None]).all()
    np.testing.assert_array_equal(timestamps.asi8[safe][inside], source_times[inside])
    assert np.isnan(history[~allowed]).all()
    extra = np.column_stack((history.reshape(len(indices), 648), np.isfinite(history).mean(axis=1))).astype(np.float32)
    return station_ids, columns, grid, history, extra


def main(run):
    report = json.loads((run / 'report.json').read_text('utf-8'))
    assert report['status'].startswith('completed'), report['status']
    assert Path(report['source_run']).resolve() == SOURCE_RUN.resolve()
    assert Path(report['buffer_run']).resolve() == BUFFER_RUN.resolve()
    assert report['lag'] == 1 and report['feature_dimension'] == 998
    assert Path(report['no_weather_run']).resolve() == MAX_RUN.resolve()
    assert report['base_feature_dimension'] == 323 and report['station_feature_dimension'] == 675
    assert digest(STATION_SOURCE) == STATION_SHA
    assert Path(report['station_source']).resolve() == STATION_SOURCE.resolve()
    assert not report['station_actual_receipt_time_verified'] and not report['station_actual_revision_time_verified']
    params = dict(objective='regression_l1', n_estimators=300, learning_rate=.04,
        num_leaves=24, min_child_samples=30, random_state=42, n_jobs=4,
        verbosity=-1, deterministic=True, force_col_wise=True)
    equal(params, report['params'], 'params')
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for path, expected in report['input_sha256'].items():
        assert digest(path) == expected, path
    for section in ('artifact_sha256', 'models'):
        for filename, expected in report[section].items():
            assert digest(run / filename) == expected, filename
    assert digest(HERE / 'station_max.py') == report['code_sha256']
    assert digest(HERE / 'STATION_MAX_PROTOCOL.md') == report['protocol_sha256']
    reference = json.loads((SOURCE_RUN / 'report.json').read_text('utf-8'))
    buffered = json.loads((BUFFER_RUN / 'report.json').read_text('utf-8'))
    for frozen_run, frozen_report in [(SOURCE_RUN, reference), (BUFFER_RUN, buffered)]:
        assert json.loads((frozen_run / 'independent_audit.json').read_text('utf-8'))['status'] == 'passed'
        for section in ('artifact_sha256', 'models'):
            for filename, expected in frozen_report.get(section, {}).items():
                assert digest(frozen_run / filename) == expected, filename
    no_weather = json.loads((MAX_RUN / 'report.json').read_text('utf-8'))
    prior_audit = json.loads((MAX_RUN / 'independent_audit.json').read_text('utf-8'))
    assert prior_audit['status'] == 'passed' and prior_audit['report_sha256'] == digest(MAX_RUN / 'report.json')
    for section in ('artifact_sha256', 'models'):
        for filename, expected in no_weather[section].items():
            assert digest(MAX_RUN / filename) == expected, filename
    source = next(Path(p) for p in reference['source_sha256'] if p.endswith('city_hourly.parquet'))
    assert digest(source) == reference['source_sha256'][str(source)]
    raw = pd.read_parquet(source)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    assert dates.is_unique and (np.diff(dates.asi8) == 3600 * 10**9).all()
    station_raw = pd.read_parquet(STATION_SOURCE)
    assert set(station_raw.city.dropna().unique()) == {'chengdu'}
    assert not station_raw.duplicated(['timestamp', 'station_id']).any()
    pollutant_columns = ['pm2_5', 'pm10', 'aqi', 'so2', 'no2', 'co', 'o3_8h']
    city_medians = station_raw.groupby('timestamp')[pollutant_columns].median().reindex(dates)
    np.testing.assert_array_equal(city_medians.to_numpy(), raw[pollutant_columns].to_numpy())
    for pollutant in pollutant_columns:
        counts = station_raw.groupby('timestamp')[pollutant].count().reindex(dates).to_numpy()
        np.testing.assert_array_equal(counts, raw[pollutant + '_station_count'].to_numpy())
    pd.testing.assert_frame_equal(raw[['timestamp']], pd.read_parquet(run / 'station_timestamps.parquet'))
    core = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    bad = ((core < 0) | ~np.isfinite(core)).any(axis=1)
    cumulative = np.r_[0, np.cumsum(bad)]
    possible = np.arange(23, len(raw) - 24)
    complete = cumulative[possible + 25] == cumulative[possible - 23]
    years = dates.year.to_numpy()
    partition = np.select([years <= 2023, years == 2024, years == 2025], [0, 1, 2], default=3)
    same = (partition[possible - 23] == partition[possible]) & (partition[possible + 24] == partition[possible])
    indices = possible[complete & same]
    population = dict(source_rows=len(raw), edge_rows=len(raw) - len(possible),
        invalid_window_rows=int((~complete).sum()),
        split_boundary_rows_with_complete_data=int((complete & ~same).sum()), eligible=len(indices))
    pooled = {name: [] for name in ('station', 'no_weather', 'raw', 'buffered')}
    actual_pool, model_names, retraining = [], set(), []
    diagnostic_pool = {name: [] for name in ('target', 'prediction', 'derived', 'origin')}
    grid_count = 0
    for cutoff in (2019, 2020, 2021):
        year = str(cutoff + 2)
        lo, hi = years[indices - 23], years[indices + 24]
        part = np.full(len(indices), 'excluded', dtype='U8')
        part[hi <= cutoff] = 'train'
        part[(lo == cutoff + 1) & (hi == cutoff + 1)] = 'dev'
        part[(lo == cutoff + 2) & (hi == cutoff + 2)] = 'test'
        train, ev = part == 'train', np.isin(part, ['dev', 'test'])
        ei, ti = indices[ev], indices[train]
        meta = pd.read_parquet(run / f'{year}_origins.parquet')
        pd.testing.assert_frame_equal(meta, pd.read_parquet(SOURCE_RUN / f'{year}_origins.parquet'))
        pd.testing.assert_frame_equal(meta, pd.read_parquet(MAX_RUN / f'{year}_origins.parquet'))
        np.testing.assert_array_equal(meta.source_index, ei)
        np.testing.assert_array_equal(pd.DatetimeIndex(meta.timestamp).asi8, dates[ei].asi8)
        np.testing.assert_array_equal(meta.partition, part[ev])
        origin = core[ei, 2].astype(np.float32)
        np.testing.assert_array_equal(meta.origin_aqi, origin)
        target = core[ei[:, None] + np.arange(1, 25), 2].astype(np.float32)
        np.testing.assert_array_equal(target, np.load(run / f'{year}_targets.npy'))
        np.testing.assert_array_equal(target, np.load(SOURCE_RUN / f'{year}_targets.npy'))
        np.testing.assert_array_equal(target, np.load(MAX_RUN / f'{year}_targets.npy'))
        true = first_hit(window_labels(target))
        target_max = maxima(target)
        np.testing.assert_array_equal(true, first_hit(target_max <= 100))
        np.testing.assert_array_equal(target_max, np.load(run / f'{year}_maximum_targets.npy'))
        train_target = core[ti[:, None] + np.arange(1, 25), 2].astype(np.float32)
        train_max = maxima(train_target)
        np.testing.assert_array_equal(train_max, np.load(run / f'{year}_train_maximum_targets.npy'))
        np.testing.assert_array_equal(train_max, np.load(MAX_RUN / f'{year}_train_maximum_targets.npy'))
        training_meta = pd.read_parquet(run / f'{year}_train_origins.parquet')
        assert list(training_meta.columns) == ['source_index', 'timestamp']
        np.testing.assert_array_equal(training_meta.source_index, ti)
        np.testing.assert_array_equal(pd.DatetimeIndex(training_meta.timestamp).asi8, dates[ti].asi8)
        fold = report['folds'][year]
        assert fold['train_end_year'] == cutoff and fold['train_n'] == train.sum()
        assert fold['train_clean'] == np.count_nonzero(train & (core[indices, 2] <= 100))
        assert fold['train_polluted'] == np.count_nonzero(train & (core[indices, 2] > 100))
        x, _ = reconstruct(raw, dates, indices, cutoff, 1)
        assert x.shape == (len(indices), 323) and not np.isinf(x).any()
        station_ids, columns, grid, history, extras = station_features(station_raw, dates, indices, cutoff)
        equal(station_ids, fold['station_ids'], f'{year}/station_ids')
        column_names = [dict(station_id=sid, channel=channel) for sid, channel in columns]
        equal(column_names, fold['grid_columns'], f'{year}/grid_columns')
        column_info = dict(station_ids=station_ids, grid_columns=column_names,
            history_columns=[dict(lag=lag, **column) for lag in range(24, 0, -1) for column in column_names],
            observed_fraction_columns=column_names)
        equal(column_info, json.loads((run / f'{year}_station_columns.json').read_text('utf-8')), f'{year}/column_info')
        np.testing.assert_array_equal(grid, np.load(run / f'{year}_station_grid.npy'))
        x = np.column_stack((x, extras)).astype(np.float32)
        assert x.shape == (len(indices), 998) and not np.isinf(x).any()
        probe = int(ei[len(ei) // 2])
        perturbed_station = station_raw.copy()
        future_rows = pd.DatetimeIndex(perturbed_station.timestamp).asi8 >= dates.asi8[probe]
        perturbed_station.loc[future_rows, CHANNELS] = 99999
        changed_ids, _, _, _, changed_extra = station_features(perturbed_station, dates, np.array([probe]), cutoff)
        original_ids, _, _, _, original_extra = station_features(station_raw, dates, np.array([probe]), cutoff)
        assert changed_ids == original_ids
        np.testing.assert_array_equal(changed_extra, original_extra)
        coverage = {}
        for pname in ('dev', 'test'):
            mask = part == pname
            finite = np.isfinite(history[mask])
            coverage[pname] = dict(origins=int(mask.sum()), observed_fraction_by_channel=finite.mean(axis=(0, 1)).tolist(),
                all_channels_complete_window_fraction=float(finite.all(axis=(1, 2)).mean()))
        equal(coverage, fold['coverage'], f'{year}/coverage')
        sample = ei[::100]
        altered = raw.copy()
        for column in ('pm2_5', 'pm10', 'aqi', 'so2', 'no2', 'co', 'o3_8h'):
            altered.loc[altered.index[sample], column] = 99999
        check, _ = reconstruct(altered, dates, sample, cutoff, 1)
        original, _ = reconstruct(raw, dates, sample, cutoff, 1)
        np.testing.assert_array_equal(check, original)
        prediction = np.load(run / f'{year}_maximum_predictions.npy')
        assert prediction.shape == (len(ei), 19) and prediction.dtype == np.float32
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
        for j in range(19):
            filename = f'{year}_window{j + 1}.txt'
            model_names.add(filename)
            model = lgb.Booster(model_str=(run / filename).read_text('utf-8'))
            assert model.num_feature() == 998 and model.current_iteration() == 300
            assert model.params['objective'] == 'regression_l1'
            assert model.params['learning_rate'] == .04 and model.params['num_leaves'] == 24
            assert model.params['min_data_in_leaf'] == 30
            replay_prediction = np.maximum(model.predict(x[ev], num_threads=4), 0).astype(np.float32)
            np.testing.assert_array_equal(replay_prediction, prediction[:, j])
        replay = lgb.LGBMRegressor(objective='regression_l1', n_estimators=300,
            learning_rate=.04, num_leaves=24, min_child_samples=30, random_state=42,
            n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
        replay.fit(x[train], train_max[:, 0])
        np.testing.assert_array_equal(np.maximum(replay.predict(x[ev]), 0).astype(np.float32), prediction[:, 0])
        retraining.append(dict(year=int(year), window=1, exact_prediction_match=True))
        raw_curve = np.load(SOURCE_RUN / f'{year}_lag1_chemistry.npy')
        raw_start = first_hit(window_labels(raw_curve))
        buffer_delta = buffered['folds'][year]['lag1_chemistry']['delta']
        buffer_start = first_hit(window_labels(raw_curve.astype(float) + buffer_delta))
        dev = (meta.partition.to_numpy() == 'dev') & (origin > 100)
        test = (meta.partition.to_numpy() == 'test') & (origin > 100)
        raw_dev = statistics(true[dev], raw_start[dev])
        rows = []
        assert len(fold['development_grid']) == 102
        for delta, saved in zip(range(102), fold['development_grid']):
            metric = statistics(true[dev], first_hit(prediction[dev].astype(float) + delta <= 100))
            feasible = (metric['fpr'] is not None and metric['fpr'] <= .05 and
                metric['premature_start_fraction_all'] <= raw_dev['premature_start_fraction_all'])
            row = dict(delta=delta, metrics=metric, feasible=feasible)
            equal(row, saved, f'{year}/grid/{delta}')
            rows.append(row)
            grid_count += 1
        selected = min([row for row in rows if row['feasible']], key=lambda row:
            (row['metrics']['capped_min_T_20_mae_hours'], -(row['metrics']['recall'] or 0),
                row['metrics']['fpr'], -row['delta']))
        equal(selected['delta'], fold['delta'], f'{year}/selected')
        direct_start = first_hit(prediction.astype(float) + selected['delta'] <= 100)
        equal(statistics(true[dev], direct_start[dev]), fold['dev'], f'{year}/dev')
        no_weather_prediction = np.load(MAX_RUN / f'{year}_maximum_predictions.npy')
        no_weather_start = first_hit(no_weather_prediction.astype(float) + no_weather['folds'][year]['delta'] <= 100)
        for name, predicted, key in [('station', direct_start, 'test'),
                ('no_weather', no_weather_start, 'no_weather_test'),
                ('raw', raw_start, 'raw_test'), ('buffered', buffer_start, 'buffered_test')]:
            equal(statistics(true[test], predicted[test]), fold[key], f'{year}/{key}')
            if name == 'no_weather':
                equal(statistics(true[test], predicted[test]), no_weather['folds'][year]['test'], f'{year}/frozen_no_weather')
            pooled[name].append(predicted[test])
        derived = maxima(raw_curve)
        for pname in ('dev', 'test'):
            mask = meta.partition.to_numpy() == pname
            equal(maximum_diagnostics(target_max[mask], prediction[mask], derived[mask], origin[mask]),
                fold['max_diagnostics'][pname], f'{year}/{pname}/max_diagnostics')
        all_test = meta.partition.to_numpy() == 'test'
        for name, values in dict(target=target_max, prediction=prediction,
                derived=derived, origin=origin).items():
            diagnostic_pool[name].append(values[all_test])
        actual_pool.append(true[test])
        del x
        print(f'Independent station-maximum fold {year}: passed', flush=True)
    assert len(model_names) == 57 and set(report['models']) == model_names
    actual = np.concatenate(actual_pool)
    for name, arrays in pooled.items():
        equal(statistics(actual, np.concatenate(arrays)), report['pooled_test'][name], f'pooled/{name}')
    diagnostics = {name: np.concatenate(values) for name, values in diagnostic_pool.items()}
    equal(maximum_diagnostics(**diagnostics), report['pooled_max_diagnostics'], 'pooled/max_diagnostics')
    candidate, control = report['pooled_test']['station'], report['pooled_test']['buffered']
    gate = dict(timing_at_least_10pct_better=candidate['capped_min_T_20_mae_hours'] <= .9 * control['capped_min_T_20_mae_hours'],
        recall_at_least_5pp_better=candidate['recall'] >= control['recall'] + .05,
        pooled_fpr_at_most_5pct=candidate['fpr'] <= .05,
        all_years_fpr_at_most_5pct=all(fold['test']['fpr'] <= .05 for fold in report['folds'].values()),
        no_year_timing_worse=all(fold['test']['capped_min_T_20_mae_hours'] <=
            fold['buffered_test']['capped_min_T_20_mae_hours'] for fold in report['folds'].values()))
    success = all(gate.values())
    gate['passed'] = success
    equal(gate, report['clear_improvement_gate'], 'clear_improvement_gate')
    audit = dict(status='passed', independently_reinferred_models=57,
        independently_scored_candidates=grid_count, independently_retrained_models=retraining,
        independently_computed_eligibility=population, success_gate_passed=success,
        training_feature_builder_imported=False, report_sha256=digest(run / 'report.json'),
        verifier_sha256=digest(Path(__file__)), checks=[
            'Frozen input, model, code and protocol hashes',
            'Independent complete-window population and purged annual folds',
            'Independent lag-one 323-feature reconstruction and issue-hour perturbation',
            'All six-hour maximum labels independently reconstructed from true AQI',
            'Independent station train-only membership, feature ordering and exact joins',
            'Independent city pollutant medians and station counts',
            'Station future perturbation, annual masks and all-condition coverage',
            'All 57 model inference replays and three exact training replays',
            'All 306 development candidates and selected buffers',
            'Annual and pooled recovery metrics and fixed improvement gate'], limits=[
            'Independent retraining covers 3 of 57 models',
            'Lag-one receipt availability remains a historical assumption',
            'All intermediate raw-station preprocessing remains unverified',
            'City per-pollutant medians are not certified official city AQI',
            'Repeated historical tests, not blind or prospective',
            '20 represents a censoring cap, not observed recovery',
            'AQI recovery is not particulate-primary haze dissipation',
            'Overlapping origins are not independent events'])
    (run / 'independent_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)


