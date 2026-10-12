"""Independent six-hour maximum regression and recovery decision audit."""
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


def main(run):
    report = json.loads((run / 'report.json').read_text('utf-8'))
    assert report['status'].startswith('completed'), report['status']
    assert Path(report['source_run']).resolve() == SOURCE_RUN.resolve()
    assert Path(report['buffer_run']).resolve() == BUFFER_RUN.resolve()
    assert report['lag'] == 1 and report['feature_dimension'] == 323
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for path, expected in report['input_sha256'].items():
        assert digest(path) == expected, path
    for section in ('artifact_sha256', 'models'):
        for filename, expected in report[section].items():
            assert digest(run / filename) == expected, filename
    assert digest(HERE / 'six_hour_max.py') == report['code_sha256']
    assert digest(HERE / 'SIX_HOUR_MAX_PROTOCOL.md') == report['protocol_sha256']
    reference = json.loads((SOURCE_RUN / 'report.json').read_text('utf-8'))
    buffered = json.loads((BUFFER_RUN / 'report.json').read_text('utf-8'))
    for frozen_run, frozen_report in [(SOURCE_RUN, reference), (BUFFER_RUN, buffered)]:
        assert json.loads((frozen_run / 'independent_audit.json').read_text('utf-8'))['status'] == 'passed'
        for section in ('artifact_sha256', 'models'):
            for filename, expected in frozen_report.get(section, {}).items():
                assert digest(frozen_run / filename) == expected, filename
    source = next(Path(p) for p in reference['source_sha256'] if p.endswith('city_hourly.parquet'))
    assert digest(source) == reference['source_sha256'][str(source)]
    raw = pd.read_parquet(source)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    assert dates.is_unique and (np.diff(dates.asi8) == 3600 * 10**9).all()
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
    pooled = {name: [] for name in ('maximum', 'raw', 'buffered')}
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
        np.testing.assert_array_equal(meta.source_index, ei)
        np.testing.assert_array_equal(pd.DatetimeIndex(meta.timestamp).asi8, dates[ei].asi8)
        np.testing.assert_array_equal(meta.partition, part[ev])
        origin = core[ei, 2].astype(np.float32)
        np.testing.assert_array_equal(meta.origin_aqi, origin)
        target = core[ei[:, None] + np.arange(1, 25), 2].astype(np.float32)
        np.testing.assert_array_equal(target, np.load(run / f'{year}_targets.npy'))
        np.testing.assert_array_equal(target, np.load(SOURCE_RUN / f'{year}_targets.npy'))
        true = first_hit(window_labels(target))
        target_max = maxima(target)
        np.testing.assert_array_equal(true, first_hit(target_max <= 100))
        np.testing.assert_array_equal(target_max, np.load(run / f'{year}_maximum_targets.npy'))
        train_target = core[ti[:, None] + np.arange(1, 25), 2].astype(np.float32)
        train_max = maxima(train_target)
        np.testing.assert_array_equal(train_max, np.load(run / f'{year}_train_maximum_targets.npy'))
        fold = report['folds'][year]
        assert fold['train_end_year'] == cutoff and fold['train_n'] == train.sum()
        assert fold['train_clean'] == np.count_nonzero(train & (core[indices, 2] <= 100))
        assert fold['train_polluted'] == np.count_nonzero(train & (core[indices, 2] > 100))
        x, _ = reconstruct(raw, dates, indices, cutoff, 1)
        assert x.shape == (len(indices), 323) and not np.isinf(x).any()
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
            assert model.num_feature() == 323 and model.current_iteration() == 300
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
        for name, predicted, key in [('maximum', direct_start, 'test'),
                ('raw', raw_start, 'raw_test'), ('buffered', buffer_start, 'buffered_test')]:
            equal(statistics(true[test], predicted[test]), fold[key], f'{year}/{key}')
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
        print(f'Independent maximum-regression fold {year}: passed', flush=True)
    assert len(model_names) == 57 and set(report['models']) == model_names
    actual = np.concatenate(actual_pool)
    for name, arrays in pooled.items():
        equal(statistics(actual, np.concatenate(arrays)), report['pooled_test'][name], f'pooled/{name}')
    diagnostics = {name: np.concatenate(values) for name, values in diagnostic_pool.items()}
    equal(maximum_diagnostics(**diagnostics), report['pooled_max_diagnostics'], 'pooled/max_diagnostics')
    candidate, control = report['pooled_test']['maximum'], report['pooled_test']['buffered']
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
            'All 57 model inference replays and three exact training replays',
            'All 306 development candidates and selected buffers',
            'Annual and pooled recovery metrics and fixed improvement gate'], limits=[
            'Independent retraining covers 3 of 57 models',
            'Lag-one receipt availability remains a historical assumption',
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
