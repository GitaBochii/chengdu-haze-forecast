"""Independent censored integer-bin AFT reconstruction, refitting and scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
import xgboost as xgb

from verify_issue_delay import digest, equal
from verify_delayed_chemistry import reconstruct
from verify_recovery_windows import window_labels, first_hit, statistics

HERE = Path(__file__).resolve().parent
SOURCE_RUN = HERE / 'full_curve_runs/20260930T033524196780Z'
BUFFER_RUN = HERE / 'recovery_calibration_runs/20260930T121425927928Z'
SCALES = (.35, .7, 1.2)
THRESHOLDS = list(np.arange(1, 21, dtype=float) / 20) + [1.000001]


def bounds(starts):
    starts = np.asarray(starts)
    assert ((starts >= 1) & (starts <= 20)).all()
    low = starts.astype(float) - .5
    high = starts.astype(float) + .5
    low[starts == 1] = 0
    high[starts == 20] = np.inf
    return np.column_stack((low, high))


def decode(margin, sigma, threshold):
    margin = np.asarray(margin, dtype=float)
    assert np.isfinite(margin).all()
    probability = ndtr((np.log(19.5) - margin) / sigma)
    inverse_input = np.clip(.5 * probability, 1e-12, 1 - 1e-12)
    # Rounding only occurs on announced recoveries; overflow remains censored.
    conditional_log = margin + sigma * ndtri(inverse_input)
    conditional = np.exp(np.minimum(conditional_log, np.log(19.5)))
    timing = np.floor(conditional + .5).clip(1, 19).astype(np.int64)
    timing[probability < threshold] = 20
    return timing, probability


def fixed_params(sigma):
    return dict(objective='survival:aft', eval_metric='aft-nloglik',
        aft_loss_distribution='normal', aft_loss_distribution_scale=sigma,
        eta=.05, max_depth=4, min_child_weight=8, subsample=.9,
        colsample_bytree=.8, reg_lambda=3, seed=42, tree_method='hist', nthread=4, verbosity=0)


def main(run):
    report = json.loads((run / 'report.json').read_text('utf-8'))
    assert report['status'].startswith('completed'), report['status']
    assert Path(report['source_run']).resolve() == SOURCE_RUN.resolve()
    assert Path(report['buffer_run']).resolve() == BUFFER_RUN.resolve()
    assert report['lag'] == 1 and report['feature_dimension'] == 323
    assert report['num_boost_round'] == 500
    equal(list(SCALES), report['scales'], 'scales')
    equal(THRESHOLDS, report['thresholds'], 'thresholds')
    params = fixed_params(.35)
    del params['aft_loss_distribution_scale']
    equal(params, report['params'], 'fixed_params')
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for path, expected in report['input_sha256'].items():
        assert digest(path) == expected, path
    for section in ('artifact_sha256', 'models'):
        for filename, expected in report[section].items():
            assert digest(run / filename) == expected, filename
    assert digest(HERE / 'aft_binned_recovery.py') == report['code_sha256']
    assert digest(HERE / 'AFT_BINNED_RECOVERY_PROTOCOL.md') == report['protocol_sha256']
    reference = json.loads((SOURCE_RUN / 'report.json').read_text('utf-8'))
    buffered = json.loads((BUFFER_RUN / 'report.json').read_text('utf-8'))
    for frozen_run, frozen_report in [(SOURCE_RUN, reference), (BUFFER_RUN, buffered)]:
        audit = json.loads((frozen_run / 'independent_audit.json').read_text('utf-8'))
        assert audit['status'] == 'passed' and audit['report_sha256'] == digest(frozen_run / 'report.json')
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
    actual_pool, model_names, retraining = [], set(), []
    pooled = {name: [] for name in ('aft', 'raw', 'buffered')}
    candidate_count = 0
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
        truth = first_hit(window_labels(target))
        training_truth = first_hit(window_labels(core[ti[:, None] + np.arange(1, 25), 2].astype(np.float32)))
        expected_bounds = bounds(training_truth)
        np.testing.assert_array_equal(expected_bounds, np.load(run / f'{year}_train_bounds.npy'))
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
        sample = ei[::100]
        altered = raw.copy()
        for column in ('pm2_5', 'pm10', 'aqi', 'so2', 'no2', 'co', 'o3_8h'):
            altered.loc[altered.index[sample], column] = 99999
        check, _ = reconstruct(altered, dates, sample, cutoff, 1)
        original, _ = reconstruct(raw, dates, sample, cutoff, 1)
        np.testing.assert_array_equal(check, original)
        train_matrix = xgb.DMatrix(x[train])
        train_matrix.set_float_info('label_lower_bound', expected_bounds[:, 0])
        train_matrix.set_float_info('label_upper_bound', expected_bounds[:, 1])
        evaluation = xgb.DMatrix(x[ev])
        margins = {}
        for sigma in SCALES:
            stem = f'{year}_sigma{sigma}'
            model_names.add(stem + '.ubj')
            model = xgb.Booster()
            model.load_model(bytearray((run / (stem + '.ubj')).read_bytes()))
            assert model.num_features() == 323 and model.num_boosted_rounds() == 500
            margin = np.load(run / (stem + '_margins.npy'))
            assert margin.shape == (len(ei),) and margin.dtype == np.float32 and np.isfinite(margin).all()
            np.testing.assert_array_equal(model.predict(evaluation, output_margin=True), margin)
            np.testing.assert_allclose(model.predict(evaluation), np.exp(margin), rtol=1e-6, atol=1e-6)
            independent = xgb.train(fixed_params(sigma), train_matrix, num_boost_round=500)
            np.testing.assert_array_equal(independent.predict(evaluation, output_margin=True), margin)
            retraining.append(dict(year=int(year), sigma=sigma, exact_prediction_match=True))
            margins[sigma] = margin
        curve = np.load(SOURCE_RUN / f'{year}_lag1_chemistry.npy')
        raw_start = first_hit(window_labels(curve))
        buffer_delta = buffered['folds'][year]['lag1_chemistry']['delta']
        buffered_start = first_hit(window_labels(curve.astype(float) + buffer_delta))
        dev = (meta.partition.to_numpy() == 'dev') & (origin > 100)
        test = (meta.partition.to_numpy() == 'test') & (origin > 100)
        raw_dev = statistics(truth[dev], raw_start[dev])
        grid = []
        assert len(fold['development_grid']) == 63
        for sigma in SCALES:
            for threshold in THRESHOLDS:
                starts, _ = decode(margins[sigma], sigma, threshold)
                metric = statistics(truth[dev], starts[dev])
                feasible = metric['fpr'] <= .05 and metric['premature_start_fraction_all'] <= raw_dev['premature_start_fraction_all']
                grid.append(dict(sigma=sigma, threshold=threshold, metrics=metric, feasible=feasible))
        for index, (actual, saved) in enumerate(zip(grid, fold['development_grid'])):
            equal(actual, saved, f'{year}/candidate/{index}')
            candidate_count += 1
        selected = min([row for row in grid if row['feasible']], key=lambda row:
            (row['metrics']['capped_min_T_20_mae_hours'], -(row['metrics']['recall'] or 0),
                row['metrics']['fpr'], -row['threshold'], -row['sigma']))
        equal(selected['sigma'], fold['sigma'], f'{year}/sigma')
        equal(selected['threshold'], fold['threshold'], f'{year}/threshold')
        starts, _ = decode(margins[selected['sigma']], selected['sigma'], selected['threshold'])
        equal(statistics(truth[dev], starts[dev]), fold['dev'], f'{year}/dev')
        for name, predicted, key in [('aft', starts, 'test'), ('raw', raw_start, 'raw_test'), ('buffered', buffered_start, 'buffered_test')]:
            equal(statistics(truth[test], predicted[test]), fold[key], f'{year}/{key}')
            pooled[name].append(predicted[test])
        actual_pool.append(truth[test])
        print(f'Independent binned-AFT fold {year}: passed', flush=True)
        del x, train_matrix, evaluation
    assert len(model_names) == 9 and model_names == set(report['models'])
    actual = np.concatenate(actual_pool)
    for name, values in pooled.items():
        equal(statistics(actual, np.concatenate(values)), report['pooled_test'][name], f'pooled/{name}')
    candidate, control = report['pooled_test']['aft'], report['pooled_test']['buffered']
    gate = dict(timing_at_least_10pct_better=candidate['capped_min_T_20_mae_hours'] <= .9 * control['capped_min_T_20_mae_hours'],
        recall_at_least_5pp_better=candidate['recall'] >= control['recall'] + .05,
        pooled_fpr_at_most_5pct=candidate['fpr'] <= .05,
        all_years_fpr_at_most_5pct=all(fold['test']['fpr'] <= .05 for fold in report['folds'].values()),
        no_year_timing_worse=all(fold['test']['capped_min_T_20_mae_hours'] <= fold['buffered_test']['capped_min_T_20_mae_hours'] for fold in report['folds'].values()))
    gate['passed'] = all(gate.values())
    equal(gate, report['clear_improvement_gate'], 'clear_improvement_gate')
    audit = dict(status='passed', independently_reinferred_models=9,
        independently_retrained_models=retraining, independently_scored_candidates=candidate_count,
        independently_computed_eligibility=population, clear_improvement_gate=gate,
        training_feature_builder_imported=False, report_sha256=digest(run / 'report.json'),
        verifier_sha256=digest(Path(__file__)), checks=[
            'Frozen input, protocol, source and model hashes',
            'Independent complete-window population and purged annual folds',
            'Independent lag-one feature construction and issue-hour perturbation',
            'Independent integer recovery bins and right-censoring bounds',
            'All nine saved-model inference and exact training replays',
            'Default AFT prediction equals exponentiated margin',
            'All 189 development candidates and selected scales and thresholds',
            'Annual and pooled metrics and fixed improvement gate'], limits=[
            'Model-based probabilities are not proven calibrated',
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
