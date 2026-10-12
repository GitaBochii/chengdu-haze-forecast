"""Independent weighted-window inputs, training replay and decision audit."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from verify_issue_delay import digest, equal
from verify_delayed_chemistry import reconstruct
from verify_recovery_windows import window_labels, first_hit, statistics

HERE = Path(__file__).resolve().parent
SOURCE_RUN = HERE / 'full_curve_runs/20260930T033524196780Z'
BUFFER_RUN = HERE / 'recovery_calibration_runs/20260930T121425927928Z'
UNIFORM_RUN = HERE / 'recovery_window_runs/20260930T122334448606Z'


def independently_weight(last):
    last = np.asarray(last, dtype=float)
    valid = np.isfinite(last) & (last >= 0)
    clean = valid & (last <= 100)
    polluted = valid & (last > 100)
    assert clean.any() and polluted.any()
    weight = np.ones(len(last), dtype=float)
    weight[clean] = valid.sum() / (2 * clean.sum())
    weight[polluted] = valid.sum() / (2 * polluted.sum())
    np.testing.assert_allclose(weight.mean(), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(weight[clean].sum(), weight[polluted].sum(), rtol=0, atol=1e-9)
    return weight, dict(origins=len(last), valid=int(valid.sum()), clean=int(clean.sum()),
        polluted=int(polluted.sum()), missing_or_negative=int((~valid).sum()),
        sum=float(weight.sum()), mean=float(weight.mean()), clean_mass=float(weight[clean].sum()),
        polluted_mass=float(weight[polluted].sum()))


def qualifying_statistics(target, probability, predicted, origin):
    labels = window_labels(target)
    truth = first_hit(labels)
    result = {}
    for name, mask in [('all', np.ones(len(target), bool)),
            ('clean_origin', origin <= 100), ('polluted_origin', origin > 100)]:
        recovered = truth[mask] != 20
        score = np.max(probability[mask], axis=1)
        result[name] = dict(origins=int(mask.sum()),
            window_brier=float(np.mean((probability[mask] - labels[mask]) ** 2)),
            any_window_auc=float(roc_auc_score(recovered, score)) if len(np.unique(recovered)) == 2 else None,
            qualifying_window_metrics=statistics(truth[mask], predicted[mask]))
    return result


def main(run):
    report = json.loads((run / 'report.json').read_text('utf-8'))
    assert report['status'].startswith('completed'), report['status']
    assert Path(report['source_run']).resolve() == SOURCE_RUN.resolve()
    assert Path(report['uniform_run']).resolve() == UNIFORM_RUN.resolve()
    assert report['lag'] == 1 and report['feature_dimension'] == 323
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for path, expected in report['input_sha256'].items():
        assert digest(path) == expected, path
    for section in ('artifact_sha256', 'models'):
        for filename, expected in report[section].items():
            assert digest(run / filename) == expected, filename
    assert digest(HERE / 'weighted_recovery_windows.py') == report['code_sha256']
    assert digest(HERE / 'WEIGHTED_RECOVERY_WINDOWS_PROTOCOL.md') == report['protocol_sha256']
    reference = json.loads((SOURCE_RUN / 'report.json').read_text('utf-8'))
    buffered = json.loads((BUFFER_RUN / 'report.json').read_text('utf-8'))
    uniform = json.loads((UNIFORM_RUN / 'report.json').read_text('utf-8'))
    assert json.loads((UNIFORM_RUN / 'independent_audit.json').read_text('utf-8'))['status'] == 'passed'
    for section in ('artifact_sha256', 'models'):
        for filename, expected in uniform[section].items():
            assert digest(UNIFORM_RUN / filename) == expected, filename
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
    population = dict(source_rows=len(raw), edge_rows=len(raw)-len(possible),
        invalid_window_rows=int((~complete).sum()),
        split_boundary_rows_with_complete_data=int((complete & ~same).sum()), eligible=len(indices))
    pooled = {name: [] for name in ('weighted', 'uniform', 'raw', 'buffered')}
    qualifying_pool = {name: [] for name in ('target', 'weighted', 'uniform', 'weighted_start', 'uniform_start', 'origin')}
    actual_pool, model_names, retraining, weights_summary = [], set(), [], {}
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
        train_target = core[ti[:, None] + np.arange(1, 25), 2].astype(np.float32)
        labels = window_labels(train_target)
        np.testing.assert_array_equal(labels, np.load(run / f'{year}_train_labels.npy'))
        fold = report['folds'][year]
        assert fold['train_end_year'] == cutoff and fold['train_n'] == train.sum()
        assert fold['train_clean'] == np.count_nonzero(train & (core[indices, 2] <= 100))
        assert fold['train_polluted'] == np.count_nonzero(train & (core[indices, 2] > 100))
        x, _ = reconstruct(raw, dates, indices, cutoff, 1)
        assert x.shape == (len(indices), 323) and not np.isinf(x).any()
        last = x[train, 71]
        raw_last = core[ti - 1, 2].astype(np.float32)
        raw_last[(~np.isfinite(raw_last)) | (raw_last < 0)] = np.nan
        np.testing.assert_array_equal(last, raw_last)
        weight, summary = independently_weight(last)
        weights_summary[year] = summary
        valid_last = np.isfinite(last) & (last >= 0)
        expected_info = dict(clean_count=summary['clean'], polluted_count=summary['polluted'],
            finite_count=summary['valid'], invalid_count=summary['missing_or_negative'],
            clean_weight_mass=summary['clean_mass'], polluted_weight_mass=summary['polluted_mass'],
            mean_weight=summary['mean'], clean_weight=float(weight[valid_last & (last <= 100)][0]),
            polluted_weight=float(weight[valid_last & (last > 100)][0]))
        equal(expected_info, fold['weights'], f'{year}/weights')
        saved_weights = pd.read_parquet(run / f'{year}_train_weights.parquet')
        np.testing.assert_array_equal(saved_weights.source_index, ti)
        np.testing.assert_array_equal(saved_weights.last_observed_aqi, last)
        np.testing.assert_array_equal(saved_weights.weight, weight)
        sample = ei[::100]
        altered = raw.copy()
        for column in ('pm2_5', 'pm10', 'aqi', 'so2', 'no2', 'co', 'o3_8h'):
            altered.loc[altered.index[sample], column] = 99999
        check, _ = reconstruct(altered, dates, sample, cutoff, 1)
        original, _ = reconstruct(raw, dates, sample, cutoff, 1)
        np.testing.assert_array_equal(check, original)
        probability = np.load(run / f'{year}_probabilities.npy')
        assert probability.shape == (len(ei), 19) and probability.dtype == np.float64
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        for j in range(19):
            filename = f'{year}_window{j+1}.txt'
            model_names.add(filename)
            model = lgb.Booster(model_str=(run / filename).read_text('utf-8'))
            assert model.num_feature() == 323 and model.current_iteration() == 300
            assert model.params['objective'] == 'binary'
            assert model.params['learning_rate'] == .04 and model.params['num_leaves'] == 24
            assert model.params['min_data_in_leaf'] == 30 and not model.params.get('is_unbalance', False)
            assert model.params.get('scale_pos_weight', 1) == 1
            np.testing.assert_array_equal(model.predict(x[ev], num_threads=4), probability[:, j])
        replay = lgb.LGBMClassifier(objective='binary', n_estimators=300, learning_rate=.04,
            num_leaves=24, min_child_samples=30, random_state=42, n_jobs=4,
            verbosity=-1, deterministic=True, force_col_wise=True)
        replay.fit(x[train], labels[:, 0], sample_weight=weight)
        np.testing.assert_array_equal(replay.predict_proba(x[ev])[:, 1], probability[:, 0])
        retraining.append(dict(year=int(year), window=1, exact_prediction_match=True))
        raw_curve = np.load(SOURCE_RUN / f'{year}_lag1_chemistry.npy')
        raw_start = first_hit(window_labels(raw_curve))
        delta = buffered['folds'][year]['lag1_chemistry']['delta']
        buffer_start = first_hit(window_labels(raw_curve.astype(float) + delta))
        uniform_probability = np.load(UNIFORM_RUN / f'{year}_probabilities.npy')
        uniform_threshold = uniform['folds'][year]['threshold']
        uniform_start = first_hit(uniform_probability >= uniform_threshold)
        dev = (meta.partition.to_numpy() == 'dev') & (origin > 100)
        test = (meta.partition.to_numpy() == 'test') & (origin > 100)
        raw_dev = statistics(true[dev], raw_start[dev])
        grid = list(np.arange(1, 21, dtype=float) / 20) + [1.000001]
        rows = []
        assert len(fold['development_grid']) == len(grid)
        for threshold, saved in zip(grid, fold['development_grid']):
            metric = statistics(true[dev], first_hit(probability[dev] >= threshold))
            feasible = (metric['fpr'] is not None and metric['fpr'] <= .05 and
                metric['premature_start_fraction_all'] <= raw_dev['premature_start_fraction_all'])
            row = dict(threshold=threshold, metrics=metric, feasible=feasible)
            equal(row, saved, f'{year}/grid/{threshold}')
            rows.append(row)
            grid_count += 1
        selected = min([row for row in rows if row['feasible']], key=lambda row:
            (row['metrics']['capped_min_T_20_mae_hours'], -(row['metrics']['recall'] or 0),
                row['metrics']['fpr'], -row['threshold']))
        equal(selected['threshold'], fold['threshold'], f'{year}/selected')
        direct_start = first_hit(probability >= selected['threshold'])
        equal(statistics(true[dev], direct_start[dev]), fold['dev'], f'{year}/dev')
        for name, predicted, key in [('weighted', direct_start, 'test'),
                ('uniform', uniform_start, 'uniform_test'), ('raw', raw_start, 'raw_test'),
                ('buffered', buffer_start, 'buffered_test')]:
            metric = statistics(true[test], predicted[test])
            equal(metric, fold[key], f'{year}/{key}')
            if name == 'uniform':
                equal(metric, uniform['folds'][year]['test'], f'{year}/frozen_uniform')
            pooled[name].append(predicted[test])
        for partition_name in ('dev', 'test'):
            mask = meta.partition.to_numpy() == partition_name
            for arm, arm_probability, arm_start in [('weighted', probability, direct_start),
                    ('uniform', uniform_probability, uniform_start)]:
                equal(qualifying_statistics(target[mask], arm_probability[mask], arm_start[mask], origin[mask]),
                    fold['qualifying_diagnostics'][partition_name][arm], f'{year}/{partition_name}/{arm}/qualifying')
        all_test = meta.partition.to_numpy() == 'test'
        for name, values in dict(target=target, weighted=probability, uniform=uniform_probability,
                weighted_start=direct_start, uniform_start=uniform_start, origin=origin).items():
            qualifying_pool[name].append(values[all_test])
        actual_pool.append(true[test])
        del x
        print(f'Independent weighted-window fold {year}: passed', flush=True)
    assert len(model_names) == 57 and set(report['models']) == model_names
    actual = np.concatenate(actual_pool)
    for name, arrays in pooled.items():
        equal(statistics(actual, np.concatenate(arrays)), report['pooled_test'][name], f'pooled/{name}')
    q = {name: np.concatenate(arrays) for name, arrays in qualifying_pool.items()}
    for arm in ('weighted', 'uniform'):
        equal(qualifying_statistics(q['target'], q[arm], q[arm + '_start'], q['origin']),
            report['pooled_qualifying_diagnostics'][arm], f'pooled/{arm}/qualifying')
    audit = dict(status='passed', independently_reinferred_models=57,
        independently_scored_candidates=grid_count, independently_retrained_models=retraining,
        independently_computed_training_weight_summary=weights_summary,
        independently_computed_eligibility=population,
        training_feature_builder_imported=False, report_sha256=digest(run / 'report.json'),
        verifier_sha256=digest(Path(__file__)), checks=[
            'Frozen inputs including uniform-control artifacts and model hashes',
            'Independent complete-window population and purged yearly splits',
            'Independent lag-one 323-feature reconstruction and issue-hour perturbation',
            'Independent training-only equal-stratum weights and all 57 training labels',
            'All 57 saved-model inference reproductions and fixed model settings',
            'Independent exact weighted retraining of first window in each of three folds',
            'All 63 development candidates, feasibility and selected thresholds',
            'All annual and pooled weighted/uniform/raw/buffered recovery metrics'], limits=[
            'Independent retraining covers 3 of 57 models, not every training invocation',
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
