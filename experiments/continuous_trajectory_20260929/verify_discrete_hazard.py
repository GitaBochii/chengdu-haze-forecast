"""Independent shared discrete hazard risk-set, refitting and scoring audit."""
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
THRESHOLDS = [i / 20 for i in range(1, 21)] + [1.000001]
PARAMS = dict(objective='binary', n_estimators=300, learning_rate=.04,
    num_leaves=24, min_child_samples=30, random_state=42, n_jobs=4,
    verbosity=-1, deterministic=True, force_col_wise=True)


def risk_set(starts):
    starts = np.asarray(starts)
    assert starts.ndim == 1 and ((starts >= 1) & (starts <= 20)).all()
    rows, leads, events = [], [], []
    for row, start in enumerate(starts):
        for lead in range(1, min(int(start), 19) + 1):
            rows.append(row)
            leads.append(lead)
            events.append(start < 20 and lead == start)
    return np.asarray(rows, dtype=np.int64), np.asarray(leads, dtype=np.int64), np.asarray(events, dtype=bool)


def decode(hazards, threshold):
    hazards = np.asarray(hazards, dtype=float)
    assert hazards.ndim == 2 and hazards.shape[1] == 19 and np.isfinite(hazards).all()
    assert ((hazards >= 0) & (hazards <= 1)).all()
    survival = np.ones(len(hazards), dtype=float)
    cumulative = np.empty_like(hazards)
    for lead in range(19):
        survival *= 1 - hazards[:, lead]
        cumulative[:, lead] = 1 - survival
    assert (np.diff(cumulative, axis=1) >= 0).all()
    total = cumulative[:, -1]
    median = first_hit(cumulative >= .5)
    median[total < threshold] = 20
    return median, total


def all_hazards(model, features):
    predictions = []
    for lead in range(1, 20):
        augmented = np.column_stack((features, np.full(len(features), lead, dtype=np.float32)))
        predictions.append(model.predict(augmented, num_threads=4))
    return np.column_stack(predictions)


def main(run):
    report = json.loads((run / 'report.json').read_text('utf-8'))
    assert report['status'].startswith('completed'), report['status']
    assert Path(report['source_run']).resolve() == SOURCE_RUN.resolve()
    assert Path(report['buffer_run']).resolve() == BUFFER_RUN.resolve()
    assert report['lag'] == 1 and report['feature_dimension'] == 324
    assert report['base_feature_dimension'] == 323
    equal(PARAMS, report['params'], 'fixed_params')
    equal(THRESHOLDS, report['thresholds'], 'fixed_thresholds')
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for path, expected in report['input_sha256'].items():
        assert digest(path) == expected, path
    for section in ('artifact_sha256', 'models'):
        for filename, expected in report[section].items():
            assert digest(run / filename) == expected, filename
    assert digest(HERE / 'discrete_hazard.py') == report['code_sha256']
    assert digest(HERE / 'DISCRETE_HAZARD_PROTOCOL.md') == report['protocol_sha256']
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
    pooled = {name: [] for name in ('hazard', 'raw', 'buffered')}
    candidate_count = 0
    risk_counts = {}
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
        training_meta = pd.read_parquet(run / f'{year}_train_origins.parquet')
        assert list(training_meta.columns) == ['source_index', 'timestamp', 'true_start']
        np.testing.assert_array_equal(training_meta.source_index, ti)
        np.testing.assert_array_equal(pd.DatetimeIndex(training_meta.timestamp).asi8, dates[ti].asi8)
        np.testing.assert_array_equal(training_meta.true_start, training_truth)
        rows, leads, events = risk_set(training_truth)
        risk = pd.read_parquet(run / f'{year}_train_risk.parquet')
        assert list(risk.columns) == ['source_index', 'lead', 'event']
        np.testing.assert_array_equal(risk.source_index, ti[rows])
        np.testing.assert_array_equal(risk.lead, leads)
        np.testing.assert_array_equal(risk.event, events)
        assert risk.event.dtype == bool
        assert len(rows) == np.minimum(training_truth, 19).sum()
        assert events.sum() == (training_truth < 20).sum()
        risk_counts[year] = dict(rows=len(rows), events=int(events.sum()),
            censored_origins=int((training_truth == 20).sum()))
        fold = report['folds'][year]
        assert fold['train_end_year'] == cutoff and fold['train_n'] == train.sum()
        assert fold['train_risk_n'] == len(rows)
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
        filename = f'{year}_hazard.txt'
        model_names.add(filename)
        model = lgb.Booster(model_str=(run / filename).read_text('utf-8'))
        assert model.num_feature() == 324 and model.current_iteration() == 300
        assert model.params['objective'] == 'binary'
        assert model.params['learning_rate'] == .04 and model.params['num_leaves'] == 24
        assert model.params['min_data_in_leaf'] == 30
        assert not model.params.get('is_unbalance', False)
        assert model.params.get('scale_pos_weight', 1) == 1
        hazard = np.load(run / f'{year}_hazards.npy')
        assert hazard.dtype == np.float64 and hazard.shape == (len(ei), 19)
        np.testing.assert_array_equal(all_hazards(model, x[ev]), hazard)
        replay = lgb.LGBMClassifier(**PARAMS)
        person_period = np.column_stack((x[train][rows], leads.astype(np.float32)))
        replay.fit(person_period, events.astype(np.int8))
        np.testing.assert_array_equal(all_hazards(replay.booster_, x[ev]), hazard)
        retraining.append(dict(year=int(year), exact_all_19_hazards_match=True))
        curve = np.load(SOURCE_RUN / f'{year}_lag1_chemistry.npy')
        raw_start = first_hit(window_labels(curve))
        buffer_delta = buffered['folds'][year]['lag1_chemistry']['delta']
        buffered_start = first_hit(window_labels(curve.astype(float) + buffer_delta))
        dev = (meta.partition.to_numpy() == 'dev') & (origin > 100)
        test = (meta.partition.to_numpy() == 'test') & (origin > 100)
        raw_dev = statistics(truth[dev], raw_start[dev])
        grid = []
        assert len(fold['development_grid']) == len(THRESHOLDS)
        for threshold in THRESHOLDS:
            starts, _ = decode(hazard, threshold)
            metric = statistics(truth[dev], starts[dev])
            feasible = metric['fpr'] <= .05 and metric['premature_start_fraction_all'] <= raw_dev['premature_start_fraction_all']
            grid.append(dict(threshold=threshold, metrics=metric, feasible=feasible))
        for index, (actual, saved) in enumerate(zip(grid, fold['development_grid'])):
            equal(actual, saved, f'{year}/candidate/{index}')
            candidate_count += 1
        selected = min([row for row in grid if row['feasible']], key=lambda row:
            (row['metrics']['capped_min_T_20_mae_hours'], -(row['metrics']['recall'] or 0),
                row['metrics']['fpr'], -row['threshold']))
        equal(selected['threshold'], fold['threshold'], f'{year}/threshold')
        starts, _ = decode(hazard, selected['threshold'])
        equal(statistics(truth[dev], starts[dev]), fold['dev'], f'{year}/dev')
        for name, predicted, key in [('hazard', starts, 'test'), ('raw', raw_start, 'raw_test'), ('buffered', buffered_start, 'buffered_test')]:
            equal(statistics(truth[test], predicted[test]), fold[key], f'{year}/{key}')
            pooled[name].append(predicted[test])
        actual_pool.append(truth[test])
        print(f'Independent shared-hazard fold {year}: passed', flush=True)
        del x, person_period
    assert len(model_names) == 3 and model_names == set(report['models'])
    actual = np.concatenate(actual_pool)
    for name, values in pooled.items():
        equal(statistics(actual, np.concatenate(values)), report['pooled_test'][name], f'pooled/{name}')
    candidate, control = report['pooled_test']['hazard'], report['pooled_test']['buffered']
    gate = dict(timing_at_least_10pct_better=candidate['capped_min_T_20_mae_hours'] <= .9 * control['capped_min_T_20_mae_hours'],
        recall_at_least_5pp_better=candidate['recall'] >= control['recall'] + .05,
        pooled_fpr_at_most_5pct=candidate['fpr'] <= .05,
        all_years_fpr_at_most_5pct=all(fold['test']['fpr'] <= .05 for fold in report['folds'].values()),
        no_year_timing_worse=all(fold['test']['capped_min_T_20_mae_hours'] <= fold['buffered_test']['capped_min_T_20_mae_hours'] for fold in report['folds'].values()))
    gate['passed'] = all(gate.values())
    equal(gate, report['clear_improvement_gate'], 'clear_improvement_gate')
    audit = dict(status='passed', independently_reinferred_models=3,
        independently_retrained_models=retraining, independently_scored_candidates=candidate_count,
        independently_computed_eligibility=population, independently_computed_risk_sets=risk_counts,
        clear_improvement_gate=gate, training_feature_builder_imported=False,
        report_sha256=digest(run / 'report.json'), verifier_sha256=digest(Path(__file__)), checks=[
            'Frozen input, code, protocol and model hashes',
            'Independent complete-window population and purged annual folds',
            'Independent lag-one chemistry features and issue-hour perturbation',
            'Independent all-origin person-period risk rows, lead and event labels',
            'All three model replays and full refits match all 19 hazard predictions',
            'Independent monotone recovery CDF and unconditional capped-time median',
            'All 63 development candidates and selected thresholds',
            'Annual and pooled metrics and fixed improvement gate'], limits=[
            'Model-based hazard probabilities are not proven calibrated',
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
