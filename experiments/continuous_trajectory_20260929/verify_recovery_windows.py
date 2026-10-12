"""Independent window-classifier inference and development-selection audit."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from verify_issue_delay import digest, equal
from verify_delayed_chemistry import reconstruct

HERE = Path(__file__).resolve().parent
SOURCE_RUN = HERE / 'full_curve_runs/20260930T033524196780Z'
BUFFER_RUN = HERE / 'recovery_calibration_runs/20260930T121425927928Z'


def window_labels(curves):
    a = np.asarray(curves)
    assert a.ndim == 2 and a.shape[1] == 24 and np.isfinite(a).all()
    labels = np.empty((len(a), 19), dtype=bool)
    for start in range(19):
        labels[:, start] = np.sum(a[:, start:start+6] <= 100, axis=1) == 6
    return labels


def first_hit(hits):
    result = np.full(len(hits), 20, dtype=np.int64)
    for j in range(19):
        result[(result == 20) & hits[:, j]] = j+1
    return result


def statistics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    yes, hat = actual != 20, predicted != 20
    tp = int(np.count_nonzero(yes & hat))
    fn = int(np.count_nonzero(yes & ~hat))
    fp = int(np.count_nonzero(~yes & hat))
    tn = int(np.count_nonzero(~yes & ~hat))
    early = int(np.count_nonzero(hat & (predicted < actual)))
    ratio = lambda a, b: float(a/b) if b else None
    return dict(origins=len(actual), actual_right_censored=int((~yes).sum()),
        predicted_no_recovery=int((~hat).sum()),
        capped_min_T_20_mae_hours=float(np.abs(actual-predicted).mean()) if len(actual) else None,
        confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn),
        precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn), fpr=ratio(fp, fp+tn),
        premature_start_count=early, premature_start_fraction_all=ratio(early, len(actual)),
        premature_start_fraction_predicted_recovery=ratio(early, hat.sum()))


def main(run):
    report = json.loads((run/'report.json').read_text('utf-8'))
    assert report['status'].startswith('completed'), report['status']
    assert Path(report['source_run']).resolve() == SOURCE_RUN.resolve()
    assert report['lag'] == 1 and report['feature_dimension'] == 323
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for path, expected in report['input_sha256'].items():
        assert digest(path) == expected, path
    for section in ('artifact_sha256', 'models'):
        for filename, expected in report[section].items():
            assert digest(run/filename) == expected, filename
    assert digest(HERE/'recovery_windows.py') == report['code_sha256']
    assert digest(HERE/'RECOVERY_WINDOWS_PROTOCOL.md') == report['protocol_sha256']
    reference = json.loads((SOURCE_RUN/'report.json').read_text('utf-8'))
    buffered = json.loads((BUFFER_RUN/'report.json').read_text('utf-8'))
    source = next(Path(p) for p in reference['source_sha256'] if p.endswith('city_hourly.parquet'))
    assert digest(source) == reference['source_sha256'][str(source)]
    raw = pd.read_parquet(source)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    assert dates.is_unique and (np.diff(dates.asi8) == 3600*10**9).all()
    core = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    bad = ((core < 0) | ~np.isfinite(core)).any(axis=1)
    cumulative = np.r_[0, np.cumsum(bad)]
    possible = np.arange(23, len(raw)-24)
    complete = cumulative[possible+25] == cumulative[possible-23]
    years = dates.year.to_numpy()
    partition = np.select([years <= 2023, years == 2024, years == 2025], [0, 1, 2], default=3)
    same = (partition[possible-23] == partition[possible]) & (partition[possible+24] == partition[possible])
    indices = possible[complete & same]
    pooled = {name: [] for name in ('direct', 'raw', 'buffered')}
    actual_pool = []
    model_names = set()
    grid_count = 0
    for cutoff in (2019, 2020, 2021):
        year = str(cutoff+2)
        lo, hi = years[indices-23], years[indices+24]
        part = np.full(len(indices), 'excluded', dtype='U8')
        part[hi <= cutoff] = 'train'
        part[(lo == cutoff+1) & (hi == cutoff+1)] = 'dev'
        part[(lo == cutoff+2) & (hi == cutoff+2)] = 'test'
        train, ev = part == 'train', np.isin(part, ['dev', 'test'])
        ei = indices[ev]
        meta = pd.read_parquet(run/f'{year}_origins.parquet')
        pd.testing.assert_frame_equal(meta, pd.read_parquet(SOURCE_RUN/f'{year}_origins.parquet'))
        np.testing.assert_array_equal(meta.source_index, ei)
        np.testing.assert_array_equal(pd.DatetimeIndex(meta.timestamp).asi8, dates[ei].asi8)
        np.testing.assert_array_equal(meta.partition, part[ev])
        origin = core[ei, 2].astype(np.float32)
        np.testing.assert_array_equal(meta.origin_aqi, origin)
        target = core[ei[:, None]+np.arange(1, 25), 2].astype(np.float32)
        np.testing.assert_array_equal(target, np.load(run/f'{year}_targets.npy'))
        np.testing.assert_array_equal(target, np.load(SOURCE_RUN/f'{year}_targets.npy'))
        true = first_hit(window_labels(target))
        fold = report['folds'][year]
        assert fold['train_end_year'] == cutoff and fold['train_n'] == train.sum()
        assert fold['train_clean'] == np.count_nonzero(train & (core[indices, 2] <= 100))
        assert fold['train_polluted'] == np.count_nonzero(train & (core[indices, 2] > 100))
        assert fold['train_clean'] > 0 and fold['train_polluted'] > 0
        x, _ = reconstruct(raw, dates, indices, cutoff, 1)
        assert x.shape == (len(indices), 323) and not np.isinf(x).any()
        # Alter the issue-hour observations; each selected origin must retain its inputs.
        sample = ei[::100]
        altered = raw.copy()
        for column in ('pm2_5', 'pm10', 'aqi', 'so2', 'no2', 'co', 'o3_8h'):
            altered.loc[altered.index[sample], column] = 99999
        check, _ = reconstruct(altered, dates, sample, cutoff, 1)
        original, _ = reconstruct(raw, dates, sample, cutoff, 1)
        np.testing.assert_array_equal(check, original)
        probability = np.load(run/f'{year}_probabilities.npy')
        assert probability.shape == (len(ei), 19) and probability.dtype == np.float64
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        for j in range(19):
            filename = f'{year}_window{j+1}.txt'
            model_names.add(filename)
            model = lgb.Booster(model_str=(run/filename).read_text('utf-8'))
            assert model.num_feature() == 323 and model.current_iteration() == 300
            assert model.params['objective'] == 'binary'
            assert model.params['learning_rate'] == .04 and model.params['num_leaves'] == 24
            assert model.params['min_data_in_leaf'] == 30 and not model.params.get('is_unbalance', False)
            assert model.params.get('scale_pos_weight', 1) == 1
            np.testing.assert_array_equal(model.predict(x[ev], num_threads=4), probability[:, j])
        raw_curve = np.load(SOURCE_RUN/f'{year}_lag1_chemistry.npy')
        raw_start = first_hit(window_labels(raw_curve))
        delta = buffered['folds'][year]['lag1_chemistry']['delta']
        buffer_start = first_hit(window_labels(raw_curve.astype(float)+delta))
        dev = (meta.partition.to_numpy() == 'dev') & (origin > 100)
        test = (meta.partition.to_numpy() == 'test') & (origin > 100)
        raw_dev = statistics(true[dev], raw_start[dev])
        grid = list(np.arange(1, 21, dtype=float)/20) + [1.000001]
        rows = []
        assert len(fold['development_grid']) == len(grid)
        for threshold, saved in zip(grid, fold['development_grid']):
            metric = statistics(true[dev], first_hit(probability[dev] >= threshold))
            feasible = metric['fpr'] is not None and metric['fpr'] <= .05 and metric['premature_start_fraction_all'] <= raw_dev['premature_start_fraction_all']
            row = dict(threshold=threshold, metrics=metric, feasible=feasible)
            equal(row, saved, f'{year}/grid/{threshold}')
            rows.append(row)
            grid_count += 1
        eligible = [row for row in rows if row['feasible']]
        selected = min(eligible, key=lambda row: (row['metrics']['capped_min_T_20_mae_hours'],
            -(row['metrics']['recall'] or 0), row['metrics']['fpr'], -row['threshold']))
        equal(selected['threshold'], fold['threshold'], f'{year}/selected')
        direct_start = first_hit(probability >= selected['threshold'])
        equal(statistics(true[dev], direct_start[dev]), fold['dev'], f'{year}/dev')
        for name, predicted, key in [('direct', direct_start, 'test'), ('raw', raw_start, 'raw_test'), ('buffered', buffer_start, 'buffered_test')]:
            equal(statistics(true[test], predicted[test]), fold[key], f'{year}/{key}')
            pooled[name].append(predicted[test])
        actual_pool.append(true[test])
        del x
        print(f'Independent recovery-window fold {year}: passed', flush=True)
    assert len(model_names) == 57 and set(report['models']) == model_names
    actual = np.concatenate(actual_pool)
    for name, arrays in pooled.items():
        equal(statistics(actual, np.concatenate(arrays)), report['pooled_test'][name], f'pooled/{name}')
    audit = dict(status='passed', independently_reinferred_models=57, independently_scored_candidates=grid_count,
        training_feature_builder_imported=False, report_sha256=digest(run/'report.json'),
        verifier_sha256=digest(Path(__file__)), checks=[
            'Frozen inputs, saved artifacts, model, code and protocol hashes',
            'Independent complete-window population and purged yearly splits',
            'All-condition clean and polluted training counts',
            'Independent lag-one 323-feature reconstruction and issue-hour perturbation exclusion',
            'All 57 binary saved-model inference reproductions and fixed model settings',
            'Independent six-hour window labels and earliest recovery or censoring cap',
            'All 63 development candidates, feasibility and selected thresholds',
            'All annual and pooled direct/raw/buffered recovery metrics'], limits=[
            'No independent retraining or proof of labels supplied to fit',
            'Lag-one receipt availability remains a historical assumption',
            'Repeated historical tests, not blind or prospective',
            '20 represents a censoring cap, not observed recovery',
            'AQI recovery is not particulate-primary haze dissipation',
            'Overlapping origins are not independent events'])
    (run/'independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
