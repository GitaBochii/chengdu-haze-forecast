"""Independent hourly-curve inference and censored-recovery audit."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from verify_issue_delay import digest, equal, independent_features
from verify_delayed_chemistry import reconstruct


def starts(curves):
    a = np.asarray(curves)
    assert a.ndim == 2 and a.shape[1] == 24 and np.isfinite(a).all()
    result = np.full(len(a), 20, dtype=int)
    for k in range(19):
        hit = np.sum(a[:, k:k+6] <= 100, axis=1) == 6
        result[(result == 20) & hit] = k+1
    return result


def diagnostics(y, p, origin):
    result = {}
    for name, mask in [('all', np.ones(len(origin), bool)),
                       ('clean_origin', origin <= 100), ('polluted_origin', origin > 100)]:
        errors = np.abs(y[mask].astype(np.float64)-p[mask].astype(np.float64))
        result[name] = dict(origins=int(mask.sum()), mae=float(errors.mean()) if len(errors) else None,
                            lead_mae=errors.mean(axis=0).tolist() if len(errors) else None)
    mask = origin > 100
    actual, pred = starts(y[mask]), starts(p[mask])
    yes, hat = actual != 20, pred != 20
    counts = dict(tp=int(np.count_nonzero(yes & hat)), fn=int(np.count_nonzero(yes & ~hat)),
                  fp=int(np.count_nonzero(~yes & hat)), tn=int(np.count_nonzero(~yes & ~hat)))
    ratio = lambda a, b: float(a/b) if b else None
    early = int(np.count_nonzero((pred < actual) & hat))
    result['aqi_recovery'] = dict(origins=len(actual), actual_right_censored=int((~yes).sum()),
        predicted_no_recovery=int((~hat).sum()),
        capped_min_T_20_mae_hours=float(np.abs(actual-pred).mean()) if len(actual) else None,
        confusion=counts, precision=ratio(counts['tp'], counts['tp']+counts['fp']),
        recall=ratio(counts['tp'], counts['tp']+counts['fn']),
        fpr=ratio(counts['fp'], counts['fp']+counts['tn']),
        premature_start_count=early, premature_start_fraction_all=ratio(early, len(actual)),
        premature_start_fraction_predicted_recovery=ratio(early, hat.sum()))
    return result


def main(run):
    report = json.loads((run/'report.json').read_text('utf-8'))
    assert report['status'] == 'completed_fixed_protocol'
    assert report['leads'] == list(range(1, 25)) and report['lag'] == 1
    assert report['feature_dimensions'] == {'lag1_trend': 139, 'lag1_chemistry': 323}
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for section in ('source_sha256', 'artifact_sha256', 'models'):
        for path, expected in report[section].items():
            assert digest(Path(path) if section == 'source_sha256' else run/path) == expected, path
    for name, item in report['reused_models'].items():
        assert digest(run/name) == digest(item['path']) == item['sha256'], name
    source = next(Path(p) for p in report['source_sha256'] if p.endswith('city_hourly.parquet'))
    raw = pd.read_parquet(source)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    assert dates.is_unique and (np.diff(dates.asi8) == 3600*10**9).all()
    core = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    bad = ((core < 0) | ~np.isfinite(core)).any(axis=1)
    sums = np.r_[0, np.cumsum(bad)]
    possible = np.arange(23, len(raw)-24)
    complete = sums[possible+25] == sums[possible-23]
    years = dates.year.to_numpy()
    partition = np.select([years <= 2023, years == 2024, years == 2025], [0, 1, 2], default=3)
    same = (partition[possible-23] == partition[possible]) & (partition[possible+24] == partition[possible])
    indices = possible[complete & same]
    equal(dict(source_rows=len(raw), edge_rows=len(raw)-len(possible),
        invalid_window_rows=int((~complete).sum()),
        split_boundary_rows_with_complete_data=int((complete & ~same).sum()), eligible=len(indices)),
        report['eligibility'], 'eligibility')
    clean = core.copy()
    clean[(clean < 0) | ~np.isfinite(clean)] = np.nan
    pooled = {name: [] for name in report['feature_dimensions']}
    model_count = 0
    expected_names = set()
    for cutoff in (2019, 2020, 2021):
        key = str(cutoff+2)
        lo, hi = years[indices-23], years[indices+24]
        part = np.full(len(indices), 'excluded', dtype='U8')
        part[hi <= cutoff] = 'train'
        part[(lo == cutoff+1) & (hi == cutoff+1)] = 'dev'
        part[(lo == cutoff+2) & (hi == cutoff+2)] = 'test'
        train = part == 'train'
        ev = np.isin(part, ['dev', 'test'])
        ei = indices[ev]
        meta = pd.read_parquet(run/f'{key}_origins.parquet')
        np.testing.assert_array_equal(meta.source_index, ei)
        np.testing.assert_array_equal(pd.DatetimeIndex(meta.timestamp).asi8, dates[ei].asi8)
        np.testing.assert_array_equal(meta.partition, part[ev])
        origin = core[ei, 2].astype(np.float32)
        np.testing.assert_array_equal(meta.origin_aqi, origin)
        target = core[ei[:, None]+np.arange(1, 25), 2].astype(np.float32)
        np.testing.assert_array_equal(target, np.load(run/f'{key}_targets.npy'))
        fold = report['folds'][key]
        assert fold['train_end_year'] == cutoff and fold['train_n'] == train.sum()
        assert fold['train_clean'] == np.count_nonzero(train & (core[indices, 2] <= 100))
        assert fold['train_polluted'] == np.count_nonzero(train & (core[indices, 2] > 100))
        for name, dim in report['feature_dimensions'].items():
            x = independent_features(clean, dates, indices, cutoff, 1, True) if name == 'lag1_trend' else reconstruct(raw, dates, indices, cutoff, 1)[0]
            assert x.shape == (len(indices), dim) and not np.isinf(x).any()
            pred = np.load(run/f'{key}_{name}.npy')
            assert pred.shape == target.shape and np.isfinite(pred).all() and (pred >= 0).all()
            for j in range(24):
                filename = f'{key}_{name}_h{j+1}.txt'
                expected_names.add(filename)
                model = lgb.Booster(model_str=(run/filename).read_text('utf-8'))
                assert model.num_feature() == dim and model.current_iteration() == 300
                assert model.params['objective'] == 'regression_l1'
                actual = model.predict(x[ev], num_threads=4).clip(min=0).astype(np.float32)
                np.testing.assert_array_equal(actual, pred[:, j])
                assert (filename in report['reused_models']) == (j+1 in (1, 6, 24))
                if filename in report['reused_models']:
                    base = Path(report['reused_models'][filename]['path']).parent
                    pd.testing.assert_frame_equal(meta, pd.read_parquet(base/f'{key}_origins.parquet'))
                    np.testing.assert_array_equal(target[:, [0, 5, 23]], np.load(base/f'{key}_targets.npy'))
                    np.testing.assert_array_equal(actual, np.load(base/f'{key}_{name}.npy')[:, (1, 6, 24).index(j+1)])
                model_count += 1
            for subset in ('dev', 'test'):
                m = meta.partition.eq(subset).to_numpy()
                equal(diagnostics(target[m], pred[m], origin[m]), fold['arms'][name][subset], f'{key}/{name}/{subset}')
                if subset == 'test':
                    pooled[name].append((target[m], pred[m], origin[m]))
            del x
        print(f'Independent full-curve fold {key}: passed', flush=True)
    for name, items in pooled.items():
        y, p, o = [np.concatenate([item[j] for item in items]) for j in range(3)]
        equal(diagnostics(y, p, o), report['pooled_test'][name], f'pooled/{name}')
    assert model_count == 144 and set(report['models']) == expected_names
    assert len(report['reused_models']) == 18
    audit = dict(status='passed', independently_reinferred_models=model_count, reused_models_verified=18,
        training_feature_builder_imported=False, report_sha256=digest(run/'report.json'),
        verifier_sha256=digest(Path(__file__)),
        checks=['all frozen source and artifact hashes', 'independent complete-window eligibility and folds',
            'independent 139 and 323 feature reconstruction', 'all 144 complete saved-model inference reproductions',
            'all 24 labels, origin metadata and clean/polluted training counts',
            '18 original model hash, metadata, target and prediction agreements',
            'all annual dev/test and pooled subgroup MAE and lead MAE',
            'independent six-hour recovery search, censoring, capped MAE, confusion and premature starts'],
        limits=['No independent retraining', 'Historical delay assumption, not audited receipt times',
            'Repeated historical tests, not blind or prospective', '20 is a censoring cap, not observed recovery',
            'AQI recovery is not particulate-primary haze dissipation', 'Overlapping origins are not independent events'])
    (run/'independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
