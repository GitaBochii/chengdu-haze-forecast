"""Independent delayed chemistry features, saved predictions and diagnostics."""
import argparse
import json
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from verify_issue_delay import digest, equal, independent_features, diagnostics

HERE = Path(__file__).resolve().parent
BASE = HERE/'issue_delay_runs/20260929T133825296179Z'
LEADS = (1, 6, 24)
SPECIES = ['so2', 'no2', 'co', 'o3_8h']


def reconstruct(raw, dates, indices, cutoff, lag):
    primary = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    primary[(primary < 0) | ~np.isfinite(primary)] = np.nan
    base = independent_features(primary, dates, indices, cutoff, lag, True)
    chemistry = raw[SPECIES].to_numpy(float)
    chemistry[(chemistry < 0) | ~np.isfinite(chemistry)] = np.nan
    positions = indices[:, None] - lag - np.arange(23, -1, -1)
    safe = positions.clip(0, len(dates)-1)
    years = dates.year.to_numpy()
    allowed = (positions >= 0) & (positions < len(dates))
    allowed &= ((years[indices, None] <= cutoff) & (years[safe] <= cutoff)) | (
        (years[indices, None] > cutoff) & (years[safe] == years[indices, None]))
    history = chemistry[safe].copy()
    history[~allowed] = np.nan
    summary = [history[:, 23] - history[:, 23-k] for k in (1, 3, 6, 12, 23)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for width in (3, 6, 12, 24):
            subset = history[:, 24-width:]
            mean = np.nanmean(subset, axis=1)
            summary += [mean, np.nanstd(subset, axis=1),
                        np.nanmax(subset, axis=1)-np.nanmin(subset, axis=1), history[:, 23]-mean]
    coverage = np.isfinite(history).mean(axis=1)
    x = np.hstack([base, history.reshape(len(indices), 96), np.hstack(summary), coverage]).astype(np.float32)
    return x, history


def paired(y, candidate, control, origin):
    result = {}
    for label, mask in [('all', np.ones(len(y), bool)), ('clean_origin', origin <= 100),
                        ('polluted_origin', origin > 100)]:
        a = np.abs(candidate[mask].astype(float)-y[mask].astype(float))
        b = np.abs(control[mask].astype(float)-y[mask].astype(float))
        d = a-b
        result[label] = dict(origins=int(mask.sum()), mae_difference=float(d.mean()),
            relative_mae_reduction=float(-d.mean()/b.mean()),
            leads={f'h{lead}': dict(mae_difference=float(d[:, j].mean()),
                relative_mae_reduction=float(-d[:, j].mean()/b[:, j].mean())) for j, lead in enumerate(LEADS)})
    return result


def main(run):
    report = json.loads((run/'report.json').read_text('utf-8'))
    old = json.loads((BASE/'report.json').read_text('utf-8'))
    assert report['status'] == 'completed_fixed_protocol'
    assert report['leads'] == list(LEADS) and report['channels'] == SPECIES
    assert report['feature_dimensions'] == 323 and report['units'] == 'AQI'
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for metadata, directory in [(old, BASE), (report, run)]:
        for section in ('source_sha256', 'artifact_sha256', 'models'):
            for path, sha in metadata[section].items():
                assert digest(Path(path) if section == 'source_sha256' else directory/path) == sha, path
    source = next(Path(p) for p in report['source_sha256'] if p.endswith('city_hourly.parquet'))
    raw = pd.read_parquet(source)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    assert dates.is_unique and (np.diff(dates.asi8) == 3600*10**9).all()
    core = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    invalid = ((core < 0) | ~np.isfinite(core)).any(axis=1)
    cumulative = np.r_[0, np.cumsum(invalid)]
    all_indices = np.arange(23, len(raw)-24)
    complete = cumulative[all_indices+25] == cumulative[all_indices-23]
    years = dates.year.to_numpy()
    split = np.select([years <= 2023, years == 2024, years == 2025], [0, 1, 2], default=3)
    same = (split[all_indices-23] == split[all_indices]) & (split[all_indices+24] == split[all_indices])
    indices = all_indices[complete & same]
    eligibility = dict(source_rows=len(raw), edge_rows=len(raw)-len(all_indices),
        invalid_window_rows=int((~complete).sum()), split_boundary_rows_with_complete_data=int((complete & ~same).sum()),
        eligible=len(indices))
    equal(eligibility, report['eligibility'], 'eligibility')
    equal(eligibility, old['eligibility'], 'old_eligibility')
    pooled = {f'lag{lag}_chemistry': [] for lag in (1, 3)}
    model_count = 0
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
        pd.testing.assert_frame_equal(meta, pd.read_parquet(BASE/f'{key}_origins.parquet'))
        for name in (f'{key}_origins.parquet', f'{key}_targets.npy'):
            assert digest(run/name) == digest(BASE/name), name
        np.testing.assert_array_equal(meta.source_index, ei)
        np.testing.assert_array_equal(pd.DatetimeIndex(meta.timestamp).asi8, dates[ei].asi8)
        np.testing.assert_array_equal(meta.partition, part[ev])
        origin = core[ei, 2].astype(np.float32)
        np.testing.assert_array_equal(meta.origin_aqi, origin)
        target = core[ei[:, None]+np.array(LEADS), 2].astype(np.float32)
        np.testing.assert_array_equal(target, np.load(run/f'{key}_targets.npy'))
        fold = report['folds'][key]
        assert fold['train_end_year'] == cutoff and fold['train_n'] == train.sum() == old['folds'][key]['train_n']
        for lag in (1, 3):
            name = f'lag{lag}_chemistry'
            x, history = reconstruct(raw, dates, indices, cutoff, lag)
            assert x.shape == (len(indices), 323) and not np.isinf(x).any()
            pred = np.load(run/f'{key}_{name}.npy')
            assert pred.shape == target.shape and np.isfinite(pred).all() and (pred >= 0).all()
            for j, lead in enumerate(LEADS):
                model = lgb.Booster(model_str=(run/f'{key}_{name}_h{lead}.txt').read_text('utf-8'))
                assert model.num_feature() == 323 and model.current_iteration() == 300
                assert model.params['objective'] == 'regression_l1'
                actual = model.predict(x[ev], num_threads=4).clip(min=0).astype(np.float32)
                np.testing.assert_array_equal(actual, pred[:, j])
                model_count += 1
            record = fold['arms'][name]
            coverage = dict(zip(SPECIES, np.isfinite(history[train]).mean(axis=(0, 1)).tolist()))
            equal(coverage, record['train_extra_coverage'], f'{key}/{name}/traincoverage')
            controls = {kind: np.load(BASE/f'{key}_lag{lag}_{kind}.npy') for kind in ('base', 'trend')}
            for partition in ('dev', 'test'):
                m = meta.partition.eq(partition).to_numpy()
                equal(diagnostics(target[m], pred[m], origin[m]), record[partition]['diagnostics'], f'{key}/{name}/{partition}')
                coverage = dict(zip(SPECIES, np.isfinite(history[ev][m]).mean(axis=(0, 1)).tolist()))
                equal(coverage, record[partition]['extra_coverage'], 'evalcoverage')
                for kind, control in controls.items():
                    equal(paired(target[m], pred[m], control[m], origin[m]), record[partition]['comparisons'][kind], 'paired')
                    equal(diagnostics(target[m], control[m], origin[m]), old['folds'][key]['arms'][f'lag{lag}_{kind}'][partition], 'old_control')
                if partition == 'test':
                    pooled[name].append((target[m], pred[m], origin[m], controls['base'][m], controls['trend'][m]))
            del x, history
        print(f'Independent chemistry fold {key}: passed', flush=True)
    for name, rows in pooled.items():
        y, p, o, b, t = [np.concatenate([row[j] for row in rows]) for j in range(5)]
        equal(diagnostics(y, p, o), report['pooled_test'][name], f'pooled/{name}')
        for kind, control in [('base', b), ('trend', t)]:
            equal(paired(y, p, control, o), report['paired_comparisons'][name][kind], 'pooledpaired')
    assert model_count == 18 and len(report['models']) == 18
    audit = dict(status='passed', independently_reinferred_models=model_count, feature_dimensions=323,
        training_feature_builder_imported=False,
        checks=['source and old/new artifact hashes', 'identical origin and label artifact hashes',
                'independent eligibility and temporal partition masking', 'all inputs uniformly delayed',
                'independent 139 base and 184 chemistry feature reconstruction',
                '18 complete saved-model inference reproductions', 'train/dev/test extra coverage',
                'all dev/test/pooled and origin subgroup MAE/RMSE', 'paired comparisons with both frozen controls',
                'code-reviewed all-conditions training without origin AQI routing'],
        limits=['Historical delay assumptions, not verified receipt times', 'Not a blind or prospective test',
                'No independent retraining', 'AQI errors, not dissipation hours',
                'Origin AQI used for retrospective diagnostics only'],
        verifier_sha256=digest(Path(__file__)), report_sha256=digest(run/'report.json'))
    (run/'independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n', 'utf-8')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
