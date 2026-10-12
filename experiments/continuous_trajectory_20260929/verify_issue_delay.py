"""Independent feature, target, saved-model and metric reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEADS = (1, 6, 24)
ARMS = [('lag0_base', 0, False), ('lag1_base', 1, False),
        ('lag1_trend', 1, True), ('lag3_base', 3, False), ('lag3_trend', 3, True)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key in expected:
            equal(actual[key], expected[key], label+'/'+key)
    elif isinstance(expected, float):
        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def independent_features(values, dates, indices, cutoff, lag, trend):
    positions = indices[:, None] - lag - np.arange(23, -1, -1)
    safe = np.maximum(0, np.minimum(len(dates)-1, positions))
    yrs = dates.year.to_numpy()
    allowed = (positions >= 0) & (positions < len(dates))
    allowed &= ((yrs[indices, None] <= cutoff) & (yrs[safe] <= cutoff)) | (
        (yrs[indices, None] > cutoff) & (yrs[safe] == yrs[indices, None]))
    history = values[safe].copy()
    history[~allowed] = np.nan
    issue = dates[indices]
    clock = np.array([np.sin(2*np.pi*issue.hour/24), np.cos(2*np.pi*issue.hour/24),
                      np.sin(2*np.pi*issue.dayofyear/366), np.cos(2*np.pi*issue.dayofyear/366)]).T
    base = np.hstack((history.astype(np.float32).reshape(len(indices), -1), clock)).astype(np.float32)
    if not trend:
        return base
    extra = [history[:, 23] - history[:, 23-k] for k in (1, 3, 6, 12, 23)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for width in (3, 6, 12, 24):
            section = history[:, 24-width:]
            mean = np.nanmean(section, axis=1)
            extra += [mean, np.nanstd(section, axis=1),
                      np.nanmax(section, axis=1)-np.nanmin(section, axis=1), history[:, 23]-mean]
    return np.hstack((base, np.concatenate(extra, axis=1).astype(np.float32))).astype(np.float32)


def metrics(y, p):
    keep = np.isfinite(p).all(axis=1)
    result = dict(origins=len(y), scored_origins=int(keep.sum()),
                  coverage=float(keep.mean()) if len(keep) else None)
    if not keep.any():
        return dict(result, mae=None, rmse=None, leads={})
    error = p[keep].astype(np.float64)-y[keep].astype(np.float64)
    return dict(result, mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error**2).mean())),
                leads={f'h{h}': dict(mae=float(np.abs(error[:, j]).mean()),
                       rmse=float(np.sqrt((error[:, j]**2).mean()))) for j, h in enumerate(LEADS)})


def diagnostics(y, p, origin):
    return {name: metrics(y[m], p[m]) for name, m in [
        ('all', np.ones(len(y), bool)), ('clean_origin', origin <= 100), ('polluted_origin', origin > 100)]}


def main(run):
    report = json.loads((run/'report.json').read_text('utf-8'))
    assert report['status'] == 'completed_fixed_protocol'
    assert report['leads'] == list(LEADS)
    assert report['historical_stress_test'] and not any(report[k] for k in ('blind_test', 'prospective', 'deployed'))
    for section in ('source_sha256', 'artifact_sha256', 'models'):
        for path, sha in report[section].items():
            assert digest(Path(path) if section == 'source_sha256' else run/path) == sha, path
    source = next(Path(p) for p in report['source_sha256'] if p.endswith('city_hourly.parquet'))
    raw = pd.read_parquet(source)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    assert dates.is_unique and (np.diff(dates.asi8) == 3600*10**9).all()
    original = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(np.float64)
    invalid = (~np.isfinite(original) | (original < 0)).any(axis=1)
    counts = np.r_[0, np.cumsum(invalid)]
    all_idx = np.arange(23, len(raw)-24)
    complete = counts[all_idx+25] == counts[all_idx-23]
    year = dates.year.to_numpy()
    partition = np.where(year <= 2023, 0, np.where(year == 2024, 1, np.where(year == 2025, 2, 3)))
    same = (partition[all_idx-23] == partition[all_idx]) & (partition[all_idx+24] == partition[all_idx])
    idx = all_idx[complete & same]
    equal(dict(source_rows=len(raw), edge_rows=len(raw)-len(all_idx), invalid_window_rows=int((~complete).sum()),
               split_boundary_rows_with_complete_data=int((complete & ~same).sum()), eligible=len(idx)),
          report['eligibility'], 'eligibility')
    values = original.copy()
    values[~np.isfinite(values) | (values < 0)] = np.nan
    pool = {name: [] for name, _, _ in ARMS}
    pool.update({f'persistence_lag{lag}': [] for lag in (0, 1, 3)})
    model_count = 0
    for cutoff in (2019, 2020, 2021):
        key = str(cutoff+2)
        lo, hi = year[idx-23], year[idx+24]
        parts = np.full(len(idx), 'excluded', dtype='U8')
        parts[hi <= cutoff] = 'train'
        parts[(lo == cutoff+1) & (hi == cutoff+1)] = 'dev'
        parts[(lo == cutoff+2) & (hi == cutoff+2)] = 'test'
        train = parts == 'train'
        selected = np.isin(parts, ['dev', 'test'])
        ei = idx[selected]
        meta = pd.read_parquet(run/f'{key}_origins.parquet')
        np.testing.assert_array_equal(meta.source_index, ei)
        np.testing.assert_array_equal(pd.DatetimeIndex(meta.timestamp).asi8, dates[ei].asi8)
        np.testing.assert_array_equal(meta.partition, parts[selected])
        origin = original[ei, 2].astype(np.float32)
        np.testing.assert_array_equal(meta.origin_aqi, origin)
        target = original[ei[:, None]+np.array(LEADS), 2].astype(np.float32)
        np.testing.assert_array_equal(target, np.load(run/f'{key}_targets.npy'))
        fold = report['folds'][key]
        assert fold['train_end_year'] == cutoff and fold['train_n'] == train.sum()
        assert fold['train_clean'] == (original[idx[train], 2] <= 100).sum()
        assert fold['train_polluted'] == (original[idx[train], 2] > 100).sum()
        for name, lag, trend in ARMS:
            x = independent_features(values, dates, ei, cutoff, lag, trend)
            assert x.shape == (len(ei), 139 if trend else 76)
            prediction = np.load(run/f'{key}_{name}.npy')
            assert prediction.shape == target.shape and np.isfinite(prediction).all() and (prediction >= 0).all()
            for j, lead in enumerate(LEADS):
                model = lgb.Booster(model_str=(run/f'{key}_{name}_h{lead}.txt').read_text('utf-8'))
                assert model.num_feature() == x.shape[1]
                assert model.current_iteration() == 300 and model.params['objective'] == 'regression_l1'
                p = model.predict(x, num_threads=4).clip(min=0).astype(np.float32)
                np.testing.assert_array_equal(p, prediction[:, j])
                model_count += 1
            candidates = {name: prediction}
            if not trend:
                pers = np.repeat(x[:, 71:72], 3, axis=1)
                np.testing.assert_array_equal(pers, np.load(run/f'{key}_persistence_lag{lag}.npy'))
                candidates[f'persistence_lag{lag}'] = pers
            for candidate, p in candidates.items():
                for part in ('dev', 'test'):
                    m = meta.partition.eq(part).to_numpy()
                    equal(diagnostics(target[m], p[m], origin[m]), fold['arms'][candidate][part], f'{key}/{candidate}/{part}')
                    if part == 'test':
                        pool[candidate].append((target[m], p[m], origin[m]))
        print(f'Independent fold {key}: models, labels, features and diagnostics pass', flush=True)
    for name, items in pool.items():
        y, p, o = [np.concatenate([v[j] for v in items]) for j in range(3)]
        equal(diagnostics(y, p, o), report['pooled_test'][name], f'pooled/{name}')
    for lag in (1, 3):
        b, t = [report['pooled_test'][f'lag{lag}_{arm}']['all']['mae'] for arm in ('base', 'trend')]
        comp = dict(mae_difference=t-b, relative_mae_reduction=(b-t)/b,
                    years_improved=sum(report['folds'][str(yr)]['arms'][f'lag{lag}_trend']['test']['all']['mae'] <
                      report['folds'][str(yr)]['arms'][f'lag{lag}_base']['test']['all']['mae'] for yr in (2021, 2022, 2023)))
        equal(comp, report['paired_comparisons'][f'lag{lag}_trend_vs_base'], 'paired_comparison')
    assert model_count == 45 and len(report['models']) == 45
    audit = dict(status='passed', independently_reinferred_models=model_count,
        feature_dimensions=[76, 139], training_feature_builder_imported=False,
        checks=['source/artifact/model hashes', 'complete population and boundary purge',
                'labels at issue+1,+6,+24 without delay shift', 'issue calendar and delayed partition-masked histories',
                'independent trend summaries', '45 saved-model inference equality', 'missing persistence coverage',
                'development/test/pooled MAE and RMSE and origin groups', 'paired comparisons',
                'code-reviewed all-conditions training without current-AQI routing'],
        limits=['Historical delay assumptions, not receipt-verified prospective data',
                'AQI units, not dissipation hours', 'No independent retraining',
                'Previously used years, not blind validation', 'Origin AQI groups are post-hoc diagnostics'],
        verifier_sha256=digest(Path(__file__)), report_sha256=digest(run/'report.json'))
    (run/'independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n', 'utf-8')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
