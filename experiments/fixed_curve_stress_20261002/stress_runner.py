"""Fixed old-model historical stress test. No fitting or parameter selection."""
import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '4')

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/continuous_trajectory_20260929'
sys.path.insert(0, str(OLD))
from delayed_chemistry import build as chemistry_build

SOURCE = ROOT / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
SOURCE_SHA = 'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'
SOURCE_RUNS = {
    'lgbm76': ROOT / 'experiments/learned_path_weights_20261002/runs/20261002T130057090429Z',
    'lgbm323': ROOT / 'experiments/empirical_paths_20261002/runs/20261002T121844976298Z',
}
YEARS = tuple(range(2021, 2026))
CHANNELS = ('pm2_5', 'pm10', 'aqi')
CHEMICALS = ('so2', 'no2', 'co', 'o3_8h')
HOUR_NS = 3600 * 10**9
BUFFERS = {'lgbm76': 14, 'lgbm323': 18}
ARMS = ('raw76', 'conservative76', 'raw323', 'conservative323')
CURVE_KEYS = dict(raw76='raw_curve76', conservative76='conservative_curve76',
    raw323='raw_curve323', conservative323='conservative_curve323')
START_KEYS = dict(raw76='raw_start76', conservative76='conservative_start76',
    raw323='raw_start323', conservative323='conservative_start323')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def event(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def validate_source(frame, require_full=False):
    if not all(name in frame for name in ('timestamp', *CHANNELS, *CHEMICALS)):
        raise ValueError('Required source channels absent')
    ts = pd.DatetimeIndex(frame.timestamp)
    if ts.tz is None or ts.hasnans or not ts.is_unique:
        raise ValueError('Unique timezone-aware source timestamps required')
    if len(ts) and not (np.diff(ts.asi8) == HOUR_NS).all():
        raise ValueError('Source must already be ordered on a continuous hourly grid')
    ts = ts.tz_convert('Asia/Shanghai')
    if not (ts.asi8 % HOUR_NS == 0).all():
        raise ValueError('Source timestamps must lie on exact hours')
    if require_full:
        if len(ts) != 107928 or ts[0].year != 2014 or ts[-1].year != 2026:
            raise ValueError('Frozen full-history source dimensions changed')
        for year in YEARS:
            expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01',
                inclusive='left', freq='h', tz='Asia/Shanghai')
            if not np.array_equal(ts[ts.year == year].asi8, expected.asi8):
                raise ValueError(f'Incomplete calendar year: {year}')
    return ts


def process_groups(timestamps, aqi):
    ts = pd.DatetimeIndex(timestamps).tz_convert('Asia/Shanghai')
    aqi = np.asarray(aqi, dtype=float)
    if len(aqi) != len(ts):
        raise ValueError('AQI and timestamps differ in length')
    epoch = pd.Timestamp('2014-01-01', tz='Asia/Shanghai').value
    result = []
    segment, episode, active, clean = -1, 0, False, 0
    for index, value in enumerate(aqi):
        reset = index == 0 or not np.isfinite(aqi[index-1]) or \
            ts.asi8[index] - ts.asi8[index-1] != HOUR_NS or ts.year[index] != ts.year[index-1]
        if reset:
            segment += 1
            active, clean = False, 0
        year = ts.year[index]
        if not np.isfinite(value):
            result.append(f'{year}:invalid')
            active, clean = False, 0
            continue
        if not active and value > 100:
            episode += 1
            active, clean = True, 0
        if active:
            result.append(f'{year}:s{segment}:p{episode}')
            clean = clean + 1 if value <= 100 else 0
            if clean == 6:
                active, clean = False, 0
        else:
            block = (ts.asi8[index] - epoch) // (72 * HOUR_NS)
            result.append(f'{year}:s{segment}:c{block}')
    return np.asarray(result, dtype=str)


def recovery(aqi):
    aqi = np.asarray(aqi)
    if aqi.ndim != 2 or aqi.shape[1] != 24 or not np.isfinite(aqi).all():
        raise ValueError('Finite 24h AQI required')
    runs = np.lib.stride_tricks.sliding_window_view(aqi <= 100, 6, axis=1).all(axis=2)
    return np.where(runs.any(axis=1), runs.argmax(axis=1) + 1, 20)


def prepare_year(raw, year):
    ts = validate_source(raw)
    if len(ts) == 0 or not (ts.year == year).all():
        raise ValueError('Nonempty unfiltered single-calendar-year source required')
    values = raw[list(CHANNELS)].to_numpy(dtype=float)
    groups = process_groups(ts, values[:, 2])
    indices, exclusions = [], []
    for index in range(len(ts)):
        if index < 24 or index + 24 >= len(ts):
            reason = 'year_edge'
        elif not np.isfinite(values[index, 2]):
            reason = 'origin_aqi_nonfinite'
        elif not np.isfinite(values[index-24:index]).all():
            reason = 'history_nonfinite'
        elif not np.isfinite(values[index+1:index+25]).all():
            reason = 'future_nonfinite'
        else:
            indices.append(index)
            continue
        exclusions.append((year, index, ts[index], reason))
    indices = np.asarray(indices, dtype=np.int64)
    history = values[indices[:, None] + np.arange(-24, 0)]
    paths = values[indices[:, None] + np.arange(1, 25)]
    issue = ts[indices]
    calendar = np.stack([np.sin(2*np.pi*issue.hour/24), np.cos(2*np.pi*issue.hour/24),
        np.sin(2*np.pi*issue.month/12), np.cos(2*np.pi*issue.month/12)], axis=1)
    features76 = np.column_stack([history.reshape(len(indices), 72), calendar])
    if len(indices):
        features323 = chemistry_build(raw,
            dict(indices=indices, timestamps=issue), 2019, 1)
    else:
        features323 = np.empty((0, 323), dtype=np.float32)
    if features76.shape != (len(indices), 76) or features76.dtype != np.float64 or \
            not np.isfinite(features76).all():
        raise ValueError('Invalid original 76-feature input')
    if features323.shape != (len(indices), 323) or features323.dtype != np.float32 or np.isinf(features323).any():
        raise ValueError('Invalid original chemistry input')
    bundle = dict(indices=indices, timestamps=issue.asi8, groups=groups[indices],
        origin_aqi=values[indices, 2], features76=features76, features323=features323,
        paths=paths, labels=recovery(paths[:, :, 2]))
    return bundle, pd.DataFrame(exclusions, columns=['year', 'source_index', 'timestamp', 'reason'])


def model_paths(width):
    name = 'lgbm76_aqi_h' if width == 76 else 'chemistry_aqi_h'
    return [SOURCE_RUNS[f'lgbm{width}'] / f'{name}{lead:02}.txt' for lead in range(1, 25)]


def load_heads(width):
    models = [lgb.Booster(model_str=path.read_text(encoding='utf-8')) for path in model_paths(width)]
    if any(model.num_feature() != width for model in models):
        raise ValueError('Reused model input width changed')
    return models


def predict_heads(features, models):
    if len(models) != 24:
        raise ValueError('Exactly 24 fixed AQI heads required')
    result = np.empty((len(features), 24), dtype=np.float32)
    if len(features):
        for index, model in enumerate(models):
            result[:, index] = np.asarray(model.predict(features, num_threads=4)).clip(min=0).astype(np.float32)
    return result


def fixed_conservative(raw, width):
    return np.asarray(raw).astype(np.float64) + BUFFERS[f'lgbm{width}']


def add_predictions(bundle, models):
    for width in (76, 323):
        curve = predict_heads(bundle[f'features{width}'], models[width])
        bundle[f'raw_curve{width}'] = curve
        bundle[f'conservative_curve{width}'] = fixed_conservative(curve, width)
        bundle[f'covered{width}'] = np.isfinite(curve).all(axis=1)
    return bundle


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def recovery_metrics(truth, prediction, groups, with_groups=True):
    truth, prediction, groups = map(np.asarray, (truth, prediction, groups))
    if not (len(truth) == len(prediction) == len(groups)):
        raise ValueError('Metric population dimensions differ')
    if not (np.isin(truth, np.arange(1, 21)).all() and np.isin(prediction, np.arange(1, 21)).all()):
        raise ValueError('Recovery metrics require capped 1..20 starts')
    positive, hat = truth < 20, prediction < 20
    tp, fn, fp, tn = [int(mask.sum()) for mask in
        (positive & hat, positive & ~hat, ~positive & hat, ~positive & ~hat)]
    error = np.abs(truth - prediction)
    early = (prediction < truth) & hat
    n = len(truth)
    result = dict(n=n, actual_right_censored=int((~positive).sum()), predicted_recovery=int(hat.sum()),
        mae=float(error.mean()) if n else None,
        group_equal_mae=float(pd.Series(error).groupby(groups).mean().mean()) if n else None,
        groups=len(set(groups)), confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn),
        recall=ratio(tp, tp+fn), precision=ratio(tp, tp+fp), fpr=ratio(fp, fp+tn),
        early_fraction=ratio(int(early.sum()), n), early_count=int(early.sum()))
    if with_groups:
        records = [recovery_metrics(truth[groups == group], prediction[groups == group],
            groups[groups == group], False) for group in np.unique(groups)]
        result['group_equal'] = group_equal_rates(records,
            ('mae', 'recall', 'precision', 'fpr', 'early_fraction'))
    return result


def group_equal_rates(records, names):
    result = {}
    for name in names:
        valid = [record[name] for record in records if record[name] is not None]
        result[name] = dict(value=float(np.mean(valid)) if valid else None,
            valid_groups=len(valid), total_groups=len(records))
    return result


def onset_metrics(truth, predicted, groups, with_groups=True):
    truth, predicted, groups = map(np.asarray, (truth, predicted, groups))
    positive, hat = truth.astype(bool), predicted.astype(bool)
    tp, fn, fp, tn = [int(mask.sum()) for mask in
        (positive & hat, positive & ~hat, ~positive & hat, ~positive & ~hat)]
    result = dict(n=len(truth), positive_n=int(positive.sum()), negative_n=int((~positive).sum()),
        predicted_positive=int(hat.sum()), confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn),
        recall=ratio(tp, tp+fn), precision=ratio(tp, tp+fp), fpr=ratio(fp, fp+tn))
    if with_groups:
        records = [onset_metrics(truth[groups == group], predicted[groups == group],
            groups[groups == group], False) for group in np.unique(groups)]
        result['group_equal'] = group_equal_rates(records, ('recall', 'precision', 'fpr'))
    return result


def concentration_metrics(truth, prediction, groups):
    error = np.abs(prediction.astype(float) - truth)
    n = len(truth)
    return dict(n=n, hour_cells=n*24, mae=float(error.mean()) if n else None,
        lead_mae=error.mean(axis=0).tolist() if n else [None]*24,
        group_equal_mae=float(pd.Series(error.mean(axis=1)).groupby(groups).mean().mean()) if n else None,
        group_equal_lead_mae=pd.DataFrame(error).groupby(groups).mean().mean().tolist() if n else [None]*24,
        units='AQI')


def population_masks(bundle, width):
    covered = bundle[f'covered{width}']
    common = bundle['covered76'] & bundle['covered323']
    polluted = bundle['origin_aqi'] > 100
    return dict(all=covered, polluted=covered & polluted, clean=covered & ~polluted,
        common=common, common_polluted=common & polluted, common_clean=common & ~polluted)


def bundle_metrics(bundle):
    result = dict(recovery={}, concentration={}, clean_onset={})
    truth, groups = bundle['labels'], bundle['groups']
    truth_aqi = bundle['paths'][:, :, 2]
    for arm in ARMS:
        width = 76 if arm.endswith('76') else 323
        curve = bundle[CURVE_KEYS[arm]]
        masks = population_masks(bundle, width)
        starts = np.full(len(truth), -1, dtype=np.int64)
        starts[bundle[f'covered{width}']] = recovery(curve[bundle[f'covered{width}']])
        result['recovery'][arm] = {name: recovery_metrics(truth[mask], starts[mask], groups[mask])
            for name, mask in masks.items()}
        result['concentration'][arm] = {name: concentration_metrics(truth_aqi[mask], curve[mask], groups[mask])
            for name, mask in masks.items()}
        clean = masks['clean']
        result['clean_onset'][arm] = onset_metrics((truth_aqi[clean] > 100).any(axis=1),
            (curve[clean] > 100).any(axis=1), groups[clean])
    return result


def chemical_coverage(bundle):
    history = bundle['features323'][:, 139:235].reshape(-1, 24, 4)
    polluted = bundle['origin_aqi'] > 100
    result = {}
    for name, mask in dict(all=np.ones(len(history), bool), polluted=polluted, clean=~polluted).items():
        values = np.isfinite(history[mask])
        n = int(mask.sum())
        result[name] = dict(n=n)
        for index, channel in enumerate(CHEMICALS):
            valid = values[:, :, index]
            result[name][channel] = dict(finite_cells=int(valid.sum()), total_cells=n*24,
                history_fraction=ratio(int(valid.sum()), n*24),
                origins_all24=int(valid.all(axis=1).sum()), origins_any=int(valid.any(axis=1).sum()))
    return result


def value_diagnostics(values, channels):
    values = np.asarray(values)
    result = {}
    for index, channel in enumerate(channels):
        column = values[..., index]
        finite = np.isfinite(column)
        result[channel] = dict(finite_cells=int(finite.sum()),
            negative_cells=int((finite & (column < 0)).sum()), nonfinite_cells=int((~finite).sum()))
    return result


def negative_diagnostics(raw, bundle):
    indices = bundle['indices']
    values = raw[list(CHANNELS)].to_numpy(float)
    return dict(raw=value_diagnostics(raw[[*CHANNELS, *CHEMICALS]].to_numpy(float), (*CHANNELS, *CHEMICALS)),
        eligible_history=value_diagnostics(values[indices[:, None] + np.arange(-24, 0)], CHANNELS),
        eligible_future=value_diagnostics(bundle['paths'], CHANNELS),
        issue=value_diagnostics(values[indices], CHANNELS))


def bundle_counts(bundle, candidate_hours, excluded_hours):
    polluted = bundle['origin_aqi'] > 100
    n = len(bundle['indices'])
    return dict(candidate_hours=int(candidate_hours), excluded_hours=int(excluded_hours),
        eligible_origins=n, polluted=int(polluted.sum()), clean=int((~polluted).sum()),
        polluted_positive=int((polluted & (bundle['labels'] < 20)).sum()),
        polluted_negative=int((polluted & (bundle['labels'] == 20)).sum()),
        covered76=int(bundle['covered76'].sum()), covered323=int(bundle['covered323'].sum()),
        common_covered=int((bundle['covered76'] & bundle['covered323']).sum()),
        coverage76=ratio(int(bundle['covered76'].sum()), n),
        coverage323=ratio(int(bundle['covered323'].sum()), n),
        eligible_fraction_of_calendar=ratio(n, candidate_hours))


def year_equal_summary(annual, years=YEARS):
    records = [annual[str(year)] for year in years]
    def walk(items):
        first = items[0]
        if isinstance(first, dict):
            return {name: walk([item[name] for item in items]) for name in first}
        if isinstance(first, list):
            return [walk([item[index] for item in items]) for index in range(len(first))]
        if isinstance(first, str):
            return first
        valid = [value for value in items if value is not None]
        return dict(value=float(np.mean(valid)) if valid else None,
            valid_years=len(valid), total_years=len(years))
    return walk(records)


def evaluate_gates(annual, pooled, counts, years=YEARS):
    checks = {}
    for year in years:
        count = counts[str(year)]
        checks[f'{year}:coverage76_100'] = count['coverage76'] == 1.
        checks[f'{year}:coverage323_100'] = count['coverage323'] == 1.
        checks[f'{year}:polluted_positive_and_negative'] = count['polluted_positive'] > 0 and count['polluted_negative'] > 0
    for name, metrics in [(str(year), annual[str(year)]) for year in years] + [('pooled', pooled)]:
        new = metrics['recovery']['conservative76']['polluted']
        reference = metrics['recovery']['conservative323']['polluted']
        for field in ('mae', 'group_equal_mae'):
            checks[f'{name}:{field}_not_worse'] = new[field] is not None and reference[field] is not None and new[field] <= reference[field]
        checks[f'{name}:recall_not_worse'] = new['recall'] is not None and reference['recall'] is not None and new['recall'] >= reference['recall']
        checks[f'{name}:fpr_5pct'] = new['fpr'] is not None and new['fpr'] <= .05
    stable = dict(checks=checks, passed=all(checks.values()),
        failure_reasons=[name for name, passed in checks.items() if not passed])
    new = pooled['recovery']['conservative76']['polluted']
    reference = pooled['recovery']['conservative323']['polluted']
    improvement = dict(stability_passed=stable['passed'],
        pooled_mae_10pct=new['mae'] is not None and reference['mae'] is not None and reference['mae'] > 0 and new['mae'] <= reference['mae']*.9,
        pooled_recall_5pp=new['recall'] is not None and reference['recall'] is not None and new['recall'] >= reference['recall']+.05)
    meaningful = dict(checks=improvement, passed=all(improvement.values()),
        failure_reasons=[name for name, passed in improvement.items() if not passed],
        relative_pooled_mae_reduction=ratio(reference['mae']-new['mae'], reference['mae'])
            if new['mae'] is not None and reference['mae'] is not None else None,
        pooled_recall_difference=new['recall']-reference['recall']
            if new['recall'] is not None and reference['recall'] is not None else None)
    return stable, meaningful


def tables_for_year(bundle, year):
    n = len(bundle['indices'])
    timestamps = pd.to_datetime(bundle['timestamps'], utc=True).tz_convert('Asia/Shanghai')
    eligible = pd.DataFrame(dict(year=np.full(n, year, dtype=int), source_index=bundle['indices'],
        timestamp=timestamps, group=bundle['groups'], origin_aqi=bundle['origin_aqi'],
        true_start=bundle['labels'], covered76=bundle['covered76'], covered323=bundle['covered323']))
    row = eligible.copy()
    row['year_row'] = np.arange(n)
    row['polluted'] = bundle['origin_aqi'] > 100
    row['true_onset'] = (bundle['paths'][:, :, 2] > 100).any(axis=1)
    for arm in ARMS:
        width = 76 if arm.endswith('76') else 323
        covered = bundle[f'covered{width}']
        start = np.full(n, -1, dtype=int)
        start[covered] = recovery(bundle[CURVE_KEYS[arm]][covered])
        row[START_KEYS[arm]] = start
        row[f'{arm}_absolute_error'] = np.where(covered, np.abs(start-bundle['labels']), np.nan)
        row[f'{arm}_onset'] = pd.array(np.where(covered,
            (bundle[CURVE_KEYS[arm]] > 100).any(axis=1), None), dtype='boolean')
        error = np.abs(bundle[CURVE_KEYS[arm]].astype(float)-bundle['paths'][:, :, 2]).mean(axis=1)
        row[f'{arm}_concentration_error'] = np.where(covered, error, np.nan)
    aggregations = dict(n=('year_row', 'size'))
    for arm in ARMS:
        aggregations[f'{arm}_n'] = (f'{arm}_absolute_error', 'count')
        aggregations[f'{arm}_mae'] = (f'{arm}_absolute_error', 'mean')
        aggregations[f'{arm}_concentration_mae'] = (f'{arm}_concentration_error', 'mean')
    group = row.groupby(['year', 'polluted', 'group'], as_index=False).agg(**aggregations)
    return eligible, row, group


def assert_hashes(manifest):
    for path, digest in manifest.items():
        if sha(path) != digest:
            raise ValueError(f'Frozen file changed: {path}')


def validate_reuse():
    reports, reuse, models = {}, {}, {}
    expected_status = dict(lgbm76='pending_independent_verification', lgbm323='completed_pending_verification')
    for name, run in SOURCE_RUNS.items():
        report_path, verification_path = run / 'report.json', run / 'verification.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        if report['status'] != expected_status[name] or verification['status'] != 'passed':
            raise ValueError(f'Unverified reused experiment: {name}')
        if report['train_years'] != list(range(2014, 2020)) or report['dev_year'] != 2020:
            raise ValueError('Reused training/development partition changed')
        selected = report['selected_buffers'][name] if name == 'lgbm76' else report['selected_buffer']
        if selected != BUFFERS[name]:
            raise ValueError('Frozen development buffer changed')
        independent = verification['independent_frozen_sha256']
        if independent[str(report_path)] != sha(report_path):
            raise ValueError('Independently verified source report changed')
        if report['frozen_sha256'][str(SOURCE)] != SOURCE_SHA:
            raise ValueError('Reused source SHA changed')
        reuse.update({str(report_path): sha(report_path), str(verification_path): sha(verification_path)})
        assert_hashes(report['frozen_sha256'])
        reuse.update(report['frozen_sha256'])
        # Independent Python code and imported dependencies are part of the evidence chain.
        independent_code = {path: digest for path, digest in independent.items() if path.endswith('.py')}
        assert_hashes(independent_code)
        reuse.update(independent_code)
        names = ['prediction.npz']
        if name == 'lgbm323':
            names += ['train.npz', 'dev.npz', 'eligible.parquet', 'exclusions.parquet', 'chemistry_features.npy', 'buffer_grid.json']
        else:
            names += ['buffer_grid76.json', 'buffer_grid323.json']
        width = 76 if name == 'lgbm76' else 323
        names += [path.name for path in model_paths(width)]
        for artifact in names:
            path = run / artifact
            digest = sha(path)
            if report['artifact_sha256'][artifact] != digest or independent[str(path)] != digest:
                raise ValueError(f'Reused verified artifact changed: {path}')
            reuse[str(path)] = digest
            if path.suffix == '.txt':
                models[str(path)] = digest
        reports[name] = report
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen source SHA mismatch')
    return reports, reuse, models


def dependency_files():
    files = [HERE / name for name in ('PROTOCOL.md', 'stress_runner.py', 'test_stress_runner.py',
        'verify_stress.py', 'test_verify_stress.py')]
    files.append(Path(sys.executable).resolve())
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'pyarrow'):
        distribution = importlib.metadata.distribution(name)
        files.extend(Path(distribution.locate_file(item)).resolve() for item in distribution.files if item.name == 'RECORD')
    for module in tuple(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if value and Path(value).resolve().is_relative_to(OLD):
            files.append(Path(value).resolve())
    return sorted(set(files))


def compatibility2020(raw, models):
    ts = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    annual = raw.loc[ts.year == 2020].reset_index(drop=True)
    bundle, excluded = prepare_year(annual, 2020)
    source_run = SOURCE_RUNS['lgbm323']
    with np.load(source_run / 'dev.npz', allow_pickle=False) as archive:
        old = {name: archive[name] for name in archive.files}
    for new_key, old_key in [('timestamps', 'timestamps'), ('features76', 'features'),
            ('paths', 'paths'), ('labels', 'labels'), ('origin_aqi', 'origin_aqi')]:
        np.testing.assert_array_equal(bundle[new_key], old[old_key])
    first = int(np.flatnonzero(ts.year == 2020)[0])
    np.testing.assert_array_equal(bundle['indices'] + first, old['indices'])
    # Group identifiers change namespace; their equality relations must not change.
    np.testing.assert_array_equal(pd.factorize(bundle['groups'], sort=False)[0],
        pd.factorize(old['groups'], sort=False)[0])
    with np.load(source_run / 'train.npz', allow_pickle=False) as train:
        train_n = len(train['indices'])
    chemistry = np.load(source_run / 'chemistry_features.npy', mmap_mode='r', allow_pickle=False)[train_n:]
    np.testing.assert_array_equal(bundle['features323'], chemistry)
    add_predictions(bundle, models)
    with np.load(SOURCE_RUNS['lgbm76'] / 'prediction.npz', allow_pickle=False) as old_prediction:
        for name in ('raw_curve76', 'conservative_curve76', 'raw_curve323', 'conservative_curve323'):
            np.testing.assert_array_equal(bundle[name], old_prediction[name])
    old_excluded = pd.read_parquet(source_run / 'exclusions.parquet')
    expected_excluded = old_excluded.loc[old_excluded.partition == 'dev']
    np.testing.assert_array_equal(excluded.source_index.to_numpy()+first, expected_excluded.source_index.to_numpy())
    return dict(status='passed', origins=len(bundle['indices']), excluded_hours=len(excluded),
        checks=['eligibility_indices_and_timestamps', 'labels_original_paths_origin_aqi',
            'features76_float64_month', 'features323_float32_dayofyear',
            'all_48_reloaded_heads_raw_and_fixed_conservative_curves', 'group_equivalence'])


def concatenate_bundles(bundles):
    return {name: np.concatenate([bundle[name] for bundle in bundles]) for name in bundles[0]}


def sum_nested(items):
    first = items[0]
    if isinstance(first, dict):
        return {name: sum_nested([item[name] for item in items]) for name in first}
    return sum(items)


def execute(raw, reports, reuse, model_hashes, frozen, started):
    run = HERE / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='running', run_dir=str(run), years=list(YEARS),
        source=str(SOURCE), source_sha256=SOURCE_SHA,
        source_runs={name: str(path) for name, path in SOURCE_RUNS.items()},
        frozen_sha256=frozen, reuse_verified_sha256=reuse, model_sha256=model_hashes,
        buffers=BUFFERS, train_years=list(range(2014, 2020)), selected_year=2020,
        historical_stress_test=True, blind_test=False, refitted=False, deployed=False,
        target='available-station per-pollutant median AQI', recovery_target='min(T,20); first six-hour clean window start, 1..19 or 20',
        timestamp_timezone='Asia/Shanghai', source_index_scope='year-local unfiltered source row',
        feature_dimensions=dict(lgbm76=76, lgbm323=323),
        model_disk_bytes={name: sum(path.stat().st_size for path in model_paths(width))
            for name, width in (('lgbm76', 76), ('lgbm323', 323))},
        versions=dict(python=sys.version, **{name: importlib.metadata.version(name)
            for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'pyarrow')}),
        artifact_schema=dict(bundle='bundle_YEAR.npz; indices are year-local; features76/323, paths, labels, four curves, covered76/323',
            eligible='eligible.parquet; all eligible year-local origins and prediction coverage',
            exclusions='exclusions.parquet; exactly one ordered reason per ineligible calendar hour',
            row_metrics='row_metrics.parquet; four recovery starts, errors, deterministic onset, coverage',
            group_metrics='group_metrics.parquet; year/polluted/group and four-arm recovery/concentration errors',
            annual_metrics='annual_metrics.json; report.metrics.annual mirror'))
    report['model_disk_bytes']['total'] = sum(report['model_disk_bytes'].values())
    write_json(run / 'report.json', report)
    event('run_created', run_dir=str(run))
    try:
        assert_hashes(frozen)
        models = {width: load_heads(width) for width in (76, 323)}
        report['compatibility2020'] = compatibility2020(raw, models)
        event('compatibility2020_passed', **report['compatibility2020'])
        annual, count, chemicals, negatives = {}, {}, {}, {}
        bundles, eligible_tables, exclusions_tables, row_tables, group_tables = [], [], [], [], []
        ts = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
        for year in YEARS:
            annual_raw = raw.loc[ts.year == year].reset_index(drop=True)
            bundle, excluded = prepare_year(annual_raw, year)
            add_predictions(bundle, models)
            np.savez_compressed(run / f'bundle_{year}.npz', **bundle)
            eligible, row, group = tables_for_year(bundle, year)
            key = str(year)
            annual[key] = bundle_metrics(bundle)
            count[key] = bundle_counts(bundle, len(annual_raw), len(excluded))
            chemicals[key] = chemical_coverage(bundle)
            negatives[key] = negative_diagnostics(annual_raw, bundle)
            bundles.append(bundle)
            eligible_tables.append(eligible)
            exclusions_tables.append(excluded)
            row_tables.append(row)
            group_tables.append(group)
            event('year_complete', year=year, eligible=count[key]['eligible_origins'])
        pooled = concatenate_bundles(bundles)
        pooled_metrics = bundle_metrics(pooled)
        pooled_counts = bundle_counts(pooled, sum(item['candidate_hours'] for item in count.values()),
            sum(item['excluded_hours'] for item in count.values()))
        stable, meaningful = evaluate_gates(annual, pooled_metrics, count)
        for name, tables in [('eligible', eligible_tables), ('exclusions', exclusions_tables),
                ('row_metrics', row_tables), ('group_metrics', group_tables)]:
            pd.concat(tables, ignore_index=True).to_parquet(run / f'{name}.parquet', index=False)
        write_json(run / 'annual_metrics.json', annual)
        assert_hashes(frozen)
        report.update(status='pending_independent_verification',
            counts=dict(annual=count, pooled=pooled_counts),
            metrics=dict(annual=annual, pooled=pooled_metrics, year_equal=year_equal_summary(annual),
                chemical_coverage=dict(annual=chemicals, pooled=chemical_coverage(pooled))),
            negative_diagnostics=dict(annual=negatives, pooled=sum_nested(list(negatives.values()))),
            stability_gates=stable, meaningful_gates=meaningful,
            elapsed_seconds=time.perf_counter()-started,
            limits=['Historical stress test, not unseen blind or real-time validation',
                'Fixed 2014-2019 models, not annually refitted performance',
                '100% prediction coverage refers only to eligible origins',
                'Recovery MAE is capped at 20h, not complete dissipation duration',
                'No path probability, calibration, Brier score or confidence coverage',
                'Feature width and model disk size are not measured service speed',
                'No automatic deployment authorization'])
    except BaseException as error:
        report.update(status='execution_failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['artifact_sha256'] = {path.name: sha(path) for path in run.iterdir()
            if path.is_file() and path.name != 'report.json'}
        write_json(run / 'report.json', report)
    event('pending_independent_verification', run_dir=str(run),
        stability_passed=report['stability_gates']['passed'], meaningful_passed=report['meaningful_gates']['passed'])
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--execute', action='store_true', help='Only after explicit specification and code/test approval')
    args = parser.parse_args()
    started = time.perf_counter()
    reports, reuse, model_hashes = validate_reuse()
    raw = pd.read_parquet(SOURCE)
    ts = validate_source(raw, require_full=True)
    frozen = {str(path): sha(path) for path in dependency_files()}
    frozen.update(reuse)
    frozen[str(SOURCE)] = SOURCE_SHA
    assert_hashes(frozen)
    if args.preflight_only:
        event('preflight_passed', source_rows=len(ts), years=list(YEARS),
            source_runs={name: str(path) for name, path in SOURCE_RUNS.items()},
            independently_verified_sources=2, fixed_models=len(model_hashes), frozen_files=len(frozen),
            historical_inference_started=False, training_started=False)
        return
    execute(raw, reports, reuse, model_hashes, frozen, started)


if __name__ == '__main__':
    main()
