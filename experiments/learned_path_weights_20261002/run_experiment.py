"""Frozen supervised leaf-weight experiment. Development only, not deployment."""
import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '4')

from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
SOURCE_SHA = 'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'
SOURCE_RUN = ROOT / 'experiments/empirical_paths_20261002/runs/20261002T121844976298Z'
SPEC = ROOT / 'docs/superpowers/specs/2026-10-02-learned-path-weights-design.md'
PLAN = ROOT / 'docs/superpowers/plans/2026-10-02-learned-path-weights-execution.md'
FOREST_CONFIG = dict(n_estimators=300, criterion='squared_error', bootstrap=True,
    min_samples_leaf=30, max_features=1.0, max_depth=None, random_state=42, n_jobs=4)
LGBM_CONFIG = dict(objective='regression_l1', n_estimators=300, learning_rate=.04,
    num_leaves=24, min_child_samples=30, random_state=42, n_jobs=4, verbosity=-1,
    deterministic=True, force_col_wise=True)
THRESHOLDS = [round(i * .05, 2) for i in range(1, 21)] + [1.000001]
QUANTILES = [.1, .5, .9]
CHANNELS = ['pm2_5', 'pm10', 'aqi']
ARTIFACT_SCHEMA = {
    'forest.joblib': 'sklearn RandomForestRegressor; no inference input other than 76 features',
    'forest_structure.npz': {'bootstrap_indices': '(300,n_train) int32',
        'train_leaves': '(n_train,300) int32', 'dev_leaves': '(n_dev,300) int32'},
    'weights.npz': 'scipy CSR float64 (n_dev,n_train), sorted train row indices, all nonzero weights',
    'prediction.npz': {'pmf': '(n_dev,20)', 'cdf': '(n_dev,19)', 'pred_start': '(n_dev,) int',
        'forest_aqi_mean': '(n_dev,24)', 'path_mean': '(n_dev,24,3)',
        'path_quantiles': '(n_dev,3,24,3), levels 0.1/0.5/0.9, inverse discrete CDF',
        'onset_probability': '(n_dev,)', 'median_start': '(n_dev,) int',
        'raw_curve323': '(n_dev,24) float32', 'conservative_curve323': '(n_dev,24) float64',
        'raw_curve76': '(n_dev,24) float32', 'conservative_curve76': '(n_dev,24) float64',
        'unconditional_cdf': '(19,)', 'unconditional_onset': 'scalar', 'covered': '(n_dev,) bool'},
    'support.parquet': ['dev_row', 'nonzero_support', 'effective_support', 'distinct_processes', 'max_process_weight'],
    'train_support_usage.parquet': ['train_row', 'query_support_count', 'total_weight'],
    'grid.json': '21 theta rows and polluted matched population metrics; last theta is all-reject',
    'buffer_grid76.json': '102 integer delta rows, 76-feature polluted population',
    'buffer_grid323.json': '102 integer delta rows, 323-feature polluted population',
    'row_metrics.parquet': ['dev_row', 'timestamp', 'group', 'polluted', 'covered', 'true_start',
        'pred_start', 'raw_start323', 'conservative_start323', 'raw_start76', 'conservative_start76',
        'onset_probability', 'true_onset', 'absolute_error', 'reference_absolute_error', 'reference76_absolute_error'],
    'group_metrics.parquet': ['polluted', 'group', 'n', 'mae', 'reference_mae', 'reference76_mae'],
    'reliability.parquet': ['horizon', 'bin', 'n', 'mean_probability', 'observed_fraction'],
    'lgbm76_aqi_h01.txt..h24.txt': 'UTF-8 LightGBM model strings; reloaded predictions exactly match',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def event(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def leaf_index(bootstrap_indices, train_leaves):
    bootstrap_indices = np.asarray(bootstrap_indices)
    train_leaves = np.asarray(train_leaves)
    n_trees, n_draws = bootstrap_indices.shape
    n_train = len(train_leaves)
    if train_leaves.shape[1] != n_trees or n_draws != n_train:
        raise ValueError('Full-size bootstrap and leaf identities required')
    if (bootstrap_indices < 0).any() or (bootstrap_indices >= n_train).any():
        raise ValueError('Invalid bootstrap train row')
    result = []
    for tree in range(n_trees):
        count = np.bincount(bootstrap_indices[tree], minlength=n_train)
        used = np.flatnonzero(count)
        order = np.argsort(train_leaves[used, tree], kind='stable')
        used = used[order]
        leaf = train_leaves[used, tree]
        boundaries = np.r_[0, np.flatnonzero(np.diff(leaf)) + 1, len(used)]
        lookup = {}
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            rows = used[begin:end]
            # Normalize inside each leaf first; every tree then contributes 1/B.
            weight = count[rows].astype(float) / (count[rows].sum() * n_trees)
            lookup[int(leaf[begin])] = (rows, weight)
        result.append(lookup)
    return result


def forest_weights(index, query_leaves, n_train):
    query_leaves = np.asarray(query_leaves)
    if query_leaves.ndim != 2 or query_leaves.shape[1] != len(index):
        raise ValueError('One leaf per fitted tree required')
    row_indices, row_values, pointer = [], [], [0]
    for leaves in query_leaves:
        weight = np.zeros(n_train, dtype=float)
        for lookup, leaf in zip(index, leaves):
            rows, values = lookup[int(leaf)]
            weight[rows] += values
        nonzero = np.flatnonzero(weight)
        if not np.isfinite(weight).all() or (weight < 0).any() or not np.isclose(weight.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError('Invalid conditional path mass')
        weight /= weight.sum()
        row_indices.append(nonzero.astype(np.int32))
        row_values.append(weight[nonzero])
        pointer.append(pointer[-1] + len(nonzero))
    return sparse.csr_matrix((np.concatenate(row_values), np.concatenate(row_indices),
        np.asarray(pointer, dtype=np.int64)), shape=(len(query_leaves), n_train))


def forest_structure(forest, train_features, query_features):
    bootstrap = np.asarray(forest.estimators_samples_, dtype=np.int32)
    train_leaves = forest.apply(train_features).astype(np.int32)
    query_leaves = forest.apply(query_features).astype(np.int32)
    return bootstrap, train_leaves, query_leaves


def assert_forest_equivalent(expected, actual):
    # Parallel tree summation may reorder additions after model reload.
    np.testing.assert_allclose(expected, actual, atol=1e-10, rtol=1e-12)
    return float(np.max(np.abs(expected - actual)))


def recovery(aqi):
    aqi = np.asarray(aqi)
    if aqi.ndim != 2 or aqi.shape[1] != 24 or not np.isfinite(aqi).all():
        raise ValueError('Finite 24h AQI required')
    runs = np.lib.stride_tricks.sliding_window_view(aqi <= 100, 6, axis=1).all(axis=2)
    return np.where(runs.any(axis=1), runs.argmax(axis=1) + 1, 20)


def probability_endpoints(value):
    value = np.asarray(value, dtype=float).copy()
    if not np.isfinite(value).all() or (value < -1e-12).any() or (value > 1 + 1e-12).any():
        raise ValueError('Invalid probability outside numerical endpoint tolerance')
    value[np.abs(value) <= 1e-12] = 0.
    value[np.abs(value - 1) <= 1e-12] = 1.
    return value


def path_products(weights, paths):
    paths = np.asarray(paths)
    if paths.shape != (weights.shape[1], 24, 3) or not np.isfinite(paths).all():
        raise ValueError('Finite original three-channel complete paths required')
    label = recovery(paths[:, :, 2])
    indicator = np.eye(20)[label - 1]
    pmf = weights @ indicator
    pmf /= pmf.sum(axis=1, keepdims=True)
    cdf = probability_endpoints(np.cumsum(pmf, axis=1)[:, :19])
    onset = probability_endpoints(weights @ (paths[:, :, 2] > 100).any(axis=1).astype(float))
    mean = (weights @ paths.reshape(len(paths), 72)).reshape(-1, 24, 3)
    median = (np.cumsum(pmf, axis=1) >= .5).argmax(axis=1) + 1
    quantiles = np.empty((weights.shape[0], len(QUANTILES), 24, 3))
    flat = paths.reshape(len(paths), 72)
    for row in range(weights.shape[0]):
        begin, end = weights.indptr[row:row + 2]
        values = flat[weights.indices[begin:end]]
        order = np.argsort(values, axis=0, kind='stable')
        ordered = np.take_along_axis(values, order, axis=0)
        mass = np.cumsum(weights.data[begin:end][order], axis=0)
        for level, q in enumerate(QUANTILES):
            position = (mass >= q).argmax(axis=0)
            quantiles[row, level] = ordered[position, np.arange(72)].reshape(24, 3)
    return dict(pmf=pmf, cdf=cdf, path_mean=mean, path_quantiles=quantiles,
        onset_probability=onset, median_start=median)


def decision(cdf, theta):
    hit = np.asarray(cdf) >= theta
    return np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 20)


def support_diagnostics(weights, groups):
    _, group_code = np.unique(groups, return_inverse=True)
    rows = []
    for row in range(weights.shape[0]):
        begin, end = weights.indptr[row:row + 2]
        indices = weights.indices[begin:end]
        mass = weights.data[begin:end]
        process_mass = np.bincount(group_code[indices], weights=mass)
        rows.append(dict(dev_row=row, nonzero_support=len(indices),
            effective_support=float(1 / np.sum(mass ** 2)),
            distinct_processes=int(np.count_nonzero(process_mass)),
            max_process_weight=float(process_mass.max())))
    usage = pd.DataFrame(dict(train_row=np.arange(weights.shape[1]),
        query_support_count=np.bincount(weights.indices, minlength=weights.shape[1]),
        total_weight=np.asarray(weights.sum(axis=0)).ravel()))
    return pd.DataFrame(rows), usage


def metrics(truth, prediction, groups, _groups=True):
    truth, prediction, groups = map(np.asarray, (truth, prediction, groups))
    yes, hat = truth < 20, prediction < 20
    tp, fn, fp, tn = [int(mask.sum()) for mask in (yes & hat, yes & ~hat, ~yes & hat, ~yes & ~hat)]
    n = len(truth)
    error = np.abs(truth - prediction)
    early = (prediction < truth) & hat
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    result = dict(n=n, actual_right_censored=int((~yes).sum()), predicted_recovery=int(hat.sum()),
        mae=float(error.mean()) if n else None,
        group_equal_mae=float(pd.Series(error).groupby(groups).mean().mean()) if n else None,
        groups=len(set(groups)), confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn),
        recall=ratio(tp, tp + fn), precision=ratio(tp, tp + fp), fpr=ratio(fp, fp + tn),
        early_fraction=ratio(early.sum(), n), early_count=int(early.sum()))
    if _groups:
        per_group = [metrics(truth[groups == g], prediction[groups == g], groups[groups == g], False)
            for g in np.unique(groups)]
        result['group_equal'] = {}
        for name in ('mae', 'recall', 'precision', 'fpr', 'early_fraction'):
            valid = [item[name] for item in per_group if item[name] is not None]
            result['group_equal'][name] = dict(value=float(np.mean(valid)) if valid else None,
                valid_groups=len(valid), total_groups=len(per_group))
    return result


def select_theta(rows, max_early):
    valid = [row for row in rows if row['fpr'] is not None and row['fpr'] <= .05
        and row['early_fraction'] <= max_early and row['confusion']['tp'] > 0
        and row['confusion']['fp'] + row['confusion']['tn'] > 0]
    return min(valid, key=lambda row: (row['mae'], -row['recall'], row['fpr'], -row['theta'])) if valid else None


def probability_metrics(cdf, truth, groups):
    error = (cdf - (truth[:, None] <= np.arange(1, 20))) ** 2
    result = dict(n=len(truth), brier_mean_1_19=float(error.mean()),
        group_equal_brier_mean_1_19=float(pd.Series(error.mean(axis=1)).groupby(groups).mean().mean()))
    for horizon in (6, 12, 19):
        cells = error[:, horizon - 1]
        result[f'brier_{horizon}'] = float(cells.mean())
        result[f'group_equal_brier_{horizon}'] = float(pd.Series(cells).groupby(groups).mean().mean())
    return result


def select_buffer(curve, truth, groups):
    raw = metrics(truth, recovery(curve), groups)
    rows = [dict(delta=delta, **metrics(truth, recovery(curve.astype(float) + delta), groups))
        for delta in range(102)]
    feasible = [row for row in rows if row['fpr'] is not None and row['fpr'] <= .05
        and row['early_fraction'] <= raw['early_fraction']]
    selected = min(feasible, key=lambda row: (row['mae'], -(row['recall'] or 0), row['fpr'], -row['delta'])) if feasible else None
    if selected is None:
        raise ValueError('No feasible fixed conservative buffer')
    return selected['delta'], rows


def clean_metrics(onset_probability, truth_onset, truth, prediction, groups, unconditional):
    hat = onset_probability >= .5
    positive, negative = truth_onset.astype(bool), ~truth_onset.astype(bool)
    tp, fn, fp, tn = [int(mask.sum()) for mask in (positive & hat, positive & ~hat, negative & hat, negative & ~hat)]
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    rates = []
    for group in np.unique(groups):
        mask = groups == group
        pos, neg = mask & positive, mask & negative
        rates.append(dict(fpr=float(hat[neg].mean()) if neg.any() else None,
            recall=float(hat[pos].mean()) if pos.any() else None,
            precision=float(positive[mask & hat].mean()) if (mask & hat).any() else None))
    equal = {}
    for name in ('fpr', 'recall', 'precision'):
        valid = [row[name] for row in rates if row[name] is not None]
        equal[name] = dict(value=float(np.mean(valid)) if valid else None,
            valid_groups=len(valid), total_groups=len(rates))
    error = (onset_probability - truth_onset) ** 2
    reference_error = (unconditional - truth_onset) ** 2
    return dict(n=len(truth), positive_n=int(positive.sum()), negative_n=int(negative.sum()),
        fpr=ratio(fp, fp + tn), onset_recall=ratio(tp, tp + fn), onset_precision=ratio(tp, tp + fp),
        onset_confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn), brier=float(error.mean()),
        unconditional_brier=float(reference_error.mean()), group_equal=equal,
        group_equal_brier=float(pd.Series(error).groupby(groups).mean().mean()),
        group_equal_unconditional_brier=float(pd.Series(reference_error).groupby(groups).mean().mean()),
        recovery=metrics(truth, prediction, groups))


def populations(truth, prediction, groups, polluted, covered):
    masks = dict(all=np.ones(len(truth), bool), polluted=polluted, clean=~polluted,
        matched_polluted=polluted & covered)
    return {name: metrics(truth[mask], prediction[mask], groups[mask]) for name, mask in masks.items()}


def validate_reuse(source_run=SOURCE_RUN):
    report_path = source_run / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    verification_path = source_run / 'verification.json'
    verified = json.loads(verification_path.read_text(encoding='utf-8'))
    if verified['status'] != 'passed' or report['status'] != 'completed_pending_verification':
        raise ValueError('Source experiment not independently verified')
    old_manifest = verified['independent_frozen_sha256']
    if old_manifest[str(report_path)] != sha(report_path):
        raise ValueError('Verified source report changed')
    if report['frozen_sha256'][str(SOURCE)] != SOURCE_SHA or sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Frozen source hash mismatch')
    reuse = {str(report_path): sha(report_path), str(verification_path): sha(verification_path)}
    names = ['train.npz', 'dev.npz', 'eligible.parquet', 'exclusions.parquet',
        'chemistry_features.npy', 'prediction.npz', 'buffer_grid.json']
    names += [f'chemistry_aqi_h{hour:02}.txt' for hour in range(1, 25)]
    for name in names:
        path = source_run / name
        digest = sha(path)
        if digest != report['artifact_sha256'][name] or digest != old_manifest[str(path)]:
            raise ValueError(f'Frozen verified reused artifact changed: {name}')
        reuse[str(path)] = digest
    for path, digest in report['frozen_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Old frozen source/dependency changed: {path}')
        reuse[path] = digest
    for path, digest in verified['independent_frozen_sha256'].items():
        if path.endswith('.py'):
            if sha(path) != digest:
                raise ValueError(f'Old independent verifier dependency changed: {path}')
            reuse[path] = digest
    return report, reuse


def dependency_files():
    files = []
    for package in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib', 'pyarrow'):
        distribution = importlib.metadata.distribution(package)
        files.extend(Path(distribution.locate_file(item)).resolve() for item in distribution.files
            if item.name == 'RECORD')
    from sklearn.ensemble import _forest
    files.append(Path(_forest.__file__).resolve())
    return files


def support_summary(table, usage, polluted):
    result = {}
    for name, mask in dict(all=np.ones(len(table), bool), polluted=polluted, clean=~polluted).items():
        result[name] = dict(n=int(mask.sum()))
        for column in ('nonzero_support', 'effective_support', 'distinct_processes', 'max_process_weight'):
            values = table.loc[mask, column].to_numpy()
            result[name][column] = dict(min=float(values.min()), median=float(np.median(values)),
                mean=float(values.mean()), p95=float(np.quantile(values, .95)), max=float(values.max()))
    used = usage.query_support_count > 0
    result['cross_query'] = dict(train_rows_used=int(used.sum()),
        train_rows_used_multiple_queries=int((usage.query_support_count > 1).sum()),
        max_query_support_count=int(usage.query_support_count.max()),
        mean_query_support_count_used=float(usage.loc[used, 'query_support_count'].mean()),
        total_nonzero_pairs=int(usage.query_support_count.sum()))
    return result


def build_report(products, curve323, curve76, train, dev, support, usage, old_report):
    truth, groups = dev['labels'], dev['groups']
    n = len(truth)
    covered = np.isfinite(products['cdf']).all(axis=1)
    polluted = dev['origin_aqi'] > 100
    matched = polluted & covered
    delta323, grid323 = select_buffer(curve323[polluted], truth[polluted], groups[polluted])
    if delta323 != old_report['selected_buffer']:
        raise ValueError('Reused conservative 323 buffer changed')
    delta76, grid76 = select_buffer(curve76[polluted], truth[polluted], groups[polluted])
    conservative323 = curve323.astype(float) + delta323
    conservative76 = curve76.astype(float) + delta76
    raw323_metrics = metrics(truth[matched], recovery(curve323[matched]), groups[matched])
    grid = [dict(theta=theta, **metrics(truth[matched], decision(products['cdf'][matched], theta), groups[matched]))
        for theta in THRESHOLDS]
    selected = select_theta(grid, raw323_metrics['early_fraction'])
    theta = selected['theta'] if selected is not None else 1.000001
    prediction = decision(products['cdf'], theta)
    unconditional_cdf = np.mean(train['labels'][:, None] <= np.arange(1, 20), axis=0)
    unconditional_onset = float(np.mean((train['paths'][:, :, 2] > 100).any(axis=1)))
    products.update(pred_start=prediction, covered=covered, raw_curve323=curve323,
        conservative_curve323=conservative323, raw_curve76=curve76, conservative_curve76=conservative76,
        unconditional_cdf=unconditional_cdf, unconditional_onset=np.array(unconditional_onset))
    comparators = {}
    for name, raw, conservative in [('lgbm323', curve323, conservative323), ('lgbm76', curve76, conservative76)]:
        comparators[name] = dict(raw=populations(truth, recovery(raw), groups, polluted, covered),
            conservative=populations(truth, recovery(conservative), groups, polluted, covered))
    learned = populations(truth, prediction, groups, polluted, covered)
    probability = probability_metrics(products['cdf'][matched], truth[matched], groups[matched])
    unconditional = probability_metrics(np.broadcast_to(unconditional_cdf, products['cdf'][matched].shape),
        truth[matched], groups[matched])
    clean = ~polluted & covered
    truth_onset = (dev['paths'][:, :, 2] > 100).any(axis=1)
    clean_score = clean_metrics(products['onset_probability'][clean], truth_onset[clean], truth[clean],
        prediction[clean], groups[clean], unconditional_onset)
    concentration = {}
    for name, mask in dict(all=covered, polluted=matched, clean=clean).items():
        error = np.abs(products['path_mean'][mask] - dev['paths'][mask])
        concentration[name] = {channel: dict(n=int(mask.sum()), hour_cells=int(mask.sum() * 24),
            mae=float(error[:, :, index].mean()), lead_mae=error[:, :, index].mean(axis=0).tolist(),
            group_equal_mae=float(pd.Series(error[:, :, index].mean(axis=1)).groupby(groups[mask]).mean().mean()),
            units='AQI' if channel == 'aqi' else 'ug/m3') for index, channel in enumerate(CHANNELS)}
    score = dict(comparators=comparators, learned=learned, probability=probability,
        unconditional_probability=unconditional, clean=clean_score, concentration=concentration,
        support=support_summary(support, usage, polluted))
    new = learned['matched_polluted']
    reference = comparators['lgbm323']['conservative']['matched_polluted']
    gates = dict(coverage_100=bool(covered.all()), actual_detection=selected is not None,
        mae_10pct=bool(new['mae'] <= reference['mae'] * .9),
        recall_5pp=bool(new['recall'] is not None and reference['recall'] is not None
            and new['recall'] >= reference['recall'] + .05),
        fpr_5pct=bool(new['fpr'] is not None and new['fpr'] <= .05),
        year_mae_not_worse=bool(new['mae'] <= reference['mae']),
        process_mae_not_worse=bool(new['group_equal_mae'] <= reference['group_equal_mae']),
        **{f'brier_{horizon}': bool(probability[f'brier_{horizon}'] <= unconditional[f'brier_{horizon}'])
            for horizon in (6, 12, 19)},
        clean_fpr_5pct=bool(clean_score['fpr'] is not None and clean_score['fpr'] <= .05),
        clean_brier=bool(clean_score['brier'] <= clean_score['unconditional_brier']))
    row = pd.DataFrame(dict(dev_row=np.arange(n), timestamp=pd.to_datetime(dev['timestamps'], utc=True),
        group=groups, polluted=polluted, covered=covered, true_start=truth, pred_start=prediction,
        raw_start323=recovery(curve323), conservative_start323=recovery(conservative323),
        raw_start76=recovery(curve76), conservative_start76=recovery(conservative76),
        onset_probability=products['onset_probability'], true_onset=truth_onset))
    row['absolute_error'] = np.abs(prediction - truth)
    row['reference_absolute_error'] = np.abs(row.conservative_start323 - truth)
    row['reference76_absolute_error'] = np.abs(row.conservative_start76 - truth)
    group = row.groupby(['polluted', 'group'], as_index=False).agg(n=('dev_row', 'size'),
        mae=('absolute_error', 'mean'), reference_mae=('reference_absolute_error', 'mean'),
        reference76_mae=('reference76_absolute_error', 'mean'))
    reliability = []
    for horizon in (6, 12, 19):
        probability_column = products['cdf'][matched, horizon - 1]
        target = truth[matched] <= horizon
        for bin_index in range(10):
            mask = (probability_column >= bin_index / 10) & ((probability_column < (bin_index + 1) / 10)
                if bin_index < 9 else (probability_column <= 1))
            reliability.append(dict(horizon=horizon, bin=bin_index, n=int(mask.sum()),
                mean_probability=float(probability_column[mask].mean()) if mask.any() else None,
                observed_fraction=float(target[mask].mean()) if mask.any() else None))
    return dict(metrics=score, gates=gates, preliminary_development_pass=all(gates.values()),
        failure_reasons=[name for name, passed in gates.items() if not passed],
        selected_theta=selected['theta'] if selected is not None else None,
        selected_buffers=dict(lgbm323=delta323, lgbm76=delta76)), grid, grid323, grid76, row, group, pd.DataFrame(reliability)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Run only after specification and quality approval')
    parser.add_argument('--preflight-only', action='store_true', help='Verify frozen reuse provenance without fitting')
    args = parser.parse_args()
    if not args.execute and not args.preflight_only:
        parser.error('Choose --preflight-only or an explicitly approved --execute')
    start = time.perf_counter()
    old_report, reuse = validate_reuse()
    dependency = [SPEC, PLAN, Path(__file__), HERE / 'test_experiment.py',
        HERE / 'verify_independent.py', HERE / 'test_verify_independent.py', *dependency_files()]
    frozen = {str(path): sha(path) for path in dependency}
    frozen.update(reuse)
    if args.preflight_only:
        event('preflight_passed', source_run=str(SOURCE_RUN), reused_artifacts=len(reuse),
            frozen_files=len(frozen), training_started=False)
        return
    run = HERE / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='running', run_dir=str(run), source_run=str(SOURCE_RUN),
        source_sha256=SOURCE_SHA, frozen_sha256=frozen, reuse_verified_sha256=reuse,
        versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
            sklearn=sklearn.__version__, lightgbm=lgb.__version__, joblib=joblib.__version__,
            pyarrow=importlib.metadata.version('pyarrow')),
        forest_config=FOREST_CONFIG, lgbm76_config=LGBM_CONFIG, quantiles=QUANTILES,
        artifact_schema=ARTIFACT_SCHEMA, train_years=list(range(2014, 2020)), dev_year=2020,
        selected_on_development=True, blind_test=False, deployed=False, models={},
        qualifications_reference=str(SOURCE_RUN / 'eligible.parquet'),
        exclusions_reference=str(SOURCE_RUN / 'exclusions.parquet'),
        train_reference=str(SOURCE_RUN / 'train.npz'), dev_reference=str(SOURCE_RUN / 'dev.npz'),
        chemistry_features_reference=str(SOURCE_RUN / 'chemistry_features.npy'))
    write_json(run / 'report.json', report)
    event('run_created', run_dir=str(run))
    try:
        with np.load(SOURCE_RUN / 'train.npz', allow_pickle=False) as bundle:
            train = {name: bundle[name] for name in bundle.files}
        with np.load(SOURCE_RUN / 'dev.npz', allow_pickle=False) as bundle:
            dev = {name: bundle[name] for name in bundle.files}
        for item, name in ((train, 'train'), (dev, 'dev')):
            if item['features'].shape != (len(item['indices']), 76) or not np.isfinite(item['features']).all():
                raise ValueError(f'Invalid frozen 76-feature population: {name}')
            np.testing.assert_array_equal(item['labels'], recovery(item['paths'][:, :, 2]))
        forest = RandomForestRegressor(**FOREST_CONFIG)
        event('forest_fit', train=len(train['features']), dev=len(dev['features']))
        forest.fit(train['features'], train['paths'][:, :, 2])
        forest_prediction = forest.predict(dev['features'])
        model_path = run / 'forest.joblib'
        joblib.dump(forest, model_path, compress=3)
        reloaded = joblib.load(model_path)
        report['forest_reload_max_absolute_error'] = assert_forest_equivalent(
            forest_prediction, reloaded.predict(dev['features']))
        del reloaded
        report['models'][model_path.name] = sha(model_path)
        bootstrap, train_leaves, dev_leaves = forest_structure(forest, train['features'], dev['features'])
        np.savez_compressed(run / 'forest_structure.npz', bootstrap_indices=bootstrap,
            train_leaves=train_leaves, dev_leaves=dev_leaves)
        lookup = leaf_index(bootstrap, train_leaves)
        blocks, products = [], []
        # Only one query's dense accumulator and a small batch exist at a time.
        for begin in range(0, len(dev_leaves), 128):
            end = min(begin + 128, len(dev_leaves))
            weight = forest_weights(lookup, dev_leaves[begin:end], len(train['features']))
            product = path_products(weight, train['paths'])
            np.testing.assert_allclose(product['path_mean'][:, :, 2], forest_prediction[begin:end],
                atol=1e-10, rtol=1e-12)
            blocks.append(weight)
            products.append(product)
            event('weighted_paths', done=end, total=len(dev_leaves), nonzero_batch=int(weight.nnz))
        weights = sparse.vstack(blocks, format='csr')
        del blocks, lookup, forest
        sparse.save_npz(run / 'weights.npz', weights, compressed=True)
        product = {name: np.concatenate([batch[name] for batch in products]) for name in products[0]}
        del products
        product['forest_aqi_mean'] = forest_prediction
        support, usage = support_diagnostics(weights, train['groups'])
        support.to_parquet(run / 'support.parquet', index=False)
        usage.to_parquet(run / 'train_support_usage.parquet', index=False)
        del weights
        chemistry = np.load(SOURCE_RUN / 'chemistry_features.npy', mmap_mode='r', allow_pickle=False)
        chemistry_dev = chemistry[len(train['indices']):]
        if chemistry.shape != (len(train['indices']) + len(dev['indices']), 323):
            raise ValueError('Frozen chemistry feature population shape changed')
        with np.load(SOURCE_RUN / 'prediction.npz', allow_pickle=False) as previous:
            curve323 = previous['raw_curve'].copy()
        for hour in range(24):
            path = SOURCE_RUN / f'chemistry_aqi_h{hour + 1:02}.txt'
            model = lgb.Booster(model_str=path.read_text(encoding='utf-8'))
            prediction = model.predict(chemistry_dev, num_threads=4).clip(min=0).astype(np.float32)
            np.testing.assert_array_equal(prediction, curve323[:, hour])
            del model
        event('reused323_prediction_verified', leads=24)
        curve76 = np.empty((len(dev['indices']), 24), dtype=np.float32)
        for hour in range(24):
            model = lgb.LGBMRegressor(**LGBM_CONFIG)
            model.fit(train['features'], train['paths'][:, hour, 2])
            prediction = model.predict(dev['features']).clip(min=0).astype(np.float32)
            path = run / f'lgbm76_aqi_h{hour + 1:02}.txt'
            path.write_text(model.booster_.model_to_string(), encoding='utf-8')
            reload = lgb.Booster(model_str=path.read_text(encoding='utf-8'))
            np.testing.assert_array_equal(prediction,
                reload.predict(dev['features'], num_threads=4).clip(min=0).astype(np.float32))
            curve76[:, hour] = prediction
            report['models'][path.name] = sha(path)
            model.booster_.free_dataset()
            del model, reload
            write_json(run / 'report.json', report)
            event('lgbm76', lead=hour + 1)
        update, grid, grid323, grid76, row, group, reliability = build_report(
            product, curve323, curve76, train, dev, support, usage, old_report)
        np.savez_compressed(run / 'prediction.npz', **product)
        write_json(run / 'grid.json', grid)
        write_json(run / 'buffer_grid323.json', grid323)
        write_json(run / 'buffer_grid76.json', grid76)
        row.to_parquet(run / 'row_metrics.parquet', index=False)
        group.to_parquet(run / 'group_metrics.parquet', index=False)
        reliability.to_parquet(run / 'reliability.parquet', index=False)
        for path, digest in frozen.items():
            if sha(path) != digest:
                raise ValueError(f'Frozen input/dependency changed during execution: {path}')
        report.update(update)
        report.update(status='pending_independent_verification',
            counts=dict(train=len(train['indices']), dev=len(dev['indices']),
                covered=int(row.covered.sum()), no_prediction=int((~row.covered).sum()),
                polluted=int(row.polluted.sum()), clean=int((~row.polluted).sum()),
                exclusions=old_report['counts']['exclusions']),
            elapsed_seconds=time.perf_counter() - start,
            forest_mean_predict_max_absolute_error=float(np.max(np.abs(product['path_mean'][:, :, 2] - forest_prediction))))
    except BaseException as error:
        report.update(status='execution_failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['artifact_sha256'] = {path.name: sha(path) for path in run.iterdir()
            if path.is_file() and path.name != 'report.json'}
        write_json(run / 'report.json', report)
    event('pending_independent_verification', run_dir=str(run),
        preliminary_development_pass=report['preliminary_development_pass'], failure_reasons=report['failure_reasons'])


if __name__ == '__main__':
    main()
