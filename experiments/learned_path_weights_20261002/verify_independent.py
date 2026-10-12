"""Independent source, bootstrap, weighted-path and development audit.

Only the prior independent verifiers are reused, never predictor algorithms.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/processed/regional_air_models_full_history/city_hourly.parquet'
SOURCE_SHA = 'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'
REFERENCE_RUN = ROOT/'experiments/empirical_paths_20261002/runs/20261002T121844976298Z'
REFERENCE_VERIFIER = ROOT/'experiments/empirical_paths_20261002/verify_independent.py'
CHEMISTRY_DIR = ROOT/'experiments/continuous_trajectory_20260929'
SPEC = ROOT/'docs/superpowers/specs/2026-10-02-learned-path-weights-design.md'
_REFERENCE = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference():
    global _REFERENCE
    if _REFERENCE is None:
        spec = importlib.util.spec_from_file_location('independent_previous_path_audit', REFERENCE_VERIFIER)
        _REFERENCE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_REFERENCE)
    return _REFERENCE


def rebuild(raw):
    return load_reference().rebuild(raw)


def recovery(curves):
    return load_reference().recovery(curves)


def metric_schema(actual, prediction, groups):
    return load_reference().metric_schema(np.asarray(actual), np.asarray(prediction), np.asarray(groups))


def probability_schema(actual, cdf, groups):
    return load_reference().probability_schema(np.asarray(actual), np.asarray(cdf), np.asarray(groups))


def group_average(values, groups):
    return load_reference().group_average(np.asarray(values), np.asarray(groups))


def equal_rates(rows, keys):
    return load_reference().equal_rates(rows, keys)


def close(actual, expected, label='value'):
    return load_reference().close(actual, expected, label)


def detached_metrics(values):
    return copy.deepcopy(values)


def freeze_paths(paths):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def assert_frozen(frozen):
    for path, digest in frozen.items():
        assert sha(path) == digest, f'Frozen input changed: {path}'


def choose(rows, early_limit, key, require_detection=False):
    allowed = []
    for row in rows:
        cm = row['confusion']
        valid_classes = cm['tp']+cm['fn'] > 0 and cm['fp']+cm['tn'] > 0
        if valid_classes and row['fpr'] is not None and row['fpr'] <= .05 and row['early_fraction'] <= early_limit:
            if not require_detection or cm['tp'] > 0:
                allowed.append(row)
    return min(allowed, key=lambda row: (row['mae'], -(row['recall'] or 0), row['fpr'], -row[key])) if allowed else None


def decision(cdf, theta):
    hits = np.asarray(cdf) >= theta
    return np.where(hits.any(axis=1), hits.argmax(axis=1)+1, 20)


def recompute_weights(bootstrap, train_leaves, query_leaves):
    bootstrap, train_leaves, query_leaves = map(np.asarray, (bootstrap, train_leaves, query_leaves))
    trees, count = bootstrap.shape
    if train_leaves.shape != (count, trees) or query_leaves.shape != (trees,):
        raise ValueError('Bootstrap and leaf dimensions disagree')
    if np.any(bootstrap < 0) or np.any(bootstrap >= count):
        raise ValueError('Invalid bootstrap training identity')
    result = np.zeros(count, dtype=np.float64)
    for tree in range(trees):
        multiplicity = np.bincount(bootstrap[tree], minlength=count)
        membership = train_leaves[:, tree] == query_leaves[tree]
        mass = multiplicity*membership
        denominator = mass.sum()
        if denominator == 0:
            raise ValueError('Query leaf contains no bootstrap training mass')
        result += mass/denominator/trees
    if not np.isclose(result.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError('Forest weights do not sum to one')
    return result/result.sum()


def distribution(weights, paths, groups):
    weights, paths, groups = np.asarray(weights), np.asarray(paths), np.asarray(groups)
    if weights.ndim != 1 or paths.shape != (len(weights), 24, 3) or groups.shape != weights.shape:
        raise ValueError('Path and weight dimensions disagree')
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError('Nonnegative normalized finite weights required')
    weights = weights/weights.sum()
    labels = recovery(paths[:, :, 2])
    pmf = np.bincount(labels-1, weights=weights, minlength=20)
    pmf = pmf/pmf.sum()
    support = weights > 0
    group_mass = np.array([weights[groups == key].sum() for key in np.unique(groups[support])])
    return dict(pmf=pmf, cdf=roundoff_bounds(np.cumsum(pmf[:19])), onset_probability=float(roundoff_bounds(weights@np.any(paths[:, :, 2] > 100, axis=1))),
                path_mean=np.einsum('i,ihc->hc', weights, paths), effective_support=float(1/np.sum(weights**2)),
                nonzero_support=int(support.sum()), selected_group_count=len(group_mass), max_group_weight=float(group_mass.max()))


def roundoff_bounds(probability):
    probability = np.asarray(probability)
    if not np.isfinite(probability).all() or np.any(probability < -1e-12) or np.any(probability > 1+1e-12):
        raise ValueError('Probability exceeds roundoff-only bounds')
    result = probability.copy()
    result = np.where(np.abs(result) <= 1e-12, 0., result)
    return np.where(np.abs(result-1.) <= 1e-12, 1., result)


def weighted_quantiles(paths, weights, quantiles=(.1, .5, .9)):
    result = np.empty((len(quantiles), 24, 3), dtype=float)
    for hour in range(24):
        for channel in range(3):
            order = np.argsort(paths[:, hour, channel], kind='stable')
            cumulative = np.cumsum(weights[order])
            for q, probability in enumerate(quantiles):
                index = min(int(np.searchsorted(cumulative, probability, side='left')), len(order)-1)
                result[q, hour, channel] = paths[order[index], hour, channel]
    return result


def dependency_versions():
    names = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'lightgbm', 'pyarrow')
    return dict(python=sys.version, **{name: importlib.metadata.version(name) for name in names})


def freeze_dependencies():
    paths = [REFERENCE_VERIFIER, REFERENCE_VERIFIER.with_name('test_verify_independent.py'),
             CHEMISTRY_DIR/'verify_delayed_chemistry.py', CHEMISTRY_DIR/'verify_issue_delay.py']
    from sklearn.ensemble import _forest
    paths.append(Path(_forest.__file__))
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'lightgbm', 'pyarrow'):
        dist = importlib.metadata.distribution(name)
        paths += [Path(dist.locate_file(file)) for file in dist.files or [] if str(file).endswith('.dist-info/RECORD')]
    return paths


def verify_source(source_run, audit):
    assert sha(SOURCE) == SOURCE_SHA
    prior = json.loads((source_run/'report.json').read_text('utf-8'))
    prior_audit = json.loads((source_run/'verification.json').read_text('utf-8'))
    assert prior_audit['status'] == 'passed', 'Reused run lacks an independent passing audit'
    for path, digest in prior['frozen_sha256'].items():
        assert sha(path) == digest, path
    for name, digest in prior['artifact_sha256'].items():
        assert sha(source_run/name) == digest, name
    for path, digest in prior_audit['independent_frozen_sha256'].items():
        assert sha(path) == digest, path
    raw = pd.read_parquet(SOURCE)
    raw = raw.loc[(raw.timestamp.dt.year >= 2014) & (raw.timestamp.dt.year <= 2020)].copy()
    raw, dates, data, reasons = rebuild(raw)
    eligible = pd.read_parquet(source_run/'eligible.parquet')
    np.testing.assert_array_equal(eligible.source_index, data['indices'])
    np.testing.assert_array_equal(pd.DatetimeIndex(eligible.timestamp).asi8, data['timestamps'])
    np.testing.assert_array_equal(eligible.true_start, data['labels'])
    excluded = pd.read_parquet(source_run/'exclusions.parquet')
    exclusion_indices = np.flatnonzero(reasons != 'eligible')
    np.testing.assert_array_equal(excluded.source_index, exclusion_indices)
    names = {'boundary': 'partition_boundary', 'discontinuous': 'hour_gap'}
    np.testing.assert_array_equal(excluded.reason, [names.get(reason, reason) for reason in reasons[exclusion_indices]])
    np.testing.assert_array_equal(pd.DatetimeIndex(excluded.timestamp).asi8, dates.asi8[exclusion_indices])
    train = dates[data['indices']].year <= 2019
    splits = {}
    for name, mask in (('train', train), ('dev', ~train)):
        expected = {key: value[mask] for key, value in data.items()}
        saved = dict(np.load(source_run/f'{name}.npz', allow_pickle=False))
        for key, value in expected.items():
            if key == 'groups':
                np.testing.assert_array_equal(pd.factorize(value)[0], pd.factorize(saved[key])[0])
            else:
                np.testing.assert_allclose(value, saved[key], rtol=0, atol=1e-12, err_msg=f'{name}/{key}')
        splits[name] = expected
    audit['checks'].append('Source SHA and prior frozen audit; independently rebuilt source qualification, exclusions, 76 monthly features, labels, paths and process equivalence')
    audit['source_rows_restricted'] = len(raw)
    return splits['train'], splits['dev'], raw, dates


def verify_forest(run, train, dev, pred, audit):
    model = joblib.load(run/'forest.joblib')
    fixed = dict(n_estimators=300, criterion='squared_error', bootstrap=True, min_samples_leaf=30,
                 max_features=1.0, max_depth=None, random_state=42, n_jobs=4)
    params = model.get_params()
    assert all(params[key] == value for key, value in fixed.items()), params
    assert params['max_samples'] is None and model.n_features_in_ == 76 and model.n_outputs_ == 24
    assert len(model.estimators_) == 300
    structure = dict(np.load(run/'forest_structure.npz', allow_pickle=False))
    bootstrap = structure['bootstrap_indices']
    assert bootstrap.shape == (300, len(train['indices']))
    from sklearn.utils import check_random_state
    for tree, estimator in enumerate(model.estimators_):
        rebuilt = check_random_state(estimator.random_state).randint(0, len(train['indices']), len(train['indices']), dtype=np.int32)
        np.testing.assert_array_equal(bootstrap[tree], rebuilt)
    np.testing.assert_array_equal(model.apply(train['features']), structure['train_leaves'])
    np.testing.assert_array_equal(model.apply(dev['features']), structure['dev_leaves'])
    weights = sparse.load_npz(run/'weights.npz').tocsr()
    assert weights.shape == (len(dev['indices']), len(train['indices'])) and weights.has_canonical_format
    assert np.isfinite(weights.data).all() and np.all(weights.data > 0)
    np.testing.assert_allclose(np.asarray(weights.sum(axis=1)).ravel(), 1., rtol=0, atol=1e-12)
    samples = np.random.default_rng(42).choice(len(dev['indices']), size=min(100, len(dev['indices'])), replace=False)
    for query in samples:
        expected = recompute_weights(bootstrap, structure['train_leaves'], structure['dev_leaves'][query])
        saved = weights.getrow(query)
        np.testing.assert_allclose(saved.toarray()[0], expected, rtol=1e-11, atol=1e-14,
                                   err_msg=f'bootstrap-leaf weight/query{query}')
        quantiles = weighted_quantiles(train['paths'][saved.indices], saved.data)
        np.testing.assert_allclose(pred['path_quantiles'][query], quantiles, rtol=0, atol=0)
    path_mean = np.asarray(weights@train['paths'].reshape(len(train['indices']), 72)).reshape(len(dev['indices']), 24, 3)
    np.testing.assert_allclose(path_mean, pred['path_mean'], rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(path_mean[:, :, 2], pred['forest_aqi_mean'], rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(path_mean[:, :, 2], model.predict(dev['features']), rtol=1e-11, atol=1e-10)
    onehot = np.eye(20)[train['labels']-1]
    pmf = np.asarray(weights@onehot)
    pmf /= pmf.sum(axis=1, keepdims=True)
    cdf = roundoff_bounds(np.cumsum(pmf[:, :19], axis=1))
    np.testing.assert_allclose(pred['pmf'], pmf, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pred['cdf'], cdf, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pred['pmf'][:, 19], np.asarray(weights@(train['labels'] == 20)), rtol=1e-12, atol=1e-12)
    assert np.all(pmf >= 0) and np.all(np.diff(cdf, axis=1) >= -1e-15)
    onset = roundoff_bounds(np.asarray(weights@np.any(train['paths'][:, :, 2] > 100, axis=1)))
    np.testing.assert_allclose(pred['onset_probability'], onset, rtol=1e-12, atol=1e-12)
    support = pd.read_parquet(run/'support.parquet')
    np.testing.assert_array_equal(support.dev_row, np.arange(len(dev['indices'])))
    np.testing.assert_array_equal(support.nonzero_support, np.diff(weights.indptr))
    np.testing.assert_allclose(support.effective_support, 1/np.asarray(weights.power(2).sum(axis=1)).ravel(), rtol=1e-11, atol=1e-10)
    group_codes, group_names = pd.factorize(train['groups'])
    group_members = sparse.csr_matrix((np.ones(len(group_codes)), (np.arange(len(group_codes)), group_codes)), shape=(len(group_codes), len(group_names)))
    group_weights = (weights@group_members).tocsr()
    np.testing.assert_array_equal(support.distinct_processes, np.diff(group_weights.indptr))
    np.testing.assert_allclose(support.max_process_weight, group_weights.max(axis=1).toarray().ravel(), rtol=1e-11, atol=1e-12)
    usage = pd.read_parquet(run/'train_support_usage.parquet')
    np.testing.assert_array_equal(usage.train_row, np.arange(len(train['indices'])))
    np.testing.assert_array_equal(usage.query_support_count, np.asarray((weights > 0).sum(axis=0)).ravel())
    np.testing.assert_allclose(usage.total_weight, np.asarray(weights.sum(axis=0)).ravel(), rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(pred['unconditional_cdf'], (train['labels'][:, None] <= np.arange(1, 20)).mean(axis=0), rtol=0, atol=0)
    assert pred['unconditional_onset'] == np.any(train['paths'][:, :, 2] > 100, axis=1).mean()
    np.testing.assert_array_equal(pred['median_start'], np.argmax(np.cumsum(pmf, axis=1) >= .5, axis=1)+1)
    assert pred['path_quantiles'].shape == (len(dev['indices']), 3, 24, 3)
    assert np.isfinite(pred['path_quantiles']).all() and np.all(np.diff(pred['path_quantiles'], axis=1) >= 0)
    assert np.all(pred['path_quantiles'] >= train['paths'].min(axis=0)) and np.all(pred['path_quantiles'] <= train['paths'].max(axis=0))
    np.testing.assert_array_equal(pred['covered'], np.ones(len(dev['indices']), dtype=bool))
    audit.update(queries_sampled=len(samples), sparse_weight_nonzeros=int(weights.nnz), independently_reloaded_forest=True)
    audit['checks'].append('All 300 bootstrap samples, train/query leaf identities; seed42 100 independently recalculated multiplicity-weighted tree averages and quantiles; all CSR normalization, source distributions, censor mass, onset probability, process support/reuse and RF AQI mean identity')
    return cdf


def verify_models(run, source_run, train, dev, raw, dates, pred, audit):
    sys.path.insert(0, str(CHEMISTRY_DIR))
    from verify_delayed_chemistry import reconstruct
    x, _ = reconstruct(raw, dates, np.r_[train['indices'], dev['indices']], 2019, 1)
    np.testing.assert_allclose(x, np.load(source_run/'chemistry_features.npy'), rtol=0, atol=0)
    previous = dict(np.load(source_run/'prediction.npz', allow_pickle=False))
    np.testing.assert_array_equal(pred['raw_curve323'], previous['raw_curve'])
    np.testing.assert_array_equal(pred['conservative_curve323'], previous['conservative_curve'])
    for width, features, prefix, directory, field in ((323, x[len(train['indices']):], 'chemistry', source_run, 'raw_curve323'),
                                                     (76, dev['features'], 'lgbm76', run, 'raw_curve76')):
        for lead in range(24):
            name = f'{prefix}_aqi_h{lead+1:02}.txt'
            model = lgb.Booster(model_str=(directory/name).read_text('utf-8'))
            assert model.num_feature() == width and model.current_iteration() == 300
            assert model.params['objective'] == 'regression_l1' and model.params['num_leaves'] == 24
            assert np.isclose(float(model.params['learning_rate']), .04) and model.params['min_data_in_leaf'] == 30
            assert model.params['deterministic'] and model.params['force_col_wise']
            assert model.params['seed'] == 42 and model.params['num_threads'] == 4
            restored = model.predict(features, num_threads=4).clip(min=0).astype(np.float32)
            np.testing.assert_array_equal(restored, pred[field][:, lead])
    audit['checks'].append('Independent 323 chemistry feature reconstruction and 24 prior model re-inferences; all 24 matching-76-feature fixed LightGBM heads reloaded and re-inferred')


def main(run):
    run = Path(run).resolve()
    report = json.loads((run/'report.json').read_text('utf-8'))
    source_run = Path(report['source_run']).resolve()
    assert source_run == REFERENCE_RUN.resolve()
    paths = [Path(__file__), Path(__file__).with_name('test_verify_independent.py'), SOURCE, SPEC, run/'report.json']
    paths += freeze_dependencies()
    paths += [Path(path) for path in report['frozen_sha256']]
    paths += [run/name for name in report['artifact_sha256']]
    paths += [source_run/'report.json', source_run/'verification.json', source_run/'audit_manifest.json']
    previous = json.loads((source_run/'report.json').read_text('utf-8'))
    paths += [source_run/name for name in previous['artifact_sha256']]
    frozen, versions = freeze_paths(paths), dependency_versions()
    manifest = dict(frozen_sha256=frozen, versions=versions, reference_dependencies=[str(path) for path in freeze_dependencies()],
                    predictor_algorithms_imported=False)
    (run/'audit_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    audit = dict(status='running', verifier_sha256=sha(__file__), independent_frozen_sha256=frozen, checks=[],
                 limits=['Development selected after prior observed 2020 development failure; not a blind test',
                         'No independent retraining', 'Quantiles independently recomputed for seed42 100 queries; not coverage intervals',
                         'Independent reproducibility pass is not development-gate pass'])
    (run/'verification.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    try:
        for path, digest in report['frozen_sha256'].items():
            assert sha(path) == digest, path
        for name, digest in report['artifact_sha256'].items():
            assert sha(run/name) == digest, name
        for name, digest in report['models'].items():
            assert sha(run/name) == digest, name
        for path, digest in report['reuse_verified_sha256'].items():
            assert sha(path) == digest, path
            assert str(Path(path).resolve()) in frozen, path
        assert report['schema_version'] == 1 and report['status'] == 'pending_independent_verification'
        assert report['source_sha256'] == SOURCE_SHA
        assert report['train_years'] == list(range(2014, 2020)) and report['dev_year'] == 2020
        assert report['selected_on_development'] and not report['blind_test'] and not report['deployed']
        assert report['quantiles'] == [.1, .5, .9]
        assert set(report['models']) == {'forest.joblib'} | {f'lgbm76_aqi_h{hour:02}.txt' for hour in range(1, 25)}
        for field, name in (('qualifications_reference', 'eligible.parquet'), ('exclusions_reference', 'exclusions.parquet'),
                            ('train_reference', 'train.npz'), ('dev_reference', 'dev.npz'),
                            ('chemistry_features_reference', 'chemistry_features.npy')):
            assert Path(report[field]).resolve() == (source_run/name).resolve()
        version_keys = dict(numpy='numpy', pandas='pandas', scipy='scipy', sklearn='scikit-learn',
                            lightgbm='lightgbm', joblib='joblib', pyarrow='pyarrow', python='python')
        close(report['versions'], {key: versions[value] for key, value in version_keys.items()}, 'main_dependency_versions')
        close(report['forest_config'], dict(n_estimators=300, criterion='squared_error', bootstrap=True,
              min_samples_leaf=30, max_features=1.0, max_depth=None, random_state=42, n_jobs=4), 'fixed_forest_config')
        close(report['lgbm76_config'], dict(objective='regression_l1', n_estimators=300, learning_rate=.04,
              num_leaves=24, min_child_samples=30, random_state=42, n_jobs=4, verbosity=-1,
              deterministic=True, force_col_wise=True), 'fixed_lgbm76_config')
        train, dev, raw, dates = verify_source(source_run, audit)
        pred = dict(np.load(run/'prediction.npz', allow_pickle=False))
        cdf = verify_forest(run, train, dev, pred, audit)
        verify_models(run, source_run, train, dev, raw, dates, pred, audit)
        verify_reports(run, source_run, train, dev, pred, cdf, report, audit)
        assert_frozen(frozen)
        close(dependency_versions(), versions, 'dependencies_after_audit')
        audit['status'] = 'passed'
    except Exception as exc:
        audit.update(status='failed', error=repr(exc))
        raise
    finally:
        (run/'verification.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(audit, indent=2), flush=True)


def verify_reports(run, source_run, train, dev, pred, cdf, report, audit):
    truth, groups = dev['labels'], dev['groups']
    polluted, covered = dev['origin_aqi'] > 100, pred['covered']
    clean, matched = ~polluted, polluted & covered
    population_masks = dict(all=np.ones(len(truth), dtype=bool), polluted=polluted,
                            clean=clean, matched_polluted=matched)
    buffer_grids, comparators, starts = {}, {}, {}
    for name, suffix, grid_name in (('lgbm323', '323', 'buffer_grid323.json'),
                                    ('lgbm76', '76', 'buffer_grid76.json')):
        raw_curve = pred[f'raw_curve{suffix}']
        raw_start = recovery(raw_curve)
        original = metric_schema(truth[polluted], raw_start[polluted], groups[polluted])
        grid = [dict(delta=delta, **metric_schema(truth[polluted], recovery(raw_curve[polluted].astype(float)+delta), groups[polluted]))
                for delta in range(102)]
        close(grid, json.loads((run/grid_name).read_text('utf-8')), name+'/buffer_grid')
        selected = choose(grid, original['early_fraction'], 'delta')
        assert selected is not None
        assert report['selected_buffers'][name] == selected['delta']
        conservative = raw_curve.astype(float)+selected['delta']
        np.testing.assert_array_equal(pred[f'conservative_curve{suffix}'], conservative)
        conservative_start = recovery(conservative)
        comparators[name] = {kind: {population: metric_schema(truth[mask], values[mask], groups[mask])
                                  for population, mask in population_masks.items()}
                             for kind, values in (('raw', raw_start), ('conservative', conservative_start))}
        buffer_grids[name], starts[name] = grid, dict(raw=raw_start, conservative=conservative_start)
    old_report = json.loads((source_run/'report.json').read_text('utf-8'))
    close(report['selected_buffers']['lgbm323'], old_report['selected_buffer'], 'reused323buffer')
    close(buffer_grids['lgbm323'], json.loads((source_run/'buffer_grid.json').read_text('utf-8')), 'reused323buffergrid')
    theta_values = [round(i*.05, 2) for i in range(1, 21)]+[1.000001]
    grid = [dict(theta=theta, **metric_schema(truth[matched], decision(cdf[matched], theta), groups[matched]))
            for theta in theta_values]
    close(grid, json.loads((run/'grid.json').read_text('utf-8')), 'all_21_theta_grid_rows')
    early_limit = comparators['lgbm323']['raw']['polluted']['early_fraction']
    selected = choose(grid, early_limit, 'theta', require_detection=True)
    assert report['selected_theta'] == (selected['theta'] if selected else None)
    expected_start = decision(cdf, selected['theta'] if selected else 1.000001)
    np.testing.assert_array_equal(pred['pred_start'], expected_start)
    learned = {population: metric_schema(truth[mask], expected_start[mask], groups[mask])
               for population, mask in population_masks.items()}
    probability = probability_schema(truth[matched], cdf[matched], groups[matched])
    unconditional = probability_schema(truth[matched], np.broadcast_to(pred['unconditional_cdf'], cdf[matched].shape), groups[matched])
    truth_onset = np.any(dev['paths'][:, :, 2] > 100, axis=1)
    onset = pred['onset_probability']
    positive, negative = truth_onset[clean], ~truth_onset[clean]
    hat = onset[clean] >= .5
    tp, fn, fp, tn = [int(mask.sum()) for mask in (positive & hat, positive & ~hat, negative & hat, negative & ~hat)]
    ratio = lambda numerator, denominator: float(numerator/denominator) if denominator else None
    rates = []
    for group in np.unique(groups[clean]):
        mask = groups[clean] == group
        pos, neg = mask & positive, mask & negative
        rates.append(dict(fpr=float(hat[neg].mean()) if neg.any() else None,
                          recall=float(hat[pos].mean()) if pos.any() else None,
                          precision=float(positive[mask & hat].mean()) if (mask & hat).any() else None))
    error = (onset[clean]-truth_onset[clean])**2
    reference_error = (pred['unconditional_onset']-truth_onset[clean])**2
    clean_result = dict(n=int(clean.sum()), positive_n=int(positive.sum()), negative_n=int(negative.sum()),
                        fpr=ratio(fp, fp+tn), onset_recall=ratio(tp, tp+fn), onset_precision=ratio(tp, tp+fp),
                        onset_confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn), brier=float(error.mean()),
                        unconditional_brier=float(reference_error.mean()), group_equal=equal_rates(rates, ('fpr', 'recall', 'precision')),
                        group_equal_brier=group_average(error, groups[clean]),
                        group_equal_unconditional_brier=group_average(reference_error, groups[clean]), recovery=learned['clean'])
    concentration = {}
    for population in ('all', 'polluted', 'clean'):
        mask = population_masks[population]
        error = np.abs(pred['path_mean'][mask]-dev['paths'][mask])
        concentration[population] = {name: dict(n=int(mask.sum()), hour_cells=int(mask.sum()*24),
            mae=float(error[:, :, channel].mean()), lead_mae=error[:, :, channel].mean(axis=0).tolist(),
            group_equal_mae=group_average(error[:, :, channel].mean(axis=1), groups[mask]), units='AQI' if name == 'aqi' else 'ug/m3')
            for channel, name in enumerate(('pm2_5', 'pm10', 'aqi'))}
    support_rows = pd.read_parquet(run/'support.parquet')
    usage = pd.read_parquet(run/'train_support_usage.parquet')
    support_metrics = {}
    for population in ('all', 'polluted', 'clean'):
        mask = population_masks[population]
        item = dict(n=int(mask.sum()))
        for field in ('nonzero_support', 'effective_support', 'distinct_processes', 'max_process_weight'):
            values = support_rows[field].to_numpy()[mask]
            item[field] = dict(min=float(values.min()), median=float(np.median(values)), mean=float(values.mean()),
                               p95=float(np.quantile(values, .95)), max=float(values.max()))
        support_metrics[population] = item
    used = usage.query_support_count.to_numpy() > 0
    support_metrics['cross_query'] = dict(train_rows_used=int(used.sum()),
        train_rows_used_multiple_queries=int((usage.query_support_count.to_numpy() > 1).sum()),
        max_query_support_count=int(usage.query_support_count.max()),
        mean_query_support_count_used=float(usage.query_support_count.to_numpy()[used].mean()),
        total_nonzero_pairs=int(usage.query_support_count.sum()))
    expected = dict(comparators=comparators, learned=learned, probability=probability,
                    unconditional_probability=unconditional, clean=clean_result, concentration=concentration, support=support_metrics)
    audit['independently_recomputed_metrics'] = detached_metrics(expected)
    close(expected, report['metrics'], 'all_metrics')
    reference, candidate = comparators['lgbm323']['conservative']['matched_polluted'], learned['matched_polluted']
    gates = dict(coverage_100=bool(covered.all()), actual_detection=selected is not None,
        mae_10pct=candidate['mae'] <= reference['mae']*.9,
        recall_5pp=candidate['recall'] is not None and reference['recall'] is not None and candidate['recall'] >= reference['recall']+.05,
        fpr_5pct=candidate['fpr'] is not None and candidate['fpr'] <= .05,
        year_mae_not_worse=candidate['mae'] <= reference['mae'],
        process_mae_not_worse=candidate['group_equal_mae'] <= reference['group_equal_mae'],
        brier_6=probability['brier_6'] <= unconditional['brier_6'],
        brier_12=probability['brier_12'] <= unconditional['brier_12'],
        brier_19=probability['brier_19'] <= unconditional['brier_19'],
        clean_fpr_5pct=clean_result['fpr'] is not None and clean_result['fpr'] <= .05,
        clean_brier=clean_result['brier'] <= clean_result['unconditional_brier'])
    close(gates, report['gates'], 'all_gates')
    close([key for key, passed in gates.items() if not passed], report['failure_reasons'], 'failure_reasons')
    assert report['preliminary_development_pass'] == all(gates.values())
    counts = dict(train=len(train['indices']), dev=len(truth), covered=int(covered.sum()), no_prediction=int((~covered).sum()),
                  polluted=int(polluted.sum()), clean=int(clean.sum()), exclusions=len(pd.read_parquet(source_run/'exclusions.parquet')))
    close(counts, report['counts'], 'counts')
    assert report['forest_reload_max_absolute_error'] <= 1e-10
    close(report['forest_mean_predict_max_absolute_error'],
          float(np.max(np.abs(pred['path_mean'][:, :, 2]-pred['forest_aqi_mean']))), 'weighted_aqi_forest_error')
    rows = pd.read_parquet(run/'row_metrics.parquet')
    for key, values in dict(dev_row=np.arange(len(truth)), polluted=polluted, covered=covered, true_start=truth,
        pred_start=expected_start, raw_start323=starts['lgbm323']['raw'], conservative_start323=starts['lgbm323']['conservative'],
        raw_start76=starts['lgbm76']['raw'], conservative_start76=starts['lgbm76']['conservative'],
        onset_probability=onset, true_onset=truth_onset, absolute_error=np.abs(expected_start-truth),
        reference_absolute_error=np.abs(starts['lgbm323']['conservative']-truth),
        reference76_absolute_error=np.abs(starts['lgbm76']['conservative']-truth)).items():
        np.testing.assert_allclose(rows[key].to_numpy(), values, rtol=1e-12, atol=1e-12, err_msg=key)
    np.testing.assert_array_equal(pd.DatetimeIndex(rows.timestamp).asi8, dev['timestamps'])
    np.testing.assert_array_equal(pd.factorize(rows['group'])[0], pd.factorize(groups)[0])
    key_map = dict(zip(rows['group'], groups))
    rows['group'] = groups
    expected_groups = rows.groupby(['polluted', 'group'], as_index=False).agg(n=('dev_row', 'size'), mae=('absolute_error', 'mean'),
        reference_mae=('reference_absolute_error', 'mean'), reference76_mae=('reference76_absolute_error', 'mean'))
    saved_groups = pd.read_parquet(run/'group_metrics.parquet')
    saved_groups['group'] = saved_groups['group'].map(key_map)
    pd.testing.assert_frame_equal(expected_groups.sort_values(['polluted', 'group']).reset_index(drop=True),
                                  saved_groups.sort_values(['polluted', 'group']).reset_index(drop=True))
    reliability = []
    for horizon in (6, 12, 19):
        p, y = cdf[matched, horizon-1], truth[matched] <= horizon
        for bin_id in range(10):
            mask = (p >= bin_id/10) & (p <= 1 if bin_id == 9 else p < (bin_id+1)/10)
            reliability.append(dict(horizon=horizon, bin=bin_id, n=int(mask.sum()),
                mean_probability=float(p[mask].mean()) if mask.any() else None,
                observed_fraction=float(y[mask].mean()) if mask.any() else None))
    pd.testing.assert_frame_equal(pd.DataFrame(reliability), pd.read_parquet(run/'reliability.parquet'))
    audit['checks'].append('All learned/comparator full and matched populations; all clean/concentration/support/Brier/row/process/reliability metrics; both 102-row buffer grids and 21-row theta grid; frozen development selection, all 12 gates and truthful failure status')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
