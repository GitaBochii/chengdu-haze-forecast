"""Isolated full-recovery bridge: fixed archive versus maturity-safe AFT arms."""
import os

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.features import MEASUREMENT_COLUMNS, add_missingness_indicators
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.observed_business import build_business_states
from haze_forecast.observed_hourly_features import build_observed_hourly_features
from haze_forecast.interval_recovery_labels import interval_recovery_bounds
from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_bounds, paired_error_summary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GRID_PATH = ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet'
ORIGINS_PATH = ROOT / 'artifacts/task01_recovery_head/interval_business_labels_v1/origins.parquet'
ARCHIVE_PATH = ROOT / 'artifacts/strict_history_backtest/predictions.parquet'
HOSTED_PATH = ROOT / 'sites-haze/dist/history.json'
METADATA_PATH = ROOT / 'artifacts/cnemc_2025_deployment_candidate/model_metadata.json'
PINNED_SHA256 = {
    str(GRID_PATH): '52efbd1bfef81643c317c193895104baff61a3eee1b0ca465066e3e39cec473e',
    str(ORIGINS_PATH): '11aa1f83d6d8e3eb884c0f17a72a9428266192c6e8073481c74e9252f9efcf44',
    str(ARCHIVE_PATH): '016edd04e5cac982aa78d31a998a2bc8834215e67977e32e8ba1dd95b915dbd6',
    str(HOSTED_PATH): 'c457a1a96fd8f62b9601492d08f48461db7c6bf432ccea03f9af7f4274bc7214',
}
LEGACY_SOURCES = {
    'cnemc_2014': ROOT / 'data/processed/cnemc_chengdu_2014_training_features.parquet',
    'cnemc_2015_2021': ROOT / 'data/processed/training_features.parquet',
    'cnemc_2022': ROOT / 'data/processed/cnemc_chengdu_2022_training_features.parquet',
    'cnemc_2023': ROOT / 'data/processed/cnemc_chengdu_2023_training_features.parquet',
    'cnemc_2024_2025': ROOT / 'data/processed/cnemc_chengdu_2024_2025_training_features.parquet',
}
YEARS = (2024, 2025)
ARMS = ('archive', 'legacy_control', 'interval_candidate')
MODEL_CONFIG = dict(seed=42, num_boost_round=500, nthread=1)
AFT_PARAMETERS = dict(objective='survival:aft', eval_metric='aft-nloglik',
    aft_loss_distribution='normal', aft_loss_distribution_scale=1.2, eta=.05,
    max_depth=4, min_child_weight=8, subsample=.9, colsample_bytree=.8,
    **{'lambda': 3.0}, seed=42, tree_method='hist', verbosity=0, nthread=1)
BOUND_COLUMNS = ('lower_bound_hours', 'upper_bound_hours', 'exact_hours',
    'informative', 'label_available_at', 'reason')
HISTORICAL_AUDIT_ALLOWLIST = {
    str((ROOT / 'src/haze_forecast/app.py').resolve()): dict(
        historical_sha256='92bc58a1a818ce6ecaa58d6d8c0d4d01e6fac603eb21b346f2e8b4ad0396e1f7',
        current_sha256='7f222647d2a9bc31ee3040b121f987da5b95621fe2913a9c697e175b1568724c',
        reason='Noncomputational webpage audit entry in the old interval report; '
            'the exact historical/current pair was approved in PROTOCOL before fitting')}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(values):
    values = np.asarray(values, dtype='<f8').copy(order='C')
    values[np.isnan(values)] = np.nan
    header = json.dumps(dict(shape=list(values.shape), dtype='<f8'), sort_keys=True).encode('ascii')
    return hashlib.sha256(header + b'\n' + values.tobytes(order='C')).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def assert_hashes(manifest):
    for path, digest in manifest.items():
        if sha(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')


def merge_hash_manifests(*manifests):
    merged = {}
    for manifest in manifests:
        for path, digest in manifest.items():
            if path in merged and merged[path] != digest:
                raise ValueError(f'Conflicting frozen hash evidence: {path}')
            merged[path] = digest
    return merged


def resolve_historical_hashes(historical, current):
    if set(historical) != set(current):
        raise ValueError('Historical and current input hash identities differ')
    drift = []
    for path, digest in historical.items():
        if path in HISTORICAL_AUDIT_ALLOWLIST:
            approved = HISTORICAL_AUDIT_ALLOWLIST[path]
            if digest != approved['historical_sha256'] or current[path] != approved['current_sha256']:
                raise ValueError(f'Unapproved historical audit hash pair: {path}')
            drift.append(dict(path=path, **approved))
        elif current[path] != digest:
            raise ValueError(f'Frozen input changed: {path}')
    return dict(current), drift


def protected_directory_files(directories):
    files = []
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f'Protected directory missing or not a directory: {directory}')
        members = sorted(path for path in directory.rglob('*') if path.is_file())
        if not members:
            raise ValueError(f'Protected directory has no files: {directory}')
        files.extend(members)
    return files


def event(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def new_run_directory(parent=None, stamp=None):
    parent = Path(parent) if parent is not None else HERE / 'runs'
    stamp = stamp or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise ValueError('Timezone-aware UTC run timestamp required')
    run = parent / stamp.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    return run


def canonical_bounds(full, timestamps):
    return interval_recovery_bounds(full[['timestamp', 'aqi']], timestamps)


def rebuild_grid(measurements, feature_names):
    raw = measurements[['timestamp', *MEASUREMENT_COLUMNS]].copy()
    raw['timestamp'] = pd.to_datetime(raw.timestamp, utc=True).dt.tz_convert('Asia/Shanghai')
    full = build_business_states(build_observed_hourly_features(raw))
    if 'input_hour_observed' in measurements:
        observed = pd.Series(measurements.input_hour_observed.to_numpy(), index=raw.timestamp)
        full['input_hour_observed'] = full.timestamp.map(observed).fillna(False).astype(bool)
    pbl = [name for name in feature_names if name.startswith('boundary_layer_height_m')
        and not name.endswith('_is_missing')]
    pbl += [name.removesuffix('_is_missing') for name in feature_names
        if name.endswith('_is_missing') and name.removesuffix('_is_missing') in full]
    pbl = sorted(set(pbl))
    full = add_missingness_indicators(full, pbl)
    if 'boundary_layer_height_m_change_3_is_missing' in feature_names:
        full['boundary_layer_height_m_change_3_is_missing'] = (
            full.boundary_layer_height_m.isna() | full.boundary_layer_height_m_lag_3.isna()).astype(float)
    if not set(feature_names).issubset(full):
        raise ValueError('Canonical feature names absent from rebuilt grid')
    values = full[feature_names].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('Features must be finite or NaN')
    return full


def block_cutoffs(full):
    return full.groupby('continuity_block').timestamp.min()


def augment_bounds(frame, full):
    result = frame.drop(columns=[name for name in BOUND_COLUMNS if name in frame]).copy()
    bounds = canonical_bounds(full, result.timestamp)
    result = result.merge(bounds, on='timestamp', how='left', validate='one_to_one', sort=False)
    if len(result) != len(frame) or result.label_available_at.isna().any():
        raise ValueError('Canonical bounds lost input origins')
    return result


def prepare_legacy_population(old, full, feature_names):
    if old.timestamp.isna().any() or not old.timestamp.is_unique:
        raise ValueError('Legacy origins must be unique and complete')
    result = old[['timestamp', 'event_key', 'remaining_hours', *feature_names]].copy()
    result = result.rename(columns={'remaining_hours': 'legacy_remaining_hours'})
    grid = full.set_index('timestamp')
    if not result.timestamp.isin(grid.index).all():
        raise ValueError('Legacy origin outside canonical grid')
    result['continuity_block'] = result.timestamp.map(grid.continuity_block)
    result = augment_bounds(result, full)
    result['population_row'] = np.arange(len(result), dtype=np.int64)
    return result


def prepare_candidate_population(full):
    result = augment_bounds(full.loc[full.eligible].copy().reset_index(drop=True), full)
    result['population_row'] = np.arange(len(result), dtype=np.int64)
    return result


def prepare_evaluation(old, full, feature_names):
    if old.timestamp.isna().any() or not old.timestamp.is_unique:
        raise ValueError('Evaluation archive identities must be unique')
    grid = full.set_index('timestamp')
    if not old.timestamp.isin(grid.index).all():
        raise ValueError('Evaluation origin outside canonical grid')
    issue = grid.loc[old.timestamp].reset_index()
    result = old[['timestamp', 'event_key', 'remaining_hours', 'strict_prediction_hours']].copy()
    result = result.rename(columns={'remaining_hours': 'legacy_remaining_hours',
        'strict_prediction_hours': 'archive_prediction_hours'})
    result['eval_row'] = np.arange(len(result), dtype=np.int64)
    result['year'] = result.timestamp.dt.tz_convert('Asia/Shanghai').dt.year
    result['grid_index'] = pd.DatetimeIndex(full.timestamp).get_indexer(result.timestamp)
    for name in ('continuity_block', 'pollution_state', 'input_hour_observed'):
        result[name] = issue[name].to_numpy()
    result['causal_eligible'] = issue.eligible.to_numpy()
    result['cutoff'] = result.continuity_block.map(block_cutoffs(full))
    result = augment_bounds(result, full)
    candidate = issue[feature_names].astype(float).reset_index(drop=True)
    legacy = old[feature_names].astype(float).reset_index(drop=True)
    if any(np.isinf(frame.to_numpy()).any() for frame in (candidate, legacy)):
        raise ValueError('Evaluation feature infinity')
    return result.reset_index(drop=True), candidate, legacy


def mature_mask(population, cutoff, evaluation_block):
    if population[['timestamp', 'label_available_at', 'continuity_block']].isna().any().any():
        raise ValueError('Training identities or label maturity missing')
    return (population.timestamp.lt(cutoff) & population.label_available_at.lt(cutoff)
        & population.continuity_block.ne(evaluation_block)).to_numpy()


def label_counts(frame):
    lower = frame.lower_bound_hours.to_numpy(float)
    upper = frame.upper_bound_hours.to_numpy(float)
    return dict(exact=int((lower == upper).sum()),
        interval=int(((lower != upper) & np.isfinite(upper)).sum()),
        right_censored=int(np.isposinf(upper).sum()))


def score_arm(frame, prediction):
    result = score_predictions(frame, prediction)
    prediction = np.asarray(prediction, dtype=float)
    exact = frame.exact_hours.notna().to_numpy()
    truth = frame.exact_hours.to_numpy(float)
    signed = prediction[exact] - truth[exact]
    error = np.abs(signed)
    result.update(exact_error_p90=float(np.quantile(error, .9)) if len(error) else None,
        exact_signed_bias=float(signed.mean()) if len(signed) else None,
        exact_early_fraction=float((signed < 0).mean()) if len(signed) else None,
        exact_late_fraction=float((signed > 0).mean()) if len(signed) else None,
        tail24_blocks=int(frame.loc[exact & (truth >= 24), 'continuity_block'].nunique()))
    lower = frame.lower_bound_hours.to_numpy(float)
    upper = frame.upper_bound_hours.to_numpy(float)
    distance = np.maximum.reduce([lower-prediction, prediction-upper, np.zeros(len(frame))])
    result['subgroups'] = {}
    for name, mask in dict(finite_interval=(~exact & np.isfinite(upper)), right_censored=np.isposinf(upper)).items():
        blocks = frame.loc[mask, 'continuity_block'].to_numpy()
        values = distance[mask]
        result['subgroups'][name] = dict(rows=int(mask.sum()), blocks=len(set(blocks)),
            row_distance=float(values.mean()) if len(values) else None,
            block_distance=float(pd.Series(values).groupby(blocks).mean().mean()) if len(values) else None,
            below_fraction=float((prediction[mask] < lower[mask]).mean()) if mask.any() else None,
            above_fraction=float((prediction[mask] > upper[mask]).mean()) if mask.any() else None)
    return result


def paired_bounds(frame, baseline, candidate):
    return paired_error_bounds(frame, baseline, candidate)


def score_year(frame, predictions):
    return dict(arms={arm: score_arm(frame, predictions[arm]) for arm in ARMS},
        paired={f'interval_candidate_vs_{baseline}': paired_error_summary(
            frame, predictions[baseline], predictions['interval_candidate'])
            for baseline in ('archive', 'legacy_control')})


def year_gates(metrics):
    archive = metrics['arms']['archive']
    candidate = metrics['arms']['interval_candidate']
    paired = metrics['paired']['interval_candidate_vs_archive']
    compare = lambda a, b: a is not None and b is not None and a <= b
    checks = dict(
        coverage_100=all(metrics['arms'][arm]['prediction_coverage'] == 1. for arm in ARMS),
        nonempty_exact_block_and_tail=candidate['exact_rows'] > 0 and candidate['exact_blocks'] > 0
            and candidate['tail24_rows'] > 0 and candidate['tail24_blocks'] > 0,
        exact_row_mae_20pct=candidate['exact_row_mae'] is not None and archive['exact_row_mae'] is not None
            and archive['exact_row_mae'] > 0 and candidate['exact_row_mae'] <= .8*archive['exact_row_mae'],
        exact_block_not_worse=compare(candidate['exact_block_mae'], archive['exact_block_mae']),
        tail24_block_not_worse=compare(candidate['tail24_block_mae'], archive['tail24_block_mae']),
        paired_all_row_upper_not_positive=compare(paired['all_row_delta_upper'], 0.),
        paired_all_block_upper_not_positive=compare(paired['all_block_delta_upper'], 0.))
    return dict(checks=checks, passed=all(checks.values()),
        failure_reasons=[name for name, passed in checks.items() if not passed],
        exact_row_relative_reduction=(1-candidate['exact_row_mae']/archive['exact_row_mae'])
            if candidate['exact_row_mae'] is not None and archive['exact_row_mae'] is not None
            and archive['exact_row_mae'] > 0 else None)


def legacy_source_frames(feature_names):
    parts = []
    for name, path in LEGACY_SOURCES.items():
        frame = pd.read_parquet(path)
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert('Asia/Shanghai')
        frame['event_key'] = name + ':' + frame.event_id.astype(str)
        pbl = [column for column in feature_names if column.startswith('boundary_layer_height_m')
            and not column.endswith('_is_missing')]
        frame = add_missingness_indicators(frame, pbl)
        frame['boundary_layer_height_m_change_3_is_missing'] = (
            frame.boundary_layer_height_m.isna() | frame.boundary_layer_height_m_lag_3.isna()).astype(float)
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if len(result) != 8638 or not result.timestamp.is_unique:
        raise ValueError('Fixed legacy training population changed')
    return result


def check_equal_columns(actual, expected, columns):
    if len(actual) != len(expected):
        raise ValueError('Frozen reconstructed population size differs')
    for name in columns:
        np.testing.assert_array_equal(actual[name].to_numpy(), expected[name].to_numpy(), err_msg=name)


def prepare_inputs():
    assert_hashes(PINNED_SHA256)
    metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    names = list(metadata['feature_names'])
    if len(names) != 87 or len(set(names)) != 87:
        raise ValueError('Frozen 87-feature order changed')
    stored_full = pd.read_parquet(GRID_PATH)
    stored_origins = pd.read_parquet(ORIGINS_PATH)
    if len(stored_full) != 102024 or len(stored_origins) != 7890:
        raise ValueError('Canonical dataset dimensions changed')
    full = rebuild_grid(stored_full, names)
    check_equal_columns(full, stored_full, ['timestamp', *names, 'continuity_block', 'eligible', 'pollution_state', 'input_hour_observed'])
    candidate_population = prepare_candidate_population(full)
    check_equal_columns(candidate_population, stored_origins,
        ['timestamp', *names, 'continuity_block', 'eligible', *BOUND_COLUMNS])
    old = legacy_source_frames(names)
    legacy_population = prepare_legacy_population(old, full, names)
    archive = pd.read_parquet(ARCHIVE_PATH)
    archive['timestamp'] = pd.to_datetime(archive.timestamp, utc=True).dt.tz_convert('Asia/Shanghai')
    selection = archive.timestamp.dt.year.isin(YEARS)
    selected = archive.loc[selection].sort_values('timestamp').reset_index(drop=True)
    old_selected = old.loc[old.timestamp.isin(selected.timestamp)].sort_values('timestamp').reset_index(drop=True)
    check_equal_columns(selected, old_selected, ['timestamp', 'event_key', 'remaining_hours', *names])
    for year, expected_rows, expected_events in ((2024, 383, 15), (2025, 318, 17)):
        mask = selected.timestamp.dt.year == year
        if int(mask.sum()) != expected_rows or selected.loc[mask, 'event_key'].nunique() != expected_events:
            raise ValueError(f'Fixed evaluation cohort changed: {year}')
    evaluation, candidate_features, legacy_features = prepare_evaluation(selected, full, names)
    if not np.isfinite(evaluation.archive_prediction_hours).all() or (evaluation.archive_prediction_hours < 0).any():
        raise ValueError('Frozen archive prediction missing or invalid')
    return dict(full=full, feature_names=names, candidate_population=candidate_population,
        legacy_population=legacy_population, evaluation=evaluation,
        candidate_features=candidate_features, legacy_features=legacy_features)


def provenance_manifests():
    input_files = [GRID_PATH, ORIGINS_PATH, ARCHIVE_PATH, HOSTED_PATH, METADATA_PATH, *LEGACY_SOURCES.values()]
    dataset_report_path = GRID_PATH.parent / 'report.json'
    interval_report_path = ORIGINS_PATH.parent / 'report.json'
    input_files += [dataset_report_path, interval_report_path]
    dataset_report = json.loads(dataset_report_path.read_text(encoding='utf-8'))
    interval_report = json.loads(interval_report_path.read_text(encoding='utf-8'))
    historical = merge_hash_manifests(dataset_report['input_sha256'],
        dataset_report['deployment_sha256'], interval_report['input_sha256'])
    inherited, audit_drift = resolve_historical_hashes(historical,
        {path: sha(path) for path in historical})
    assert_hashes(inherited)
    inputs = {str(path.resolve()): sha(path) for path in input_files}
    inputs = merge_hash_manifests(inputs, inherited)
    protected_files = protected_directory_files((ROOT / 'artifacts/cnemc_2025_deployment_candidate',
        ROOT / 'artifacts/strict_history_backtest', ROOT / 'project-showcase', ROOT / 'sites-haze/dist',
        ROOT / 'releases/runtime_20261002'))
    protected = {str(path.resolve()): sha(path) for path in protected_files}
    protected.update({path: digest for path, digest in inputs.items() if Path(path).is_relative_to(ROOT / 'data')})
    code_files = [HERE / name for name in ('PROTOCOL.md', 'bridge_runner.py',
        'test_bridge_runner.py', 'verify_bridge.py', 'test_verify_bridge.py',
        'sequence_clock_adapter.py', 'test_sequence_clock_adapter.py')]
    code_files.extend([ROOT / 'pyproject.toml'])
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().is_relative_to(ROOT / 'src') and Path(path).suffix == '.py':
            code_files.append(Path(path).resolve())
    code = {str(path.resolve()): sha(path) for path in set(code_files)}
    packages = ('numpy', 'pandas', 'pyarrow', 'xgboost', 'scipy', 'scikit-learn', 'lightgbm', 'catboost', 'optuna', 'joblib')
    environment = dict(python_version=sys.version, python_executable=str(Path(sys.executable).resolve()),
        packages={name: importlib.metadata.version(name) for name in packages}, record_sha256={},
        python_executable_sha256=sha(sys.executable))
    for name in packages:
        distribution = importlib.metadata.distribution(name)
        for item in distribution.files:
            if item.name == 'RECORD':
                path = Path(distribution.locate_file(item)).resolve()
                environment['record_sha256'][str(path)] = sha(path)
    environment_hashes = dict(environment['record_sha256'])
    environment_hashes[environment['python_executable']] = environment['python_executable_sha256']
    frozen = merge_hash_manifests(inputs, protected, code, environment_hashes)
    assert_hashes(frozen)
    return dict(input_sha256=inputs, protected_sha256=protected, code_sha256=code,
        environment_sha256=environment_hashes, frozen_sha256=frozen,
        historical_input_sha256=historical, historical_audit_drift=audit_drift), environment


def cohort_counts(inputs):
    evaluation = inputs['evaluation']
    return dict(grid_rows=len(inputs['full']), candidate_population=len(inputs['candidate_population']),
        legacy_population=len(inputs['legacy_population']), evaluation_rows=len(evaluation),
        evaluation_blocks=int(evaluation.continuity_block.nunique()),
        planned_fits=int(evaluation.continuity_block.nunique()*2),
        annual={str(year): dict(rows=int(evaluation.year.eq(year).sum()),
            old_events=int(evaluation.loc[evaluation.year.eq(year), 'event_key'].nunique()),
            blocks=int(evaluation.loc[evaluation.year.eq(year), 'continuity_block'].nunique()),
            causal_eligible=int(evaluation.loc[evaluation.year.eq(year), 'causal_eligible'].sum()),
            labels=label_counts(evaluation.loc[evaluation.year.eq(year)])) for year in YEARS})


def members_for_model(population, mask, model_id, arm, evaluation_block, cutoff):
    columns = ['population_row', 'timestamp', 'continuity_block', *BOUND_COLUMNS]
    members = population.loc[mask, columns].copy()
    members['legacy_remaining_hours'] = population.loc[mask, 'legacy_remaining_hours'].to_numpy() \
        if arm == 'legacy_control' else np.nan
    members['model_id'] = model_id
    members['arm'] = arm
    members['evaluation_block'] = evaluation_block
    members['cutoff'] = cutoff
    return members


def fit_block(population, feature_names, eval_features, block, cutoff, arm, run):
    mask = mature_mask(population, cutoff, block)
    if not mask.any():
        raise ValueError(f'No mature training members: {arm} block {block}')
    train = population.loc[mask]
    features = train[feature_names].astype(float)
    model = ObservedAFTSurvivalRegressor(**MODEL_CONFIG)
    if arm == 'legacy_control':
        model.fit(features, train.legacy_remaining_hours)
    elif arm == 'interval_candidate':
        model.fit_bounds(features, train.lower_bound_hours, train.upper_bound_hours)
    else:
        raise ValueError('Only the two predeclared fitting arms are supported')
    # A native model reload may reset generic parameters, so preserve fit-time evidence first.
    actual_config = json.loads(model.booster.save_config())
    rounds = model.booster.num_boosted_rounds()
    booster_features = model.booster.feature_names
    if rounds != MODEL_CONFIG['num_boost_round']:
        raise ValueError('Actual boosted round count differs from fixed configuration')
    if booster_features != feature_names:
        raise ValueError('Actual booster feature order differs from fixed feature order')
    prediction = model.predict(eval_features)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid prediction; evaluation origins must not be dropped')
    model_id = f'{arm}_b{block}'
    path = run / 'models' / f'{model_id}.json'
    model.booster.save_model(path)
    restored = xgb.Booster(params={'nthread': 1})
    restored.load_model(path)
    again = restored.predict(xgb.DMatrix(eval_features.astype(float), missing=np.nan)).clip(min=0.)
    np.testing.assert_array_equal(prediction, again)
    members = members_for_model(population, mask, model_id, arm, block, cutoff)
    record = dict(model_id=model_id, arm=arm, continuity_block=int(block), cutoff=cutoff.isoformat(),
        model_path=path.relative_to(run).as_posix(), training_member_path='training_members.parquet',
        train_rows=len(train), eval_rows=len(eval_features), label_counts=label_counts(train),
        training_last_timestamp=train.timestamp.max().isoformat(),
        max_label_available_at=train.label_available_at.max().isoformat(),
        training_features_sha256=array_sha(features.to_numpy()),
        training_timestamps_sha256=hashlib.sha256(pd.DatetimeIndex(train.timestamp).asi8.astype('<i8').tobytes()).hexdigest(),
        feature_names=feature_names, model_sha256=sha(path), configuration=MODEL_CONFIG,
        aft_parameters=AFT_PARAMETERS,
        actual_trained_booster_config=actual_config, num_boosted_rounds=int(rounds),
        booster_feature_names=booster_features,
        right_censor_policy='global archive-end labels only; all mature rows retained')
    return prediction, members, record


def row_scores(evaluation, predictions):
    result = evaluation.copy()
    lower = result.lower_bound_hours.to_numpy(float)
    upper = result.upper_bound_hours.to_numpy(float)
    truth = result.exact_hours.to_numpy(float)
    for arm in ARMS:
        values = predictions[arm]
        score_predictions(result, values)
        result[arm] = values
        result[f'{arm}_error_lower'] = np.maximum.reduce([lower-values, values-upper, np.zeros(len(values))])
        result[f'{arm}_error_upper'] = np.maximum(abs(lower-values), abs(upper-values))
        result[f'{arm}_exact_error'] = abs(values-truth)
        result[f'{arm}_signed_error'] = values-truth
    for baseline in ('archive', 'legacy_control'):
        bounds = paired_bounds(result, predictions[baseline], predictions['interval_candidate'])
        result[f'candidate_vs_{baseline}_delta_lower'] = bounds.lower.to_numpy()
        result[f'candidate_vs_{baseline}_delta_upper'] = bounds.upper.to_numpy()
    return result


def process_scores(evaluation, predictions):
    records = []
    for (year, block), rows in evaluation.groupby(['year', 'continuity_block'], sort=True):
        for arm in ARMS:
            score = score_arm(rows, predictions[arm][rows.index.to_numpy()])
            record = dict(year=int(year), continuity_block=int(block), arm=arm,
                old_events=int(rows.event_key.nunique()))
            record.update({name: value for name, value in score.items() if name != 'subgroups'})
            for name, values in score['subgroups'].items():
                record.update({f'{name}_{field}': value for field, value in values.items()})
            records.append(record)
    return pd.DataFrame(records)


def execute(inputs, manifests, environment, started):
    assert_hashes(manifests['frozen_sha256'])
    run = new_run_directory()
    (run / 'models').mkdir()
    report = dict(schema_version=1, status='running', run_dir=str(run), years=list(YEARS),
        **manifests, configuration=MODEL_CONFIG, aft_parameters=AFT_PARAMETERS,
        feature_names=inputs['feature_names'], counts=cohort_counts(inputs),
        target='complete hours until first future six-hour AQI<=100 window start',
        historical_shadow_evaluation=True, blind_test=False, deployed=False,
        archive_is_primary_reference=True, refitted_arms=['legacy_control', 'interval_candidate'],
        uncertainty_scope='observation-compatible label bounds, not confidence intervals')
    write_json(run / 'report.json', report)
    event('run_created', run_dir=str(run))
    try:
        names = inputs['feature_names']
        evaluation = inputs['evaluation']
        evaluation.to_parquet(run / 'evaluation.parquet', index=False)
        inputs['candidate_population'].to_parquet(run / 'candidate_population.parquet', index=False)
        inputs['legacy_population'].to_parquet(run / 'legacy_population.parquet', index=False)
        np.savez_compressed(run / 'features.npz', timestamps=pd.DatetimeIndex(evaluation.timestamp).asi8,
            feature_names=np.asarray(names, dtype=str), candidate=inputs['candidate_features'].to_numpy(float),
            legacy=inputs['legacy_features'].to_numpy(float))
        write_json(run / 'environment.json', environment)
        (run / 'execution_record_at_start.md').write_bytes((HERE / 'EXECUTION_RECORD.md').read_bytes())
        predictions = dict(archive=evaluation.archive_prediction_hours.to_numpy(float),
            legacy_control=np.full(len(evaluation), np.nan), interval_candidate=np.full(len(evaluation), np.nan))
        member_tables, records = [], []
        # Each continuity block, including a cross-year block, receives one fixed model per fitting arm.
        for block, rows in evaluation.groupby('continuity_block', sort=True):
            cutoff = rows.cutoff.iloc[0]
            if not rows.cutoff.eq(cutoff).all():
                raise ValueError('One continuity block must have one full-grid cutoff')
            positions = rows.index.to_numpy()
            for arm in ('legacy_control', 'interval_candidate'):
                population = inputs['legacy_population'] if arm == 'legacy_control' else inputs['candidate_population']
                features = inputs['legacy_features'] if arm == 'legacy_control' else inputs['candidate_features']
                values, members, record = fit_block(population, names, features.iloc[positions], block, cutoff, arm, run)
                predictions[arm][positions] = values
                member_tables.append(members)
                records.append(record)
                event('model_complete', block=int(block), arm=arm, train_rows=record['train_rows'], eval_rows=len(rows))
        pd.concat(member_tables, ignore_index=True).to_parquet(run / 'training_members.parquet', index=False)
        write_json(run / 'models.json', dict(schema_version=1, feature_names=names, models=records))
        row_scores(evaluation, predictions).to_parquet(run / 'predictions.parquet', index=False)
        process_scores(evaluation, predictions).to_parquet(run / 'process_scores.parquet', index=False)
        annual, gates = {}, {}
        for year in YEARS:
            mask = evaluation.year.eq(year).to_numpy()
            annual[str(year)] = score_year(evaluation.loc[mask], {arm: values[mask] for arm, values in predictions.items()})
            gates[str(year)] = year_gates(annual[str(year)])
        failures = [f'{year}:{name}' for year in YEARS for name in gates[str(year)]['failure_reasons']]
        score = dict(annual=annual, pooled=score_year(evaluation, predictions),
            gates=dict(annual=gates, passed=all(gates[str(year)]['passed'] for year in YEARS), failure_reasons=failures))
        write_json(run / 'scores.json', score)
        assert_hashes(manifests['frozen_sha256'])
        report.update(status='pending_independent_verification', metrics=score,
            fitted_models=len(records), protected_unchanged=True,
            training_maturity_summary=[{name: record[name] for name in ('model_id', 'train_rows', 'label_counts',
                'training_last_timestamp', 'max_label_available_at')} for record in records],
            feature_missingness={arm: dict(zip(names, np.isnan(inputs[f'{key}_features'].to_numpy(float)).mean(axis=0).tolist()))
                for arm, key in (('legacy_control', 'legacy'), ('interval_candidate', 'candidate'))},
            evaluation_eligibility_change=dict(causal_eligible=int(evaluation.causal_eligible.sum()),
                retained_not_causal_eligible=int((~evaluation.causal_eligible).sum())),
            elapsed_seconds=time.perf_counter()-started,
            limits=['Known historical 2024/2025 archives, not a new blind test',
                'Complete labels only when exact; missing future evidence remains interval/censored',
                'Legacy and candidate differ in features, label and origin policies; not a single-factor attribution',
                'Global archive-end censoring is not dynamic online censoring',
                'Historical weather issue-time availability and upstream imputation are not proven',
                'No transfer to manual, CAMS online estimates or capped multimodal research',
                'Independent verification and both yearly gates required before interface shadow integration'])
    except BaseException as error:
        report.update(status='execution_failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['output_sha256'] = {path.relative_to(run).as_posix(): sha(path)
            for path in run.rglob('*') if path.is_file() and path.name != 'report.json'}
        write_json(run / 'report.json', report)
    event('pending_independent_verification', run_dir=str(run), yearly_gates_passed=report['metrics']['gates']['passed'])
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--execute', action='store_true', help='Only after explicit code/test/specification approval')
    args = parser.parse_args()
    started = time.perf_counter()
    manifests, environment = provenance_manifests()
    inputs = prepare_inputs()
    assert_hashes(manifests['frozen_sha256'])
    if args.preflight_only:
        event('preflight_passed', **cohort_counts(inputs), frozen_files=len(manifests['frozen_sha256']), training_started=False)
        return
    execute(inputs, manifests, environment, started)


if __name__ == '__main__':
    main()
