"""Single fixed full-calendar population experiment, with no production edits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRIDGE_PATH = ROOT / 'experiments/web_recovery_bridge_20261002/bridge_runner.py'
BRIDGE_SHA256 = '3684ee50916a426f44ed9c6791a8a80fc1f017c82b587dc7d45cb728d081e6f3'
PARENT = ROOT / 'experiments/web_recovery_factors_20261002/runs/20261002T161429405383Z'
PARENT_REPORT_SHA256 = '21aa39d5a08a683e23904000eeeccbb3c31fda30201cf30e64b4cf28ab063998'
PARENT_VERIFICATION_SHA256 = '17f3a9e71d15e95c3f9f02149a5bd1c1f27abe784682979c050263e44e0f8566'
PARENT_VERIFIER_SHA256 = '1c3052b2f51023d8e08649ed47dff614f05065ba1dd161d9964b8151400c6c6e'
if hashlib.sha256(BRIDGE_PATH.read_bytes()).hexdigest() != BRIDGE_SHA256:
    raise ValueError('The immutable bridge helper digest changed')
spec = importlib.util.spec_from_file_location('immutable_calendar_bridge', BRIDGE_PATH)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
np, pd = bridge.np, bridge.pd
NEW_ARMS = ('calendar_all_hours',)
PRIMARY_CANDIDATE = NEW_ARMS[0]
REFERENCE_ARMS = ('archive', 'interval_candidate', 'grid_all_legacy_origins')
ARMS = REFERENCE_ARMS + NEW_ARMS
PAIRS = tuple((arm, PRIMARY_CANDIDATE) for arm in REFERENCE_ARMS)
META_COLUMNS = ('population_row', 'timestamp', 'year', 'continuity_block', 'pollution_state',
    'input_hour_observed', 'eligible', 'aqi_category', 'label_type')
PROTECTED_DIRECTORIES = tuple(ROOT / name for name in (
    'artifacts/cnemc_2025_deployment_candidate', 'artifacts/strict_history_backtest',
    'project-showcase', 'sites-haze/dist', 'releases/runtime_20261002'))
WEIGHT_POLICY = 'Unit weight per mature calendar hour; no resampling or extra weights'
SCOPE = 'Single full-calendar training population change; fixed 701 polluted evaluation origins'
LIMITS = [
    'Known historical 2024/2025 years are not a new blind test',
    'Historical weather issue-time availability and upstream imputation remain unproven',
    'Global archive-end right-censoring is not dynamically matured unfinished-event supervision',
    'Population expansion changes target frequencies, histogram binning and effective regularization',
    'A net population effect cannot be uniquely attributed to clean-hour supervision',
    'Passing retrospective gates permits isolated shadow-interface audits, not deployment',
    'No transfer to manual 17-feature, online CAMS or capped multimodal scores is established',
    'Independent verification reuses immutable prior independent mathematics; no new second implementation',
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new_json(path, value):
    serialized = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as target:
        target.write(serialized)


def file_inventory(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f'Missing protected directory: {directory}')
    return sorted(path.relative_to(directory).as_posix() for path in directory.rglob('*') if path.is_file())


def assert_directory_inventory(directories, locked):
    for directory in directories:
        directory = Path(directory).resolve()
        actual = file_inventory(directory)
        expected = sorted(Path(path).resolve().relative_to(directory).as_posix()
            for path in locked if Path(path).resolve().is_relative_to(directory))
        if not actual or actual != expected:
            raise ValueError(f'Protected directory is empty or inventory changed: {directory}')


def parent_artifact_manifest(parent, report):
    parent = Path(parent).resolve()
    actual = file_inventory(parent)
    expected = sorted(set(report['output_sha256']) | {'report.json', 'verification.json'})
    if actual != expected:
        raise ValueError('Parent artifact inventory changed')
    for name, digest in report['output_sha256'].items():
        path = (parent / name).resolve()
        if not path.is_relative_to(parent) or bridge.sha(path) != digest:
            raise ValueError(f'Invalid or changed parent output: {name}')
    return {str((parent / name).resolve()): bridge.sha(parent / name) for name in actual}, actual


def assert_locks(manifests):
    bridge.assert_hashes(manifests['frozen_sha256'])
    if file_inventory(manifests['parent_run_dir']) != manifests['parent_artifact_inventory']:
        raise ValueError('Parent artifact inventory changed')
    if 'protected_sha256' in manifests:
        assert_directory_inventory(PROTECTED_DIRECTORIES, manifests['protected_sha256'])


def validate_parent(report, verification, report_digest):
    if report_digest != PARENT_REPORT_SHA256 or report.get('status') != 'pending_independent_verification':
        raise ValueError('Parent report is not the exact fixed completed run')
    if not (verification.get('status') == 'passed' and verification.get('independent_verification_passed') is True
            and verification.get('report_sha256') == report_digest
            and verification.get('verifier_sha256') == PARENT_VERIFIER_SHA256):
        raise ValueError('Parent independent verification chain is not passed and exact')


def check_environment(environment):
    if (str(Path(sys.executable).resolve()) != environment['python_executable']
            or sys.version != environment['python_version']):
        raise ValueError('The exact parent research interpreter is required')
    if any(importlib.metadata.version(name) != version for name, version in environment['packages'].items()):
        raise ValueError('Parent research package versions changed')


def provenance():
    report_path, verification_path = PARENT / 'report.json', PARENT / 'verification.json'
    if bridge.sha(report_path) != PARENT_REPORT_SHA256 or bridge.sha(verification_path) != PARENT_VERIFICATION_SHA256:
        raise ValueError('Parent report or verification bytes changed')
    report, verification = read_json(report_path), read_json(verification_path)
    validate_parent(report, verification, bridge.sha(report_path))
    parent_hashes, inventory = parent_artifact_manifest(PARENT, report)
    if len(report['frozen_sha256']) != 528 or len(parent_hashes) != 77:
        raise ValueError('Fixed parent frozen scope or artifact count differs')
    new_code = {str(HERE / name): bridge.sha(HERE / name) for name in
        ('PROTOCOL.md', 'calendar_runner.py', 'test_calendar_runner.py', 'verify_calendar.py', 'test_verify_calendar.py')}
    code = bridge.merge_hash_manifests(report['code_sha256'], new_code)
    manifests = dict(input_sha256=report['input_sha256'], protected_sha256=report['protected_sha256'],
        code_sha256=code, environment_sha256=report['environment_sha256'],
        original_scope_sha256=report['frozen_sha256'], parent_sha256=parent_hashes,
        parent_run_dir=str(PARENT), parent_artifact_inventory=inventory,
        parent_report_sha256=PARENT_REPORT_SHA256, parent_verification_sha256=PARENT_VERIFICATION_SHA256,
        parent_verification_report_sha256=verification['report_sha256'], parent_verifier_sha256=PARENT_VERIFIER_SHA256,
        historical_input_sha256=report['historical_input_sha256'], historical_audit_drift=report['historical_audit_drift'],
        reused_main_helper_sha256=BRIDGE_SHA256)
    manifests['frozen_sha256'] = bridge.merge_hash_manifests(*(manifests[name] for name in
        ('input_sha256', 'protected_sha256', 'code_sha256', 'environment_sha256',
         'original_scope_sha256', 'parent_sha256')))
    assert_locks(manifests)
    environment = read_json(PARENT / 'environment.json')
    check_environment(environment)
    return manifests, environment, report


def calendar_population(full, names):
    required = {'timestamp', 'continuity_block', 'pollution_state', 'input_hour_observed', 'eligible', 'aqi'} | set(names)
    if not required.issubset(full.columns) or full.empty:
        raise ValueError('Calendar grid lacks required feature or identity columns')
    if (not isinstance(full.timestamp.dtype, pd.DatetimeTZDtype) or full.timestamp.isna().any()
            or not full.timestamp.is_unique or not full.timestamp.is_monotonic_increasing
            or full.timestamp.diff().iloc[1:].ne(pd.Timedelta(hours=1)).any()
            or full[['continuity_block', 'pollution_state', 'input_hour_observed', 'eligible']].isna().any().any()
            or np.isinf(full[names].to_numpy(float)).any()):
        raise ValueError('Calendar identity must be a complete hourly grid; features may be NaN but not infinite')
    population = bridge.augment_bounds(full.copy().reset_index(drop=True), full)
    population['population_row'] = np.arange(len(population), dtype=np.int64)
    population['year'] = population.timestamp.dt.tz_convert('Asia/Shanghai').dt.year
    population['aqi_category'] = np.where(population.aqi.isna(), 'missing',
        np.where(population.aqi.le(100), 'clean', 'polluted'))
    lower, upper = population.lower_bound_hours.to_numpy(float), population.upper_bound_hours.to_numpy(float)
    if not np.isfinite(lower).all() or (lower < 1).any() or np.isnan(upper).any() or (upper < lower).any():
        raise ValueError('Canonical complete-time bounds must retain T>=1')
    population['label_type'] = np.where(lower == upper, 'exact',
        np.where(np.isfinite(upper), 'interval', 'right_censored'))
    columns = list(META_COLUMNS) + names + list(bridge.BOUND_COLUMNS)
    return population.loc[:, list(dict.fromkeys(columns))]


def assert_common_members(population, parent_grid, names):
    if not parent_grid.timestamp.is_unique or not parent_grid.timestamp.isin(population.timestamp).all():
        raise ValueError('A shared parent training identity is missing or ambiguous')
    common = population.set_index('timestamp').loc[parent_grid.timestamp].reset_index()
    bridge.check_equal_columns(common, parent_grid,
        ['timestamp', 'continuity_block', *names, *bridge.BOUND_COLUMNS])


def training_members(population, cutoff, block):
    return population.loc[bridge.mature_mask(population, cutoff, block)].copy()


def member_summary(train, names):
    if train.empty:
        raise ValueError('No mature training rows; empty support cannot pass')
    exact_one = int(train.exact_hours.eq(1.).sum())
    return dict(label_counts=bridge.label_counts(train),
        aqi_category_counts={name: int(train.aqi_category.eq(name).sum()) for name in ('clean', 'polluted', 'missing')},
        input_hour_observed_counts={'observed': int(train.input_hour_observed.sum()),
            'unobserved': int((~train.input_hour_observed).sum())},
        causal_eligible_counts={'eligible': int(train.eligible.sum()), 'ineligible': int((~train.eligible).sum())},
        exact_t1_rows=exact_one, exact_t1_fraction=exact_one / len(train),
        train_feature_missingness=dict(zip(names, train[names].isna().mean().tolist())))


def inherit_references(evaluation, saved):
    bridge.check_equal_columns(saved, evaluation, list(evaluation.columns))
    predictions = {arm: saved[arm].to_numpy(float).copy() for arm in REFERENCE_ARMS}
    for arm, values in predictions.items():
        if values.shape != (len(evaluation),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Fixed reference is incomplete or invalid: {arm}')
        np.testing.assert_array_equal(values, saved[arm].to_numpy(float))
    return predictions


def assert_inherited_predictions(predictions, inherited):
    for arm in REFERENCE_ARMS:
        np.testing.assert_array_equal(predictions[arm], inherited[arm], err_msg=arm)


def assert_parent_features(path, timestamps, names, features):
    with np.load(path, allow_pickle=False) as saved:
        required = {'timestamps', 'feature_names', 'grid_all_legacy_origins'}
        if not required.issubset(saved.files):
            raise ValueError('Parent factor features require the grid_all_legacy_origins matrix')
        values = saved['grid_all_legacy_origins']
        if not np.issubdtype(values.dtype, np.floating) or np.isinf(values).any():
            raise ValueError('Parent grid features must be floating, finite or NaN')
        np.testing.assert_array_equal(saved['feature_names'], names)
        np.testing.assert_array_equal(saved['timestamps'], pd.DatetimeIndex(timestamps).asi8)
        np.testing.assert_array_equal(values, features.to_numpy(float))


def prepare_inputs(parent_report):
    names = list(parent_report['feature_names'])
    if len(names) != 87 or len(set(names)) != 87:
        raise ValueError('The parent 87-feature order differs')
    stored = pd.read_parquet(bridge.GRID_PATH)
    full = bridge.rebuild_grid(stored, names)
    bridge.check_equal_columns(full, stored,
        ['timestamp', *names, 'continuity_block', 'eligible', 'pollution_state', 'input_hour_observed'])
    population = calendar_population(full, names)
    parent_grid = pd.read_parquet(PARENT / 'grid_population.parquet')
    if len(population) != 102024 or len(parent_grid) != 8638:
        raise ValueError('Fixed full calendar or shared old population differs')
    assert_common_members(population, parent_grid, names)
    evaluation = pd.read_parquet(PARENT / 'evaluation.parquet')
    if len(evaluation) != 701 or not np.array_equal(evaluation.eval_row, np.arange(701)):
        raise ValueError('The fixed 701 evaluation identity differs')
    issue = population.set_index('timestamp').loc[evaluation.timestamp].reset_index()
    bridge.check_equal_columns(evaluation, issue,
        ['timestamp', 'year', 'continuity_block', 'pollution_state', 'input_hour_observed', *bridge.BOUND_COLUMNS])
    np.testing.assert_array_equal(evaluation.causal_eligible, issue.eligible)
    np.testing.assert_array_equal(evaluation.cutoff, evaluation.continuity_block.map(bridge.block_cutoffs(full)))
    if evaluation.continuity_block.nunique() != 32:
        raise ValueError('The fixed 32 evaluation cutoffs differ')
    features = issue[names].astype(float)
    assert_parent_features(PARENT / 'features.npz', evaluation.timestamp, names, features)
    references = inherit_references(evaluation, pd.read_parquet(PARENT / 'predictions.parquet'))
    old_members = pd.read_parquet(PARENT / 'training_members.parquet')
    old_members = old_members.loc[old_members.arm.eq('grid_all_legacy_origins')].copy()
    return dict(population=population, feature_names=names, evaluation=evaluation,
        eval_features=features, inherited_predictions=references, parent_members=old_members)


def validate_fitted_model(rounds, actual_names, names):
    if rounds != 500 or actual_names != names:
        raise ValueError('Actual fitted rounds or feature order differs from the fixed contract')


def fit_block(inputs, rows, block, cutoff, run):
    names = inputs['feature_names']
    model_id = f'{PRIMARY_CANDIDATE}_b{block}'
    path = run / 'models' / f'{model_id}.json'
    if path.exists():
        raise FileExistsError(path)
    train = training_members(inputs['population'], cutoff, block)
    summary = member_summary(train, names)
    old = inputs['parent_members'].loc[inputs['parent_members'].evaluation_block.eq(block)]
    common = train.loc[train.timestamp.isin(old.timestamp)]
    bridge.check_equal_columns(common, old,
        ['timestamp', 'continuity_block', *bridge.BOUND_COLUMNS])
    features, eval_features = train[names].astype(float), inputs['eval_features'].iloc[rows.index.to_numpy()]
    model = bridge.ObservedAFTSurvivalRegressor(**bridge.MODEL_CONFIG)
    model.fit_bounds(features, train.lower_bound_hours, train.upper_bound_hours)
    actual_config = json.loads(model.booster.save_config())
    rounds, booster_names = model.booster.num_boosted_rounds(), model.booster.feature_names
    validate_fitted_model(rounds, booster_names, names)
    prediction = model.predict(eval_features)
    if prediction.shape != (len(rows),) or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('All evaluation origins require a finite nonnegative prediction')
    model.booster.save_model(path)
    restored = bridge.xgb.Booster(params={'nthread': 1})
    restored.load_model(path)
    again = restored.predict(bridge.xgb.DMatrix(eval_features.astype(float), missing=np.nan)).clip(min=0.)
    np.testing.assert_array_equal(prediction, again)
    members = train.loc[:, list(META_COLUMNS) + list(bridge.BOUND_COLUMNS)].copy()
    members['model_id'], members['arm'], members['evaluation_block'], members['cutoff'] = model_id, PRIMARY_CANDIDATE, block, cutoff
    record = dict(model_id=model_id, arm=PRIMARY_CANDIDATE, continuity_block=int(block), cutoff=cutoff.isoformat(),
        model_path=path.relative_to(run).as_posix(), training_member_path='training_members.parquet',
        train_rows=len(train), eval_rows=len(rows), **summary,
        shared_legacy_member_rows=len(common), additional_calendar_member_rows=len(train) - len(common),
        training_last_timestamp=train.timestamp.max().isoformat(), max_label_available_at=train.label_available_at.max().isoformat(),
        training_features_sha256=bridge.array_sha(features.to_numpy()),
        training_timestamps_sha256=hashlib.sha256(pd.DatetimeIndex(train.timestamp).asi8.astype('<i8').tobytes()).hexdigest(),
        feature_names=names, model_sha256=bridge.sha(path), configuration=bridge.MODEL_CONFIG,
        aft_parameters=bridge.AFT_PARAMETERS, actual_trained_booster_config=actual_config,
        num_boosted_rounds=int(rounds), booster_feature_names=booster_names, reload_predictions_equal=True,
        weight_policy=WEIGHT_POLICY, right_censor_policy='global archive-end labels only; all mature rows retained')
    return prediction, members, record


def score_cohort(evaluation, predictions):
    return dict(arms={arm: bridge.score_arm(evaluation, predictions[arm]) for arm in ARMS},
        paired={f'{candidate}_vs_{baseline}': bridge.paired_error_summary(
            evaluation, predictions[baseline], predictions[candidate]) for baseline, candidate in PAIRS})


def main_gates(metrics):
    gate = bridge.year_gates(dict(arms=dict(archive=metrics['arms']['archive'],
        legacy_control=metrics['arms']['grid_all_legacy_origins'], interval_candidate=metrics['arms'][PRIMARY_CANDIDATE]),
        paired=dict(interval_candidate_vs_archive=metrics['paired'][f'{PRIMARY_CANDIDATE}_vs_archive'])))
    gate['checks']['coverage_100'] = all(metrics['arms'][arm]['prediction_coverage'] == 1. for arm in ARMS)
    gate['passed'] = all(gate['checks'].values())
    gate['failure_reasons'] = [name for name, passed in gate['checks'].items() if not passed]
    return gate


def compute_scores(evaluation, predictions):
    annual, gates = {}, {}
    for year in bridge.YEARS:
        mask = evaluation.year.eq(year).to_numpy()
        annual[str(year)] = score_cohort(evaluation.loc[mask], {arm: values[mask] for arm, values in predictions.items()})
        gates[str(year)] = main_gates(annual[str(year)])
    return dict(annual=annual, pooled=score_cohort(evaluation, predictions), gates=dict(annual=gates,
        passed=all(gates[str(year)]['passed'] for year in bridge.YEARS),
        failure_reasons=[f'{year}:{name}' for year in bridge.YEARS for name in gates[str(year)]['failure_reasons']]))


def row_scores(evaluation, predictions):
    result = evaluation.copy()
    lower, upper, truth = (result[name].to_numpy(float) for name in ('lower_bound_hours', 'upper_bound_hours', 'exact_hours'))
    for arm in ARMS:
        values = np.asarray(predictions[arm], float)
        bridge.score_arm(result, values)
        result[arm] = values
        result[f'{arm}_error_lower'] = np.maximum.reduce([lower-values, values-upper, np.zeros(len(values))])
        result[f'{arm}_error_upper'] = np.maximum(abs(lower-values), abs(upper-values))
        result[f'{arm}_exact_error'], result[f'{arm}_signed_error'] = abs(values-truth), values-truth
    for baseline, candidate in PAIRS:
        bounds = bridge.paired_bounds(result, predictions[baseline], predictions[candidate])
        name = f'{candidate}_vs_{baseline}'
        result[f'{name}_delta_lower'], result[f'{name}_delta_upper'] = bounds.lower.to_numpy(), bounds.upper.to_numpy()
    return result


def process_scores(evaluation, predictions):
    records = []
    for (year, block), rows in evaluation.groupby(['year', 'continuity_block'], sort=True):
        for arm in ARMS:
            score = bridge.score_arm(rows, predictions[arm][rows.index.to_numpy()])
            record = dict(year=int(year), continuity_block=int(block), arm=arm, old_events=int(rows.event_key.nunique()))
            record.update({name: value for name, value in score.items() if name != 'subgroups'})
            for name, values in score['subgroups'].items():
                record.update({f'{name}_{field}': value for field, value in values.items()})
            records.append(record)
    return pd.DataFrame(records)


def cohort_counts(inputs, parent_report):
    population, evaluation, names = inputs['population'], inputs['evaluation'], inputs['feature_names']
    summary = member_summary(population, names)
    return dict(grid_rows=len(population), calendar_population=len(population),
        evaluation_rows=len(evaluation), evaluation_blocks=int(evaluation.continuity_block.nunique()), planned_fits=32,
        annual=parent_report['counts']['annual'], calendar_population_label_counts=summary['label_counts'],
        calendar_population_aqi_category_counts=summary['aqi_category_counts'],
        calendar_population_input_hour_observed_counts=summary['input_hour_observed_counts'],
        calendar_population_eligible_counts=summary['causal_eligible_counts'],
        calendar_population_exact_t1_rows=summary['exact_t1_rows'],
        calendar_population_exact_t1_fraction=summary['exact_t1_fraction'])


def execute(inputs, manifests, environment, parent_report, started):
    assert_locks(manifests)
    check_environment(environment)
    run = bridge.new_run_directory(HERE / 'runs')
    (run / 'models').mkdir()
    report = dict(schema_version=1, status='execution_failed', run_dir=str(run), years=list(bridge.YEARS), **manifests,
        configuration=bridge.MODEL_CONFIG, aft_parameters=bridge.AFT_PARAMETERS, feature_names=inputs['feature_names'],
        counts=cohort_counts(inputs, parent_report), primary_candidate=PRIMARY_CANDIDATE,
        new_refitted_arms=list(NEW_ARMS), inherited_arms=list(REFERENCE_ARMS), arms=list(ARMS),
        historical_shadow_evaluation=True, blind_test=False, deployed=False, production_ready=False,
        scope=SCOPE, target=parent_report['target'], uncertainty_scope=parent_report['uncertainty_scope'],
        archive_is_primary_reference=True, weight_policy=WEIGHT_POLICY, limits=LIMITS)
    try:
        names, evaluation = inputs['feature_names'], inputs['evaluation']
        evaluation.to_parquet(run / 'evaluation.parquet', index=False)
        inputs['population'].to_parquet(run / 'calendar_population.parquet', index=False)
        np.savez_compressed(run / 'features.npz', timestamps=pd.DatetimeIndex(evaluation.timestamp).asi8,
            feature_names=np.asarray(names, dtype=str), candidate=inputs['eval_features'].to_numpy(float))
        write_new_json(run / 'environment.json', environment)
        (run / 'execution_record_at_start.md').write_bytes((HERE / 'EXECUTION_RECORD.md').read_bytes())
        predictions = {arm: values.copy() for arm, values in inputs['inherited_predictions'].items()}
        predictions[PRIMARY_CANDIDATE] = np.full(len(evaluation), np.nan)
        member_tables, records = [], []
        for block, rows in evaluation.groupby('continuity_block', sort=True):
            cutoff = rows.cutoff.iloc[0]
            if not rows.cutoff.eq(cutoff).all():
                raise ValueError('Each fixed block must retain its unique full-grid cutoff')
            values, members, record = fit_block(inputs, rows, int(block), cutoff, run)
            predictions[PRIMARY_CANDIDATE][rows.index.to_numpy()] = values
            member_tables.append(members)
            records.append(record)
            bridge.event('model_complete', arm=PRIMARY_CANDIDATE, block=int(block), train_rows=len(members), eval_rows=len(rows))
        if len(records) != 32:
            raise ValueError('The fixed execution must fit exactly 32 models')
        assert_inherited_predictions(predictions, inputs['inherited_predictions'])
        pd.concat(member_tables, ignore_index=True).to_parquet(run / 'training_members.parquet', index=False)
        write_new_json(run / 'models.json', dict(schema_version=1, feature_names=names, models=records))
        row_scores(evaluation, predictions).to_parquet(run / 'predictions.parquet', index=False)
        process_scores(evaluation, predictions).to_parquet(run / 'process_scores.parquet', index=False)
        score = compute_scores(evaluation, predictions)
        write_new_json(run / 'scores.json', score)
        assert_locks(manifests)
        check_environment(environment)
        report.update(status='pending_independent_verification', metrics=score, fitted_models=len(records),
            protected_unchanged=True, parent_unchanged=True, shared_legacy_members_unchanged=True,
            inherited_predictions_unchanged=True, elapsed_seconds=time.perf_counter()-started,
            feature_missingness={PRIMARY_CANDIDATE: dict(zip(names, inputs['eval_features'].isna().mean().tolist()))},
            training_maturity_summary=[{name: record[name] for name in ('model_id', 'train_rows', 'label_counts',
                'aqi_category_counts', 'input_hour_observed_counts', 'causal_eligible_counts', 'exact_t1_rows',
                'exact_t1_fraction', 'shared_legacy_member_rows', 'additional_calendar_member_rows',
                'training_last_timestamp', 'max_label_available_at')} for record in records])
    except BaseException as error:
        report.update(status='execution_failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['output_sha256'] = {name: bridge.sha(run / name) for name in file_inventory(run)}
        write_new_json(run / 'report.json', report)
    bridge.event('pending_independent_verification', run_dir=str(run), yearly_gates_passed=score['gates']['passed'])
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--execute', action='store_true', help='Only after root protocol/code/test approval')
    args = parser.parse_args(argv)
    started = time.perf_counter()
    manifests, environment, parent_report = provenance()
    inputs = prepare_inputs(parent_report)
    assert_locks(manifests)
    if args.preflight_only:
        bridge.event('preflight_passed', **cohort_counts(inputs, parent_report),
            frozen_files=len(manifests['frozen_sha256']), training_started=False)
        return 0
    execute(inputs, manifests, environment, parent_report, started)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
