"""Offline GEFS archive-proxy comparison; no deployment."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRIDGE_DIR = ROOT / 'experiments/web_recovery_bridge_20261002'
PARENT_RUN = ROOT / 'experiments/web_recovery_factors_20261002/runs/20261002T161429405383Z'
PARENT_REPORT_SHA256 = '21aa39d5a08a683e23904000eeeccbb3c31fda30201cf30e64b4cf28ab063998'
PARENT_VERIFICATION_SHA256 = '17f3a9e71d15e95c3f9f02149a5bd1c1f27abe784682979c050263e44e0f8566'
PARENT_VERIFIER_SHA256 = '1c3052b2f51023d8e08649ed47dff614f05065ba1dd161d9964b8151400c6c6e'
spec = importlib.util.spec_from_file_location('immutable_bridge_parent', BRIDGE_DIR / 'bridge_runner.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
np, pd = bridge.np, bridge.pd
from haze_forecast.forecast_snapshot_reuse import reuse_forecast_snapshots

SNAPSHOT_PATH = ROOT / 'artifacts/task01_recovery_head/reusable_gefs_snapshots_v1/snapshots.parquet'
SNAPSHOT_SHA256 = 'a686840d341e3e838c3e0c7021da9adf34200303a8d0733ee5b9e390b1f53ca4'
SNAPSHOT_REPORT_SHA256 = '08ee7cb59d6bae4dd3a624a9f1f526e27bd1ed8216929af5d9e9466b6643dba6'
WEATHER_FIELDS = [
    'gefs_mean_temperature_c_f024', 'gefs_mean_relative_humidity_pct_f024',
    'gefs_mean_wind_u_ms_f024', 'gefs_mean_wind_v_ms_f024',
    'gefs_spread_temperature_c_f024', 'gefs_spread_relative_humidity_pct_f024',
    'gefs_spread_wind_u_ms_f024', 'gefs_spread_wind_v_ms_f024',
    'gefs_control_boundary_layer_height_m_f024']
AGE_FIELDS = ['forecast_known_age_hours', 'forecast_run_age_hours', 'forecast_remaining_lead_hours']
NEW_ARMS = ('metadata_control', 'weather_values')
INHERITED_ARMS = ('archive', 'grid_all_legacy_origins')
PRIMARY_CANDIDATE = 'weather_values'
DIAGNOSTIC_ONLY_ARMS = ('metadata_control',)
ARMS = INHERITED_ARMS + NEW_ARMS
PAIRS = tuple(('archive', arm) for arm in ARMS if arm != 'archive') + (('metadata_control', 'weather_values'),)



def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_parent(report, verification, report_digest):
    if report_digest != PARENT_REPORT_SHA256 or report.get('status') != 'pending_independent_verification':
        raise ValueError('Parent report is not the predeclared completed run')
    if not (verification.get('status') == 'passed' and verification.get('independent_verification_passed') is True
            and verification.get('report_sha256') == report_digest
            and verification.get('verifier_sha256') == PARENT_VERIFIER_SHA256):
        raise ValueError('Parent independent verification chain is not passed and exact')


def parent_artifact_manifest(parent, report):
    parent = Path(parent).resolve()
    actual = sorted(path.relative_to(parent).as_posix() for path in parent.rglob('*') if path.is_file())
    expected = sorted(set(report['output_sha256']) | {'report.json', 'verification.json'})
    if actual != expected:
        raise ValueError('Parent artifact inventory changed')
    bridge.assert_hashes({str(parent / name): digest for name, digest in report['output_sha256'].items()})
    return {str((parent / name).resolve()): bridge.sha(parent / name) for name in actual}, actual


def validate_protected_inventory(root, hashes):
    for name in ('artifacts/cnemc_2025_deployment_candidate', 'artifacts/strict_history_backtest',
                 'project-showcase', 'sites-haze/dist', 'releases/runtime_20261002'):
        folder = (Path(root) / name).resolve()
        expected = {str(Path(path).resolve()) for path in hashes
                    if Path(path).resolve().is_relative_to(folder)}
        actual = {str(path.resolve()) for path in folder.rglob('*') if path.is_file()}
        if not expected or actual != expected:
            raise ValueError(f'protected exact inventory differs: {name}')


def validate_snapshot_report(path):
    if bridge.sha(path) != SNAPSHOT_REPORT_SHA256:
        raise ValueError('Snapshot source report hash changed')


def assert_locks(manifests):
    bridge.assert_hashes(manifests['frozen_sha256'])
    validate_protected_inventory(ROOT, manifests['protected_sha256'])
    parent = Path(manifests['parent_run_dir'])
    inventory = sorted(path.relative_to(parent).as_posix() for path in parent.rglob('*') if path.is_file())
    if inventory != manifests['parent_artifact_inventory']:
        raise ValueError('Parent artifact inventory changed')


def provenance_manifests():
    report_path, verification_path = PARENT_RUN / 'report.json', PARENT_RUN / 'verification.json'
    if bridge.sha(report_path) != PARENT_REPORT_SHA256 or bridge.sha(verification_path) != PARENT_VERIFICATION_SHA256:
        raise ValueError('Parent report or verification hash changed')
    report, verification = read_json(report_path), read_json(verification_path)
    validate_parent(report, verification, bridge.sha(report_path))
    bridge.assert_hashes(report['frozen_sha256'])
    parent_hashes, inventory = parent_artifact_manifest(PARENT_RUN, report)
    files = [HERE / name for name in ('PROTOCOL.md', 'proxy_runner.py', 'test_proxy_runner.py',
        'verify_proxy.py', 'test_verify_proxy.py')]
    files.append(ROOT / 'src/haze_forecast/forecast_snapshot_reuse.py')
    if bridge.sha(SNAPSHOT_PATH) != SNAPSHOT_SHA256:
        raise ValueError('Snapshot hash changed')
    snapshot_report_path = SNAPSHOT_PATH.with_name('report.json')
    validate_snapshot_report(snapshot_report_path)
    snapshot_report = read_json(snapshot_report_path)
    if (snapshot_report['output_sha256']['snapshots.parquet'] != SNAPSHOT_SHA256
            or snapshot_report['fields'] != WEATHER_FIELDS):
        raise ValueError('Snapshot source report changed')
    historical_sources = snapshot_report['input_sha256']
    excluded = str((ROOT / 'src/haze_forecast/app.py').resolve())
    snapshot_sources = {path: digest for path, digest in historical_sources.items() if path != excluded}
    bridge.assert_hashes(snapshot_sources)
    availability = ROOT / 'src/haze_forecast/operational_gefs_availability.py'
    snapshot_sources.update({str(snapshot_report_path.resolve()): bridge.sha(snapshot_report_path),
        str(availability.resolve()): bridge.sha(availability), str(SNAPSHOT_PATH.resolve()): SNAPSHOT_SHA256})
    files.extend(Path(path) for path in snapshot_sources if Path(path).suffix == '.py')
    new_code = {str(path.resolve()): bridge.sha(path) for path in files}
    code = bridge.merge_hash_manifests(report['code_sha256'], new_code)
    frozen = bridge.merge_hash_manifests(report['frozen_sha256'], parent_hashes, code,
        snapshot_sources)
    environment = read_json(PARENT_RUN / 'environment.json')
    if str(Path(sys.executable).resolve()) != environment['python_executable']:
        raise ValueError('Factor runner must use the exact research interpreter')
    if any(importlib.metadata.version(name) != version for name, version in environment['packages'].items()):
        raise ValueError('Parent research package versions changed')
    manifests = {'input_sha256': bridge.merge_hash_manifests(report['input_sha256'], snapshot_sources), 'protected_sha256': report['protected_sha256'],
        'snapshot_source_sha256': snapshot_sources, 'snapshot_report_sha256': bridge.sha(snapshot_report_path),
        'code_sha256': code, 'environment_sha256': report['environment_sha256'], 'frozen_sha256': frozen,
        'historical_input_sha256': report['historical_input_sha256'], 'historical_audit_drift': report['historical_audit_drift'],
        'original_scope_sha256': report['frozen_sha256'], 'parent_sha256': parent_hashes,
        'parent_run_dir': str(PARENT_RUN), 'parent_artifact_inventory': inventory,
        'parent_report_sha256': PARENT_REPORT_SHA256, 'parent_verification_sha256': PARENT_VERIFICATION_SHA256,
        'parent_verification_report_sha256': verification['report_sha256'], 'parent_verifier_sha256': PARENT_VERIFIER_SHA256}
    assert_locks(manifests)
    return manifests, environment, report


def attach_weather(base, base_names, snapshots):
    fields = [name for name in snapshots.columns if name not in ('known_at', 'run_at', 'valid_at')]
    if fields != WEATHER_FIELDS or len(base_names) != 87 or len(set(base_names)) != 87:
        raise ValueError('Fixed snapshot fields or original 87-feature schema changed')
    if not (snapshots.valid_at - snapshots.run_at).eq(pd.Timedelta(hours=24)).all():
        raise ValueError('Only f024 snapshots are allowed')
    if not snapshots.known_at.lt(snapshots.valid_at).all():
        raise ValueError('Snapshot must be known before its valid time')
    names = base_names + WEATHER_FIELDS + AGE_FIELDS
    if len(set(names)) != 99:
        raise ValueError('Features collide')
    joined = reuse_forecast_snapshots(snapshots, base.timestamp, WEATHER_FIELDS)
    np.testing.assert_array_equal(pd.DatetimeIndex(joined.timestamp).asi8, pd.DatetimeIndex(base.timestamp).asi8)
    observed = joined.known_at.notna()
    if not ((joined.loc[observed, 'known_at'] <= joined.loc[observed, 'timestamp'])
            & (joined.loc[observed, 'timestamp'] < joined.loc[observed, 'valid_at'])).all():
        raise ValueError('Forecast timestamp availability violated')
    result = base.copy()
    for name in WEATHER_FIELDS + AGE_FIELDS + ['known_at', 'run_at', 'valid_at']:
        result[name] = joined[name].to_numpy()
    truth = result[names].astype(float)
    control = truth.copy()
    control.loc[:, WEATHER_FIELDS] = np.where(truth[WEATHER_FIELDS].notna(), 0., np.nan)
    if np.isinf(truth.to_numpy()).any():
        raise ValueError('Infinite feature')
    np.testing.assert_array_equal(control.isna(), truth.isna())
    np.testing.assert_array_equal(truth[base_names], base[base_names])
    return result, {'metadata_control': control, 'weather_values': truth}, names


def prepare_inputs(parent_report):
    evaluation = pd.read_parquet(PARENT_RUN / 'evaluation.parquet')
    population = pd.read_parquet(PARENT_RUN / 'grid_population.parquet')
    base_names = list(parent_report['feature_names'])
    snapshots = pd.read_parquet(SNAPSHOT_PATH)
    if (len(population), len(evaluation), evaluation.continuity_block.nunique()) != (8638, 701, 32):
        raise ValueError('Fixed population or evaluation dimensions changed')
    if not population.timestamp.is_unique or not evaluation.timestamp.is_unique:
        raise ValueError('Ambiguous origins')
    augmented, train_features, names = attach_weather(population, base_names, snapshots)
    with np.load(PARENT_RUN / 'features.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['feature_names'], base_names)
        np.testing.assert_array_equal(saved['timestamps'], pd.DatetimeIndex(evaluation.timestamp).asi8)
        eval_base = evaluation.copy()
        eval_base[base_names] = saved['grid_all_legacy_origins']
    _, features, eval_names = attach_weather(eval_base, base_names, snapshots)
    if eval_names != names:
        raise ValueError('Training and evaluation schemas differ')
    prior = pd.read_parquet(PARENT_RUN / 'predictions.parquet')
    bridge.check_equal_columns(evaluation, prior, list(evaluation))
    predictions = {arm: prior[arm].to_numpy(float).copy() for arm in INHERITED_ARMS}
    parent_members = pd.read_parquet(PARENT_RUN / 'training_members.parquet')
    parent_members = parent_members.loc[parent_members.arm.eq('grid_all_legacy_origins')]
    member_rows = {int(block): rows.population_row.tolist()
        for block, rows in parent_members.groupby('evaluation_block', sort=True)}
    if len(parent_members) != 266522 or set(member_rows) != set(evaluation.continuity_block):
        raise ValueError('Parent mature-member inventory changed')
    populations = {}
    for arm in NEW_ARMS:
        populations[arm] = augmented.copy()
        populations[arm][names] = train_features[arm]
    return {'evaluation': evaluation, 'feature_names': names, 'population': augmented,
        'populations': populations, 'features': features,
        'inherited_predictions': predictions, 'parent_member_rows': member_rows}


def mature_members(population, cutoff, block, arm, model_id, expected_population_rows):
    mask = bridge.mature_mask(population, cutoff, block)
    rows = population.loc[mask]
    if rows.population_row.tolist() != list(expected_population_rows):
        raise ValueError('Parent mature training members differ')
    if rows.empty:
        raise ValueError('No mature training members')
    if not rows.loc[rows.known_at.notna(), 'known_at'].lt(cutoff).all():
        raise ValueError('Training forecast known_at must precede cutoff')
    members = rows[['population_row', 'timestamp', 'continuity_block', *bridge.BOUND_COLUMNS,
        'legacy_remaining_hours']].copy()
    members['model_id'], members['arm'] = model_id, arm
    members['evaluation_block'], members['cutoff'] = block, cutoff
    return members


def fit_block(population, names, eval_features, block, cutoff, arm, run, expected_population_rows):
    if arm not in NEW_ARMS:
        raise ValueError('Only the two predeclared new arms can be fitted')
    model_id = f'{arm}_b{block}'
    members = mature_members(population, cutoff, block, arm, model_id, expected_population_rows)
    train = population.loc[bridge.mature_mask(population, cutoff, block)]
    features = train[names].astype(float)
    model = bridge.ObservedAFTSurvivalRegressor(**bridge.MODEL_CONFIG)
    model.fit_bounds(features, train.lower_bound_hours, train.upper_bound_hours)
    config = json.loads(model.booster.save_config())
    rounds, booster_features = model.booster.num_boosted_rounds(), model.booster.feature_names
    if rounds != 500 or booster_features != names:
        raise ValueError('Actual model rounds or feature order differ from fixed configuration')
    prediction = model.predict(eval_features)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid prediction; no evaluation origin may be dropped')
    path = run / 'models' / f'{model_id}.json'
    model.booster.save_model(path)
    restored = bridge.xgb.Booster(params={'nthread': 1})
    restored.load_model(path)
    again = restored.predict(bridge.xgb.DMatrix(eval_features.astype(float), missing=np.nan)).clip(min=0.)
    np.testing.assert_array_equal(prediction, again)
    record = {'model_id': model_id, 'arm': arm, 'continuity_block': int(block), 'cutoff': cutoff.isoformat(),
        'model_path': path.relative_to(run).as_posix(), 'training_member_path': 'training_members.parquet',
        'parent_member_model_id': f'grid_all_legacy_origins_b{block}', 'train_rows': len(train), 'eval_rows': len(eval_features),
        'label_counts': bridge.label_counts(train), 'training_last_timestamp': train.timestamp.max().isoformat(),
        'max_label_available_at': train.label_available_at.max().isoformat(),
        'training_features_sha256': bridge.array_sha(features.to_numpy()),
        'training_timestamps_sha256': hashlib.sha256(pd.DatetimeIndex(train.timestamp).asi8.astype('<i8').tobytes()).hexdigest(),
        'feature_names': names, 'model_sha256': bridge.sha(path), 'configuration': bridge.MODEL_CONFIG,
        'aft_parameters': bridge.AFT_PARAMETERS, 'actual_trained_booster_config': config,
        'num_boosted_rounds': int(rounds), 'booster_feature_names': booster_features,
        'right_censor_policy': 'global archive-end labels only; all mature rows retained'}
    return prediction, members, record


def score_cohort(evaluation, predictions):
    return {'arms': {arm: bridge.score_arm(evaluation, predictions[arm]) for arm in ARMS},
        'paired': {f'{candidate}_vs_{baseline}': bridge.paired_error_summary(
            evaluation, predictions[baseline], predictions[candidate]) for baseline, candidate in PAIRS}}


def main_gates(metrics):
    primary = bridge.year_gates({'arms': {'archive': metrics['arms']['archive'],
        'legacy_control': metrics['arms']['metadata_control'], 'interval_candidate': metrics['arms']['weather_values']},
        'paired': {'interval_candidate_vs_archive': metrics['paired']['weather_values_vs_archive']}})
    control, candidate = metrics['arms']['metadata_control'], metrics['arms']['weather_values']
    paired = metrics['paired']['weather_values_vs_metadata_control']
    compare = lambda a, b: a is not None and b is not None and a <= b
    checks = {
        'coverage_100': all(metrics['arms'][arm]['prediction_coverage'] == 1. for arm in ARMS),
        'exact_row_strict_improvement': candidate['exact_row_mae'] is not None and control['exact_row_mae'] is not None
            and candidate['exact_row_mae'] < control['exact_row_mae'],
        'exact_process_not_worse': compare(candidate['exact_block_mae'], control['exact_block_mae']),
        'tail_exact_block_not_worse': compare(candidate['tail24_block_mae'], control['tail24_block_mae']),
        'paired_row_upper_nonpositive': compare(paired['all_row_delta_upper'], 0.),
        'paired_block_upper_nonpositive': compare(paired['all_block_delta_upper'], 0.)}
    information = {'checks': checks, 'passed': all(checks.values()),
        'failure_reasons': [key for key, value in checks.items() if not value]}
    return {'primary': primary, 'weather_information': information,
        'passed': primary['passed'] and information['passed'],
        'failure_reasons': ['primary:' + key for key in primary['failure_reasons']]
            + ['weather_information:' + key for key in information['failure_reasons']]}


def validate_model_inventory(models, blocks):
    expected = {(arm, int(block)) for arm in NEW_ARMS for block in blocks}
    actual = [(record['arm'], int(record['continuity_block'])) for record in models]
    if len(expected) != 64 or len(actual) != 64 or set(actual) != expected:
        raise ValueError('Expected exactly 64 distinct fixed arm/block fits')
    if len({record['model_id'] for record in models}) != 64:
        raise ValueError('Duplicate model identity')


def row_scores(evaluation, predictions):
    result = evaluation.copy()
    lower, upper, truth = (result[name].to_numpy(float)
        for name in ('lower_bound_hours', 'upper_bound_hours', 'exact_hours'))
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
            record = {'year': int(year), 'continuity_block': int(block), 'arm': arm, 'old_events': int(rows.event_key.nunique())}
            record.update({name: value for name, value in score.items() if name != 'subgroups'})
            for name, values in score['subgroups'].items():
                record.update({f'{name}_{field}': value for field, value in values.items()})
            records.append(record)
    return pd.DataFrame(records)


def cohort_counts(inputs, parent_report):
    counts = dict(parent_report['counts'])
    population = inputs['populations'][NEW_ARMS[0]]
    if (counts['planned_fits'] != 64 or counts['new_training_population'] != len(population)
            or counts['new_training_population_label_counts'] != bridge.label_counts(population)):
        raise ValueError('Parent population counts or planned fits changed')
    return counts


def execute(inputs, manifests, environment, parent_report, started):
    assert_locks(manifests)
    run = bridge.new_run_directory(HERE / 'runs')
    (run / 'models').mkdir()
    report = dict(schema_version=1, status='running', run_dir=str(run), years=list(bridge.YEARS), **manifests,
        configuration=bridge.MODEL_CONFIG, aft_parameters=bridge.AFT_PARAMETERS, feature_names=inputs['feature_names'],
        counts=cohort_counts(inputs, parent_report), primary_candidate=PRIMARY_CANDIDATE,
        diagnostic_only_arms=list(DIAGNOSTIC_ONLY_ARMS), new_refitted_arms=list(NEW_ARMS), inherited_arms=list(INHERITED_ARMS),
        historical_shadow_evaluation=True, blind_test=False, deployed=False, production_ready=False,
        forecast_time_semantics='archive_timestamp_proxy',
        scope='offline archive-proxy GEFS values against metadata; no live availability claim',
        target=parent_report['target'], archive_is_primary_reference=True,
        uncertainty_scope=parent_report['uncertainty_scope'])
    bridge.write_json(run / 'report.json', report)
    bridge.event('run_created', run_dir=str(run))
    try:
        evaluation, names = inputs['evaluation'], inputs['feature_names']
        evaluation.to_parquet(run / 'evaluation.parquet', index=False)
        inputs['population'].to_parquet(run / 'population.parquet', index=False)
        np.savez_compressed(run / 'features.npz', timestamps=pd.DatetimeIndex(evaluation.timestamp).asi8,
            feature_names=np.asarray(names, dtype=str),
            **{arm: inputs['features'][arm].to_numpy(float) for arm in NEW_ARMS})
        bridge.write_json(run / 'environment.json', environment)
        (run / 'execution_record_at_start.md').write_bytes((HERE / 'EXECUTION_RECORD.md').read_bytes())
        predictions = dict(inputs['inherited_predictions'], **{arm: np.full(len(evaluation), np.nan) for arm in NEW_ARMS})
        member_tables, models = [], []
        for block, rows in evaluation.groupby('continuity_block', sort=True):
            cutoff, positions = rows.cutoff.iloc[0], rows.index.to_numpy()
            if not rows.cutoff.eq(cutoff).all():
                raise ValueError('One block must retain its unique full-grid cutoff')
            for arm in NEW_ARMS:
                values, members, record = fit_block(inputs['populations'][arm], names,
                    inputs['features'][arm].iloc[positions], block, cutoff, arm, run, inputs['parent_member_rows'][int(block)])
                predictions[arm][positions] = values
                member_tables.append(members)
                models.append(record)
                bridge.event('model_complete', arm=arm, block=int(block), train_rows=len(members), eval_rows=len(rows))
        validate_model_inventory(models, evaluation.continuity_block.unique())
        pd.concat(member_tables, ignore_index=True).to_parquet(run / 'training_members.parquet', index=False)
        bridge.write_json(run / 'models.json', {'schema_version': 1, 'feature_names': names, 'models': models})
        row_scores(evaluation, predictions).to_parquet(run / 'predictions.parquet', index=False)
        process_scores(evaluation, predictions).to_parquet(run / 'process_scores.parquet', index=False)
        annual, gates = {}, {}
        for year in bridge.YEARS:
            mask = evaluation.year.eq(year).to_numpy()
            annual[str(year)] = score_cohort(evaluation.loc[mask], {arm: values[mask] for arm, values in predictions.items()})
            gates[str(year)] = main_gates(annual[str(year)])
        scores = {'annual': annual, 'pooled': score_cohort(evaluation, predictions), 'gates': {'annual': gates,
            'passed': all(gates[str(year)]['passed'] for year in bridge.YEARS),
            'failure_reasons': [f'{year}:{name}' for year in bridge.YEARS for name in gates[str(year)]['failure_reasons']]}}
        bridge.write_json(run / 'scores.json', scores)
        assert_locks(manifests)
        report.update(status='pending_independent_verification', metrics=scores, fitted_models=len(models),
            protected_unchanged=True, parent_unchanged=True, accuracy_passed=scores['gates']['passed'],
            feature_missingness={arm: dict(zip(names, np.isnan(inputs['features'][arm].to_numpy(float)).mean(axis=0).tolist()))
                for arm in NEW_ARMS},
            training_maturity_summary=[{name: model[name] for name in ('model_id', 'train_rows', 'label_counts',
                'training_last_timestamp', 'max_label_available_at')} for model in models],
            elapsed_seconds=time.perf_counter()-started,
            limits=['Known historical 2024/2025; not a blind test', 'GEFS known_at is an archive proxy, not verified first publication',
                'Dynamic unfinished-event supervision remains unproven; fixed parent population is retained',
                'Historical weather issue-time availability and upstream imputation remain unproven',
                'Weather values can affect histogram binning, column sampling and regularization; not unique physical causality',
                'Passing accuracy gates only supports a further population-prefix audit, not production readiness',
                'Global archive-end censoring is not dynamically updated unfinished-event supervision',
                'Independent verification reuses immutable prior independent mathematics; no new second implementation'])
    except BaseException as error:
        report.update(status='execution_failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['output_sha256'] = {path.relative_to(run).as_posix(): bridge.sha(path)
            for path in run.rglob('*') if path.is_file() and path.name != 'report.json'}
        bridge.write_json(run / 'report.json', report)
    bridge.event('pending_independent_verification', run_dir=str(run), yearly_gates_passed=scores['gates']['passed'])
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--execute', action='store_true', help='Only after explicit protocol/code/test approval')
    args = parser.parse_args()
    started = time.perf_counter()
    manifests, environment, parent_report = provenance_manifests()
    inputs = prepare_inputs(parent_report)
    assert_locks(manifests)
    if args.preflight_only:
        bridge.event('preflight_passed', **cohort_counts(inputs, parent_report),
            frozen_files=len(manifests['frozen_sha256']), training_started=False)
        return
    execute(inputs, manifests, environment, parent_report, started)


if __name__ == '__main__':
    main()


