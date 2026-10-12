"""Isolated causal AQI-state comparison; no deployment."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from numbers import Real
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROXY_PATH = ROOT / 'experiments/web_recovery_gefs_proxy_20261003/proxy_runner.py'
PROXY_SHA256 = '5b9c0377c14816fc5e83975fa5360d6b7fd51b23b28500a76f1a08992d6ad7b3'
PARENT_RUN = ROOT / 'experiments/web_recovery_gefs_proxy_20261003/runs/20261003T145140578727Z'
PARENT_REPORT_SHA256 = 'd553ab51dbb596c0c01960c340701386ac87635b21d611e4724ab662f3b7555b'
PARENT_VERIFICATION_SHA256 = 'b11f7163cb14a8225995dc1ebe8c46761d9f7e053a876d8f99c3cee9bd316807'
PARENT_VERIFIER_SHA256 = '840b2e8308eb5084ff8d2ac64e5ae9150af3c57621e00fe591b443d065ff1457'
if hashlib.sha256(PROXY_PATH.read_bytes()).hexdigest() != PROXY_SHA256:
    raise ValueError('Frozen parent proxy source changed')
spec = importlib.util.spec_from_file_location('immutable_state_proxy', PROXY_PATH)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
bridge, np, pd = proxy.bridge, proxy.np, proxy.pd
STATE_FIELDS = ['aqi_observed_high_run_hours', 'aqi_high_run_start_observed']
NEW_ARMS = ('state_control', 'state_values')
INHERITED_ARMS = ('archive', 'weather_values')
ARMS = INHERITED_ARMS + NEW_ARMS
PAIRS = tuple(('archive', arm) for arm in ARMS if arm != 'archive') + (('state_control', 'state_values'), ('weather_values', 'state_values'))
PRIMARY_CANDIDATE = 'state_values'
DIAGNOSTIC_ONLY_ARMS = ('state_control', 'weather_values')
SCOPE = 'offline AQI observed-state against equal-dimensional control; no live input claim'
STATE_TIME_SEMANTICS = 'observed high-AQI run lower bound; start flag only from immediately preceding observed clean hour'
STATE_SCOPE = 'historical full-grid replay only; short-sequence and live inputs require separate validation'
RAW_SOURCES = ('data/interim/cnemc_chengdu_2014_weather.parquet', 'data/processed/city_labeled.parquet',
    'data/interim/cnemc_chengdu_2022_weather.parquet', 'data/interim/cnemc_chengdu_2023_weather.parquet',
    'data/interim/cnemc_chengdu_2024_2025_weather.parquet')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def observed_aqi_state(frame):
    if not frame.columns.is_unique or not {'timestamp', 'aqi'}.issubset(frame):
        raise ValueError('Unique timestamp and AQI columns required')
    try:
        times = pd.DatetimeIndex(frame.timestamp)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid timestamp grid') from error
    if times.empty or times.tz is None or times.hasnans or not times.is_unique:
        raise ValueError('Nonempty unique timezone-aware timestamp grid required')
    utc = times.tz_convert('UTC')
    if not np.all(utc.asi8 % (3600 * 10**9) == 0) or not times.equals(pd.date_range(times[0], times[-1], freq='h')):
        raise ValueError('Complete sorted exact-hour grid required')
    values = []
    for value in frame.aqi:
        if value is None or value is pd.NA:
            values.append(np.nan)
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError('AQI must be real numeric or missing, not boolean/complex/string')
        else:
            value = float(value)
            if np.isinf(value):
                raise ValueError('AQI must not be infinite')
            values.append(value)
    runs, starts = np.full(len(values), np.nan), np.full(len(values), np.nan)
    length, observed_start, preceding_clean = 0, 0., False
    for index, value in enumerate(values):
        if np.isnan(value):
            length, observed_start, preceding_clean = 0, 0., False
        elif value <= 100:
            runs[index] = 0.
            length, observed_start, preceding_clean = 0, 0., True
        else:
            if length == 0:
                observed_start = float(preceding_clean)
            length += 1
            runs[index], starts[index], preceding_clean = float(length), observed_start, False
    return pd.DataFrame({'timestamp': frame.timestamp, STATE_FIELDS[0]: runs, STATE_FIELDS[1]: starts}, index=frame.index)


def attach_state(base, names, state):
    if len(names) != len(set(names)) or len(names) != 99 or set(STATE_FIELDS).intersection(names):
        raise ValueError('Fixed original 99-feature schema required')
    if not set(names).issubset(base) or not base.columns.is_unique:
        raise ValueError('Original 99-feature columns absent or ambiguous')
    if not base.timestamp.is_unique or base.timestamp.isna().any() or not state.timestamp.is_unique:
        raise ValueError('Unique complete origins required')
    positions = pd.Index(state.timestamp).get_indexer(base.timestamp)
    if (positions < 0).any():
        raise ValueError('All origins must exist on the full state grid')
    combined = [*names, *STATE_FIELDS]
    augmented = base.copy()
    augmented[STATE_FIELDS] = state.iloc[positions][STATE_FIELDS].to_numpy(dtype=float)
    truth = augmented[combined].astype(float)
    if np.isinf(truth.to_numpy()).any():
        raise ValueError('Infinite feature')
    control = truth.copy()
    control[STATE_FIELDS] = np.where(truth[STATE_FIELDS].notna(), 0., np.nan)
    np.testing.assert_array_equal(truth[names], base[names])
    np.testing.assert_array_equal(control[names], base[names])
    np.testing.assert_array_equal(truth.isna(), control.isna())
    return augmented, {'state_control': control, 'state_values': truth}, combined


def main_gates(metrics):
    primary = bridge.year_gates({'arms': {'archive': metrics['arms']['archive'],
        'legacy_control': metrics['arms']['state_control'], 'interval_candidate': metrics['arms']['state_values']},
        'paired': {'interval_candidate_vs_archive': metrics['paired']['state_values_vs_archive']}})
    primary['checks']['coverage_100'] = all(metrics['arms'][arm]['prediction_coverage'] == 1. for arm in ARMS)
    primary['passed'] = all(primary['checks'].values())
    primary['failure_reasons'] = [name for name, passed in primary['checks'].items() if not passed]
    control, candidate = metrics['arms']['state_control'], metrics['arms']['state_values']
    paired = metrics['paired']['state_values_vs_state_control']
    compare = lambda a, b: a is not None and b is not None and a <= b
    checks = {'coverage_100': all(metrics['arms'][arm]['prediction_coverage'] == 1. for arm in ARMS),
        'exact_row_strict_improvement': candidate['exact_row_mae'] is not None and control['exact_row_mae'] is not None
            and candidate['exact_row_mae'] < control['exact_row_mae'],
        'exact_process_not_worse': compare(candidate['exact_block_mae'], control['exact_block_mae']),
        'tail_exact_block_not_worse': compare(candidate['tail24_block_mae'], control['tail24_block_mae']),
        'paired_row_upper_nonpositive': compare(paired['all_row_delta_upper'], 0.),
        'paired_block_upper_nonpositive': compare(paired['all_block_delta_upper'], 0.)}
    information = {'checks': checks, 'passed': all(checks.values()),
        'failure_reasons': [key for key, value in checks.items() if not value]}
    return {'primary': primary, 'state_information': information,
        'passed': primary['passed'] and information['passed'],
        'failure_reasons': ['primary:' + key for key in primary['failure_reasons']]
            + ['state_information:' + key for key in information['failure_reasons']]}


def validate_model_inventory(models, blocks):
    expected = {(arm, int(block)) for arm in NEW_ARMS for block in blocks}
    actual = [(record['arm'], int(record['continuity_block'])) for record in models]
    if len(expected) != 64 or len(actual) != 64 or set(actual) != expected or len({row['model_id'] for row in models}) != 64:
        raise ValueError('Expected exactly 64 unique fixed arm/block models')


def validate_parent(report, verification, report_digest):
    if report_digest != PARENT_REPORT_SHA256 or report.get('status') != 'pending_independent_verification':
        raise ValueError('Parent report is not the fixed completed run')
    if not (verification.get('status') == 'passed' and verification.get('independent_verification_passed') is True
            and verification.get('report_sha256') == report_digest
            and verification.get('verifier_sha256') == PARENT_VERIFIER_SHA256):
        raise ValueError('Parent independent verification chain is not exact and passed')


def cohort_counts(inputs, parent_report):
    counts = dict(parent_report['counts'])
    population = inputs['populations']['state_control']
    if counts['planned_fits'] != 64 or counts['new_training_population'] != len(population) or counts['new_training_population_label_counts'] != bridge.label_counts(population):
        raise ValueError('Parent population/planned-fit counts changed')
    return counts


def assert_locks(manifests):
    bridge.assert_hashes(manifests['frozen_sha256'])
    proxy.validate_protected_inventory(ROOT, manifests['protected_sha256'])
    parent = Path(manifests['parent_run_dir'])
    actual = sorted(path.relative_to(parent).as_posix() for path in parent.rglob('*') if path.is_file())
    if actual != manifests['parent_artifact_inventory']:
        raise ValueError('Parent artifact exact inventory changed')


def provenance_manifests():
    if bridge.sha(PARENT_RUN / 'report.json') != PARENT_REPORT_SHA256 or bridge.sha(PARENT_RUN / 'verification.json') != PARENT_VERIFICATION_SHA256:
        raise ValueError('Fixed parent report or verification changed')
    report, verification = read_json(PARENT_RUN / 'report.json'), read_json(PARENT_RUN / 'verification.json')
    validate_parent(report, verification, bridge.sha(PARENT_RUN / 'report.json'))
    bridge.assert_hashes(report['frozen_sha256'])
    parent_hashes, inventory = proxy.parent_artifact_manifest(PARENT_RUN, report)
    if len(inventory) != 76 or len(report['frozen_sha256']) != 621:
        raise ValueError('Fixed parent inventory/count differs')
    files = [HERE / name for name in ('PROTOCOL.md', 'aqi_state_runner.py', 'test_aqi_state_runner.py', 'verify_state.py', 'test_verify_state.py')]
    new_code = {str(path.resolve()): bridge.sha(path) for path in files}
    code = bridge.merge_hash_manifests(report['code_sha256'], new_code)
    frozen = bridge.merge_hash_manifests(report['frozen_sha256'], parent_hashes, code)
    environment = read_json(PARENT_RUN / 'environment.json')
    if str(Path(sys.executable).resolve()) != environment['python_executable'] or environment['python_version'] != sys.version:
        raise ValueError('Exact research interpreter required')
    if any(importlib.metadata.version(name) != version for name, version in environment['packages'].items()):
        raise ValueError('Parent research environment versions changed')
    manifests = {name: report[name] for name in ('input_sha256', 'protected_sha256', 'environment_sha256',
        'historical_input_sha256', 'historical_audit_drift', 'snapshot_source_sha256', 'snapshot_report_sha256')}
    manifests.update(code_sha256=code, frozen_sha256=frozen, original_scope_sha256=report['frozen_sha256'],
        parent_sha256=parent_hashes, parent_run_dir=str(PARENT_RUN), parent_artifact_inventory=inventory,
        parent_report_sha256=PARENT_REPORT_SHA256, parent_verification_sha256=PARENT_VERIFICATION_SHA256,
        parent_verification_report_sha256=verification['report_sha256'], parent_verifier_sha256=PARENT_VERIFIER_SHA256)
    assert_locks(manifests)
    return manifests, environment, report


def rebuild_raw_grid(base_names):
    parts = []
    for relative in RAW_SOURCES:
        frame = pd.read_parquet(ROOT / relative)
        part = frame[['timestamp', *[name for name in bridge.MEASUREMENT_COLUMNS if name in frame]]].copy()
        part['timestamp'] = pd.to_datetime(part.timestamp, utc=True).dt.tz_convert('Asia/Shanghai')
        for name in bridge.MEASUREMENT_COLUMNS:
            if name not in part:
                part[name] = np.nan
        parts.append(part[['timestamp', *bridge.MEASUREMENT_COLUMNS]])
    raw = pd.concat(parts, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if raw.timestamp.duplicated().any():
        raise ValueError('Raw measurement source overlap')
    full = bridge.rebuild_grid(raw, base_names)
    stored = pd.read_parquet(bridge.GRID_PATH)
    columns = ['timestamp', *base_names, 'continuity_block', 'episode_id', 'eligible', 'pollution_state', 'input_hour_observed']
    bridge.check_equal_columns(full, stored, columns)
    if len(full) != 102024:
        raise ValueError('Fixed complete raw hour grid count changed')
    return full


def prepare_inputs(parent_report):
    names = list(parent_report['feature_names'])
    if len(names) != 99 or names[-12:] != proxy.WEATHER_FIELDS + proxy.AGE_FIELDS:
        raise ValueError('Fixed parent 99 schema/order changed')
    full = rebuild_raw_grid(names[:87])
    state = observed_aqi_state(full[['timestamp', 'aqi']])
    population = pd.read_parquet(PARENT_RUN / 'population.parquet')
    evaluation = pd.read_parquet(PARENT_RUN / 'evaluation.parquet')
    if (len(population), len(evaluation), evaluation.continuity_block.nunique()) != (8638, 701, 32):
        raise ValueError('All fixed training/evaluation origins required')
    augmented, train_features, combined = attach_state(population, names, state)
    with np.load(PARENT_RUN / 'features.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['feature_names'], names)
        np.testing.assert_array_equal(saved['timestamps'], pd.DatetimeIndex(evaluation.timestamp).asi8)
        eval_base = evaluation.copy()
        eval_base[names] = saved['weather_values']
        np.testing.assert_array_equal(population.set_index('timestamp').loc[evaluation.timestamp, names], saved['weather_values'])
    _, features, eval_names = attach_state(eval_base, names, state)
    if eval_names != combined:
        raise ValueError('Training/evaluation state schema differs')
    prior = pd.read_parquet(PARENT_RUN / 'predictions.parquet')
    bridge.check_equal_columns(evaluation, prior, list(evaluation))
    predictions = {arm: prior[arm].to_numpy(float).copy() for arm in INHERITED_ARMS}
    parent_members = pd.read_parquet(PARENT_RUN / 'training_members.parquet')
    parent_members = parent_members.loc[parent_members.arm.eq('weather_values')]
    member_rows = {int(block): rows.population_row.tolist() for block, rows in parent_members.groupby('evaluation_block', sort=True)}
    if len(parent_members) != 266522 or set(member_rows) != set(evaluation.continuity_block):
        raise ValueError('Parent complete maturity inventory changed')
    populations = {}
    for arm in NEW_ARMS:
        populations[arm] = augmented.copy()
        populations[arm][combined] = train_features[arm]
    for block, rows in evaluation.groupby('continuity_block', sort=True):
        cutoff = rows.cutoff.iloc[0]
        if not rows.cutoff.eq(cutoff).all():
            raise ValueError('One block requires exactly one unchanged cutoff')
        mature_members(populations['state_values'], cutoff, block, 'state_values', f'state_values_b{block}', member_rows[int(block)])
    return {'full': full, 'evaluation': evaluation, 'population': augmented, 'populations': populations,
        'feature_names': combined, 'features': features, 'inherited_predictions': predictions, 'parent_member_rows': member_rows}


def mature_members(population, cutoff, block, arm, model_id, expected_population_rows):
    return proxy.mature_members(population, cutoff, block, arm, model_id, expected_population_rows)


def fit_block(population, names, eval_features, block, cutoff, arm, run, expected_population_rows):
    if arm not in NEW_ARMS:
        raise ValueError('Only two approved state arms can be fitted')
    model_id = f'{arm}_b{block}'
    members = mature_members(population, cutoff, block, arm, model_id, expected_population_rows)
    train = population.loc[bridge.mature_mask(population, cutoff, block)]
    features = train[names].astype(float)
    model = bridge.ObservedAFTSurvivalRegressor(**bridge.MODEL_CONFIG)
    model.fit_bounds(features, train.lower_bound_hours, train.upper_bound_hours)
    config = json.loads(model.booster.save_config())
    rounds, booster_features = model.booster.num_boosted_rounds(), model.booster.feature_names
    if rounds != 500 or booster_features != names:
        raise ValueError('Actual model rounds/schema differs')
    prediction = model.predict(eval_features)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('All origins require finite nonnegative predictions')
    path = run / 'models' / f'{model_id}.json'
    model.booster.save_model(path)
    restored = bridge.xgb.Booster(params={'nthread': 1})
    restored.load_model(path)
    again = restored.predict(bridge.xgb.DMatrix(eval_features.astype(float), missing=np.nan)).clip(min=0.)
    np.testing.assert_array_equal(prediction, again)
    record = {'model_id': model_id, 'arm': arm, 'continuity_block': int(block), 'cutoff': cutoff.isoformat(),
        'model_path': path.relative_to(run).as_posix(), 'training_member_path': 'training_members.parquet',
        'parent_member_model_id': f'weather_values_b{block}', 'train_rows': len(train), 'eval_rows': len(eval_features),
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
        'paired': {f'{candidate}_vs_{baseline}': bridge.paired_error_summary(evaluation, predictions[baseline], predictions[candidate])
                   for baseline, candidate in PAIRS}}


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
            record = {'year': int(year), 'continuity_block': int(block), 'arm': arm, 'old_events': int(rows.event_key.nunique())}
            record.update({name: value for name, value in score.items() if name != 'subgroups'})
            for name, values in score['subgroups'].items():
                record.update({f'{name}_{field}': value for field, value in values.items()})
            records.append(record)
    return pd.DataFrame(records)


def execute(inputs, manifests, environment, parent_report, started):
    assert_locks(manifests)
    run = bridge.new_run_directory(HERE / 'runs')
    (run / 'models').mkdir()
    report = dict(schema_version=1, status='running', run_dir=str(run), years=list(bridge.YEARS), **manifests,
        configuration=bridge.MODEL_CONFIG, aft_parameters=bridge.AFT_PARAMETERS, feature_names=inputs['feature_names'],
        counts=cohort_counts(inputs, parent_report), primary_candidate=PRIMARY_CANDIDATE,
        diagnostic_only_arms=list(DIAGNOSTIC_ONLY_ARMS), new_refitted_arms=list(NEW_ARMS), inherited_arms=list(INHERITED_ARMS),
        historical_shadow_evaluation=True, blind_test=False, deployed=False, production_ready=False,
        forecast_time_semantics='archive_timestamp_proxy', scope=SCOPE, state_fields=STATE_FIELDS,
        state_time_semantics=STATE_TIME_SEMANTICS, state_scope=STATE_SCOPE,
        target=parent_report['target'], archive_is_primary_reference=True, uncertainty_scope=parent_report['uncertainty_scope'])
    bridge.write_json(run / 'report.json', report)
    bridge.event('run_created', run_dir=str(run))
    try:
        evaluation, names = inputs['evaluation'], inputs['feature_names']
        evaluation.to_parquet(run / 'evaluation.parquet', index=False)
        inputs['population'].to_parquet(run / 'population.parquet', index=False)
        np.savez_compressed(run / 'features.npz', timestamps=pd.DatetimeIndex(evaluation.timestamp).asi8,
            feature_names=np.asarray(names, dtype=str), **{arm: inputs['features'][arm].to_numpy(float) for arm in NEW_ARMS})
        bridge.write_json(run / 'environment.json', environment)
        (run / 'execution_record_at_start.md').write_bytes((HERE / 'EXECUTION_RECORD.md').read_bytes())
        predictions = dict(inputs['inherited_predictions'], **{arm: np.full(len(evaluation), np.nan) for arm in NEW_ARMS})
        member_tables, models = [], []
        for block, rows in evaluation.groupby('continuity_block', sort=True):
            cutoff, positions = rows.cutoff.iloc[0], rows.index.to_numpy()
            if not rows.cutoff.eq(cutoff).all():
                raise ValueError('One block requires one fixed full-grid cutoff')
            for arm in NEW_ARMS:
                values, members, record = fit_block(inputs['populations'][arm], names, inputs['features'][arm].iloc[positions],
                    block, cutoff, arm, run, inputs['parent_member_rows'][int(block)])
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
            feature_missingness={arm: dict(zip(names, np.isnan(inputs['features'][arm].to_numpy(float)).mean(axis=0).tolist())) for arm in NEW_ARMS},
            training_maturity_summary=[{name: model[name] for name in ('model_id', 'train_rows', 'label_counts', 'training_last_timestamp', 'max_label_available_at')} for model in models],
            elapsed_seconds=time.perf_counter()-started,
            limits=['Known historical 2024/2025; not a blind test', 'GEFS known_at is an archive proxy, not verified first publication',
                'Observed high-AQI length is a lower bound after missing measurements or left boundary',
                'Full historical state is unavailable from an isolated short-sequence or single-time input',
                'Passing accuracy gates does not authorize deployment or infer live accuracy',
                'All original origins and global label intervals retained; no dynamic unfinished-event supervision claim',
                'Independent verification reuses immutable prior independent mathematics; not a wholly new second implementation'])
    except BaseException as error:
        report.update(status='execution_failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['output_sha256'] = {path.relative_to(run).as_posix(): bridge.sha(path) for path in run.rglob('*') if path.is_file() and path.name != 'report.json'}
        bridge.write_json(run / 'report.json', report)
    bridge.event('pending_independent_verification', run_dir=str(run), yearly_gates_passed=scores['gates']['passed'])
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--execute', action='store_true', help='Only after explicit frozen protocol/source/test approval')
    args = parser.parse_args()
    started = time.perf_counter()
    manifests, environment, parent_report = provenance_manifests()
    inputs = prepare_inputs(parent_report)
    assert_locks(manifests)
    if args.preflight_only:
        bridge.event('preflight_passed', **cohort_counts(inputs, parent_report), frozen_files=len(manifests['frozen_sha256']), training_started=False)
        return
    execute(inputs, manifests, environment, parent_report, started)


if __name__ == '__main__':
    main()
