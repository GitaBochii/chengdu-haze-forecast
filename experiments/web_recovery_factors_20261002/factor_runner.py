"""Fixed conditional recovery factors; only two new interval-AFT arms are fitted."""
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
BRIDGE_DIR = ROOT / 'experiments/web_recovery_bridge_20261002'
PARENT_RUN = BRIDGE_DIR / 'runs/20261002T153221834258Z'
PARENT_REPORT_SHA256 = '70217776f86326cd9a9d4436af12d2d352b4f11ea69b2f1fc68becb1951f01d0'
PARENT_VERIFICATION_SHA256 = '7003b164439cbf87e4146459ae2f219bf6f3b45af1cc70407daf7935eba1d703'
PARENT_VERIFIER_SHA256 = '9477b5452830f82756219d317a6928e12708897a061cb442eeb65304fccd6b2b'
spec = importlib.util.spec_from_file_location('immutable_bridge_parent', BRIDGE_DIR / 'bridge_runner.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
np, pd = bridge.np, bridge.pd
NEW_ARMS = ('labels_only', 'grid_all_legacy_origins')
PRIMARY_CANDIDATE = 'grid_all_legacy_origins'
DIAGNOSTIC_ONLY_ARMS = ('labels_only',)
ARMS = bridge.ARMS + NEW_ARMS
PAIRS = tuple(('archive', arm) for arm in ARMS if arm != 'archive') + (
    ('legacy_control', 'labels_only'), ('labels_only', 'grid_all_legacy_origins'),
    ('grid_all_legacy_origins', 'interval_candidate'))


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


def assert_locks(manifests):
    bridge.assert_hashes(manifests['frozen_sha256'])
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
    files = [HERE / name for name in ('PROTOCOL.md', 'factor_runner.py', 'test_factor_runner.py',
        'verify_factors.py', 'test_verify_factors.py')]
    new_code = {str(path.resolve()): bridge.sha(path) for path in files}
    code = bridge.merge_hash_manifests(report['code_sha256'], new_code)
    frozen = bridge.merge_hash_manifests(report['frozen_sha256'], parent_hashes, code)
    environment = read_json(PARENT_RUN / 'environment.json')
    if str(Path(sys.executable).resolve()) != environment['python_executable']:
        raise ValueError('Factor runner must use the exact research interpreter')
    if any(importlib.metadata.version(name) != version for name, version in environment['packages'].items()):
        raise ValueError('Parent research package versions changed')
    manifests = dict(input_sha256=report['input_sha256'], protected_sha256=report['protected_sha256'],
        code_sha256=code, environment_sha256=report['environment_sha256'], frozen_sha256=frozen,
        historical_input_sha256=report['historical_input_sha256'], historical_audit_drift=report['historical_audit_drift'],
        original_scope_sha256=report['frozen_sha256'], parent_sha256=parent_hashes,
        parent_run_dir=str(PARENT_RUN), parent_artifact_inventory=inventory,
        parent_report_sha256=PARENT_REPORT_SHA256, parent_verification_sha256=PARENT_VERIFICATION_SHA256,
        parent_verification_report_sha256=verification['report_sha256'], parent_verifier_sha256=PARENT_VERIFIER_SHA256)
    assert_locks(manifests)
    return manifests, environment, report


def grid_population(legacy, full, feature_names):
    grid = full.set_index('timestamp')
    if not grid.index.is_unique or not legacy.timestamp.is_unique or not legacy.timestamp.isin(grid.index).all():
        raise ValueError('A legacy origin is missing or ambiguous on the fixed hourly grid')
    result = legacy.copy()
    result[feature_names] = grid.loc[legacy.timestamp, feature_names].to_numpy(float)
    if np.isinf(result[feature_names].to_numpy(float)).any():
        raise ValueError('Grid features must be finite or NaN')
    return result


def prepare_inputs(parent_report):
    full = pd.read_parquet(bridge.GRID_PATH)
    evaluation = pd.read_parquet(PARENT_RUN / 'evaluation.parquet')
    legacy = pd.read_parquet(PARENT_RUN / 'legacy_population.parquet')
    names = list(parent_report['feature_names'])
    grid = grid_population(legacy, full, names)
    with np.load(PARENT_RUN / 'features.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['feature_names'], names)
        np.testing.assert_array_equal(saved['timestamps'], pd.DatetimeIndex(evaluation.timestamp).asi8)
        features = {NEW_ARMS[0]: pd.DataFrame(saved['legacy'], columns=names),
            NEW_ARMS[1]: pd.DataFrame(saved['candidate'], columns=names)}
    np.testing.assert_array_equal(features[NEW_ARMS[1]],
        grid_population(evaluation, full, names)[names].to_numpy(float))
    np.testing.assert_array_equal(features[NEW_ARMS[0]],
        legacy.set_index('timestamp').loc[evaluation.timestamp, names].to_numpy(float))
    if any(np.isinf(frame.to_numpy(float)).any() for frame in features.values()):
        raise ValueError('Evaluation feature infinity')
    prior = pd.read_parquet(PARENT_RUN / 'predictions.parquet')
    bridge.check_equal_columns(evaluation, prior, list(evaluation))
    predictions = {arm: prior[arm].to_numpy(float).copy() for arm in bridge.ARMS}
    cutoffs = bridge.block_cutoffs(full)
    np.testing.assert_array_equal(evaluation.cutoff, evaluation.continuity_block.map(cutoffs))
    parent_members = pd.read_parquet(PARENT_RUN / 'training_members.parquet')
    parent_members = parent_members.loc[parent_members.arm.eq('legacy_control')]
    member_rows = {int(block): rows.population_row.tolist()
        for block, rows in parent_members.groupby('evaluation_block', sort=True)}
    if set(member_rows) != set(evaluation.continuity_block):
        raise ValueError('Parent mature-member blocks differ from fixed evaluation')
    counts = parent_report['counts']
    if (len(full), len(legacy), len(evaluation), evaluation.continuity_block.nunique()) != (
            counts['grid_rows'], counts['legacy_population'], counts['evaluation_rows'], counts['evaluation_blocks']):
        raise ValueError('Parent fixed cohort dimensions changed')
    return dict(full=full, evaluation=evaluation, feature_names=names,
        populations={NEW_ARMS[0]: legacy, NEW_ARMS[1]: grid}, features=features,
        inherited_predictions=predictions, parent_member_rows=member_rows)


def mature_members(population, cutoff, block, arm, model_id, expected_population_rows):
    mask = bridge.mature_mask(population, cutoff, block)
    rows = population.loc[mask]
    if rows.population_row.tolist() != list(expected_population_rows):
        raise ValueError('Parent mature training members differ')
    if rows.empty:
        raise ValueError('No mature training members')
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
    record = dict(model_id=model_id, arm=arm, continuity_block=int(block), cutoff=cutoff.isoformat(),
        model_path=path.relative_to(run).as_posix(), training_member_path='training_members.parquet',
        parent_member_model_id=f'legacy_control_b{block}', train_rows=len(train), eval_rows=len(eval_features),
        label_counts=bridge.label_counts(train), training_last_timestamp=train.timestamp.max().isoformat(),
        max_label_available_at=train.label_available_at.max().isoformat(),
        training_features_sha256=bridge.array_sha(features.to_numpy()),
        training_timestamps_sha256=hashlib.sha256(pd.DatetimeIndex(train.timestamp).asi8.astype('<i8').tobytes()).hexdigest(),
        feature_names=names, model_sha256=bridge.sha(path), configuration=bridge.MODEL_CONFIG,
        aft_parameters=bridge.AFT_PARAMETERS, actual_trained_booster_config=config,
        num_boosted_rounds=int(rounds), booster_feature_names=booster_features,
        right_censor_policy='global archive-end labels only; all mature rows retained')
    return prediction, members, record


def score_cohort(evaluation, predictions):
    return dict(arms={arm: bridge.score_arm(evaluation, predictions[arm]) for arm in ARMS},
        paired={f'{candidate}_vs_{baseline}': bridge.paired_error_summary(
            evaluation, predictions[baseline], predictions[candidate]) for baseline, candidate in PAIRS})


def main_gates(metrics):
    # The old helper consumes a named view; its constants and functions remain untouched.
    gate = bridge.year_gates(dict(arms=dict(archive=metrics['arms']['archive'],
        legacy_control=metrics['arms']['legacy_control'], interval_candidate=metrics['arms'][PRIMARY_CANDIDATE]),
        paired=dict(interval_candidate_vs_archive=metrics['paired'][f'{PRIMARY_CANDIDATE}_vs_archive'])))
    gate['checks']['coverage_100'] = all(metrics['arms'][arm]['prediction_coverage'] == 1. for arm in ARMS)
    gate['passed'] = all(gate['checks'].values())
    gate['failure_reasons'] = [name for name, passed in gate['checks'].items() if not passed]
    return gate


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
            record = dict(year=int(year), continuity_block=int(block), arm=arm, old_events=int(rows.event_key.nunique()))
            record.update({name: value for name, value in score.items() if name != 'subgroups'})
            for name, values in score['subgroups'].items():
                record.update({f'{name}_{field}': value for field, value in values.items()})
            records.append(record)
    return pd.DataFrame(records)


def cohort_counts(inputs, parent_report):
    return dict(**parent_report['counts'], new_training_population=len(inputs['populations'][NEW_ARMS[0]]),
        new_training_population_label_counts=bridge.label_counts(inputs['populations'][NEW_ARMS[0]]))


def execute(inputs, manifests, environment, parent_report, started):
    assert_locks(manifests)
    run = bridge.new_run_directory(HERE / 'runs')
    (run / 'models').mkdir()
    report = dict(schema_version=1, status='running', run_dir=str(run), years=list(bridge.YEARS), **manifests,
        configuration=bridge.MODEL_CONFIG, aft_parameters=bridge.AFT_PARAMETERS, feature_names=inputs['feature_names'],
        counts=cohort_counts(inputs, parent_report), primary_candidate=PRIMARY_CANDIDATE,
        diagnostic_only_arms=list(DIAGNOSTIC_ONLY_ARMS), new_refitted_arms=list(NEW_ARMS), inherited_arms=list(bridge.ARMS),
        historical_shadow_evaluation=True, blind_test=False, deployed=False, production_ready=False,
        scope='sequential conditional decomposition, not a full factorial or unique causal attribution',
        target=parent_report['target'], archive_is_primary_reference=True,
        uncertainty_scope=parent_report['uncertainty_scope'])
    bridge.write_json(run / 'report.json', report)
    bridge.event('run_created', run_dir=str(run))
    try:
        evaluation, names = inputs['evaluation'], inputs['feature_names']
        evaluation.to_parquet(run / 'evaluation.parquet', index=False)
        inputs['populations'][NEW_ARMS[0]].to_parquet(run / 'legacy_population.parquet', index=False)
        inputs['populations'][NEW_ARMS[1]].to_parquet(run / 'grid_population.parquet', index=False)
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
        pd.concat(member_tables, ignore_index=True).to_parquet(run / 'training_members.parquet', index=False)
        bridge.write_json(run / 'models.json', dict(schema_version=1, feature_names=names, models=models))
        row_scores(evaluation, predictions).to_parquet(run / 'predictions.parquet', index=False)
        process_scores(evaluation, predictions).to_parquet(run / 'process_scores.parquet', index=False)
        annual, gates = {}, {}
        for year in bridge.YEARS:
            mask = evaluation.year.eq(year).to_numpy()
            annual[str(year)] = score_cohort(evaluation.loc[mask], {arm: values[mask] for arm, values in predictions.items()})
            gates[str(year)] = main_gates(annual[str(year)])
        scores = dict(annual=annual, pooled=score_cohort(evaluation, predictions), gates=dict(annual=gates,
            passed=all(gates[str(year)]['passed'] for year in bridge.YEARS),
            failure_reasons=[f'{year}:{name}' for year in bridge.YEARS for name in gates[str(year)]['failure_reasons']]))
        bridge.write_json(run / 'scores.json', scores)
        assert_locks(manifests)
        report.update(status='pending_independent_verification', metrics=scores, fitted_models=len(models),
            protected_unchanged=True, parent_unchanged=True,
            feature_missingness={arm: dict(zip(names, np.isnan(inputs['features'][arm].to_numpy(float)).mean(axis=0).tolist()))
                for arm in NEW_ARMS},
            training_maturity_summary=[{name: model[name] for name in ('model_id', 'train_rows', 'label_counts',
                'training_last_timestamp', 'max_label_available_at')} for model in models],
            elapsed_seconds=time.perf_counter()-started,
            limits=['Known historical 2024/2025; not a blind test', 'labels_only is diagnostic and must never be deployed',
                'Maturity does not prove a training population can be reconstructed from each observation prefix',
                'Historical weather issue-time availability and upstream imputation remain unproven',
                'Population changes can affect histogram binning, column sampling and regularization',
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
