"""Fixed descriptive direction counts from already verified saved predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PARENT = ROOT / 'experiments/web_recovery_factors_20261002/runs/20261002T161429405383Z'
PYTHON = ROOT / '.venv_air_models/Scripts/python.exe'
PARENT_REPORT_SHA256 = '21aa39d5a08a683e23904000eeeccbb3c31fda30201cf30e64b4cf28ab063998'
PARENT_VERIFICATION_SHA256 = '17f3a9e71d15e95c3f9f02149a5bd1c1f27abe784682979c050263e44e0f8566'
PARENT_VERIFIER_SHA256 = '1c3052b2f51023d8e08649ed47dff614f05065ba1dd161d9964b8151400c6c6e'
ARMS = ('archive', 'interval_candidate', 'grid_all_legacy_origins')
BUCKETS = ('lt24', 'ge24')
PRIMARY_ARM = 'grid_all_legacy_origins'
RATE_COLUMNS = tuple(f'{weight}_{rate}' for weight in ('row', 'block')
    for rate in ('certain_early', 'possible_early', 'certain_late', 'possible_late'))
IDENTITY = ('evaluation_block', 'cutoff', 'evaluation_year', 'arm', 'bucket')
MEMBER_COLUMNS = ('origin_index', 'timestamp', 'year', 'continuity_block',
    'label_available_at', 'lower_bound_hours', 'upper_bound_hours', 'exact_hours', 'reason')
DIRECTIONS = ('insufficient', 'none', 'raise', 'lower')
CURRENT_DIRECTIONS = ('raise', 'lower', 'unknown', 'current_empty')
AGREEMENTS = ('consistent', 'opposite', 'unknown', 'current_empty',
    'no_signal', 'insufficient_history')
CONFIGURATION = dict(prediction_bucket_threshold_hours=24., min_past_blocks=6,
    certain_majority_threshold=.5, required_majority_block_fraction=2/3,
    strict_timestamp_before_cutoff=True, strict_label_available_before_cutoff=True,
    exclude_current_continuity_block=True, past_years=[2024, 2025])
SCOPE = 'Fixed saved 2024/2025 predictions; descriptive prefix direction screening only'
LIMITS = [
    'Direction rates are observation-compatible bounds, not confidence intervals or p-values',
    'No fitting, new inference, calibration, model selection or accuracy improvement is performed',
    'Only saved verified 2024/2025 predictions are prefix history; earliest cutoffs lack support',
    'Current diagnostics are retrospective and never feed the same cutoff prefix signal',
    'Known historical years are not a blind test or a real-time availability demonstration',
    'A descriptive direction signal neither authorizes training nor proves lower hourly MAE',
    'No production deployment or webpage modification is authorized',
]
PROTECTED_DIRECTORIES = tuple(ROOT / name for name in (
    'artifacts/cnemc_2025_deployment_candidate', 'artifacts/strict_history_backtest',
    'project-showcase', 'sites-haze/dist', 'releases/runtime_20261002'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def merge_hashes(*maps):
    merged = {}
    for mapping in maps:
        for path, digest in mapping.items():
            if path in merged and merged[path] != digest:
                raise ValueError(f'Conflicting frozen digest: {path}')
            merged[path] = digest
    return merged


def file_inventory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'Missing artifact directory: {directory}')
    return sorted(path.relative_to(directory).as_posix()
        for path in directory.rglob('*') if path.is_file())


def parent_artifact_manifest(parent, report):
    parent = Path(parent).resolve()
    expected = sorted(set(report['output_sha256']) | {'report.json', 'verification.json'})
    if file_inventory(parent) != expected:
        raise ValueError('Parent artifact inventory differs from its passed chain')
    locked = {}
    for name in expected:
        path = (parent / name).resolve()
        if not path.is_relative_to(parent) or not path.is_file():
            raise ValueError(f'Invalid parent artifact path: {name}')
        digest = sha(path)
        if name in report['output_sha256'] and digest != report['output_sha256'][name]:
            raise ValueError(f'Parent output digest changed: {name}')
        locked[str(path)] = digest
    return locked, expected


def assert_locks(manifests):
    for path, digest in manifests['frozen_sha256'].items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError(f'Frozen file missing or changed: {path}')
    if file_inventory(manifests['parent_run_dir']) != manifests['parent_artifact_inventory']:
        raise ValueError('Parent artifact inventory changed')
    if 'protected_sha256' in manifests:
        assert_directory_inventory(PROTECTED_DIRECTORIES, manifests['protected_sha256'])


def assert_directory_inventory(directories, locked):
    for directory in directories:
        directory = Path(directory).resolve()
        actual = file_inventory(directory)
        expected = sorted(Path(path).resolve().relative_to(directory).as_posix()
            for path in locked if Path(path).resolve().is_relative_to(directory))
        if not actual or actual != expected:
            raise ValueError(f'Protected directory is empty or its inventory changed: {directory}')


def validate_parent(report, verification, report_digest):
    if report_digest != PARENT_REPORT_SHA256 or report.get('status') != 'pending_independent_verification':
        raise ValueError('Parent main report is not the exact fixed report')
    if (verification.get('status') != 'passed'
            or verification.get('independent_verification_passed') is not True
            or verification.get('report_sha256') != PARENT_REPORT_SHA256
            or verification.get('verifier_sha256') != PARENT_VERIFIER_SHA256):
        raise ValueError('Parent independent passed chain does not match')


def check_environment(environment):
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError('The fixed research interpreter is required')
    if environment['python_executable'] != str(PYTHON) or environment['python_version'] != sys.version:
        raise ValueError('Python environment differs from the passed parent')
    actual = {name: importlib.metadata.version(name) for name in environment['packages']}
    if actual != environment['packages']:
        raise ValueError('Package metadata differs from the passed parent')


def provenance():
    report_path = PARENT / 'report.json'
    verification_path = PARENT / 'verification.json'
    if sha(verification_path) != PARENT_VERIFICATION_SHA256:
        raise ValueError('Parent verification bytes changed')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    verification = json.loads(verification_path.read_text(encoding='utf-8'))
    validate_parent(report, verification, sha(report_path))
    parent_hashes, inventory = parent_artifact_manifest(PARENT, report)
    local_code = {str(HERE / name): sha(HERE / name) for name in
        ('PROTOCOL.md', 'direction_audit.py', 'test_direction_audit.py',
         'verify_direction.py', 'test_verify_direction.py')}
    manifests = dict(input_sha256=report['input_sha256'],
        protected_sha256=report['protected_sha256'],
        original_scope_sha256=report['frozen_sha256'],
        environment_sha256=report['environment_sha256'],
        code_sha256=merge_hashes(report['code_sha256'], local_code),
        parent_sha256=parent_hashes, parent_run_dir=str(PARENT),
        parent_artifact_inventory=inventory, parent_report_sha256=PARENT_REPORT_SHA256,
        parent_verification_sha256=PARENT_VERIFICATION_SHA256,
        parent_verification_report_sha256=verification['report_sha256'],
        parent_verifier_sha256=PARENT_VERIFIER_SHA256,
        historical_input_sha256=report['historical_input_sha256'],
        historical_audit_drift=report['historical_audit_drift'])
    manifests['frozen_sha256'] = merge_hashes(*(manifests[name] for name in
        ('input_sha256', 'protected_sha256', 'original_scope_sha256',
         'environment_sha256', 'code_sha256', 'parent_sha256')))
    assert_locks(manifests)
    environment = json.loads((PARENT / 'environment.json').read_text(encoding='utf-8'))
    check_environment(environment)
    return manifests, report, environment


def new_run_directory(parent=None, stamp=None):
    parent = Path(parent) if parent is not None else HERE / 'runs'
    stamp = stamp or datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('Run timestamp must be timezone aware')
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / stamp.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(exist_ok=False)
    return directory


def rates(rows, prediction):
    prediction = np.asarray(prediction, dtype=float)
    lower = rows.lower_bound_hours.to_numpy(dtype=float)
    upper = rows.upper_bound_hours.to_numpy(dtype=float)
    if (prediction.shape != lower.shape or not np.isfinite(prediction).all()
            or (prediction < 0).any() or not np.isfinite(lower).all() or (lower < 0).any()
            or np.isnan(upper).any() or (upper < lower).any()
            or rows.continuity_block.isna().any()):
        raise ValueError('Invalid predictions, label bounds or continuity block')
    result = dict(past_rows=len(rows), past_blocks=int(rows.continuity_block.nunique()))
    if not len(rows):
        result.update({name: None for name in RATE_COLUMNS})
        result.update(certain_early_majority_blocks=0, certain_late_majority_blocks=0,
            certain_early_majority_block_fraction=None, certain_late_majority_block_fraction=None)
        return result
    indicators = pd.DataFrame(dict(certain_early=prediction < lower,
        possible_early=prediction < upper, certain_late=prediction > upper,
        possible_late=prediction > lower))
    block_rates = indicators.groupby(rows.continuity_block.to_numpy(), sort=True).mean()
    for rate in indicators:
        result[f'row_{rate}'] = float(indicators[rate].mean())
        result[f'block_{rate}'] = float(block_rates[rate].mean())
    for direction in ('early', 'late'):
        count = int((block_rates[f'certain_{direction}'] > .5).sum())
        result[f'certain_{direction}_majority_blocks'] = count
        result[f'certain_{direction}_majority_block_fraction'] = count / result['past_blocks']
    return result


def screen(stats):
    blocks = stats['past_blocks']
    if blocks < 6:
        return 'insufficient'
    for label, direction in (('early', 'raise'), ('late', 'lower')):
        if (stats[f'block_certain_{label}'] > .5
                and 3 * stats[f'certain_{label}_majority_blocks'] >= 2 * blocks):
            return direction
    return 'none'


def bucket_mask(prediction, bucket):
    if bucket not in BUCKETS:
        raise ValueError(f'Unknown prediction bucket: {bucket}')
    return prediction.lt(24.) if bucket == 'lt24' else prediction.ge(24.)


def prefix_mask(origins, cutoff, block, arm, bucket):
    return (origins.timestamp.lt(cutoff) & origins.label_available_at.lt(cutoff)
        & origins.continuity_block.ne(block) & bucket_mask(origins[arm], bucket))


def validate_origins(origins):
    required = set(MEMBER_COLUMNS) | {'cutoff'} | set(ARMS)
    if not required.issubset(origins.columns):
        raise ValueError(f'Missing origin columns: {sorted(required - set(origins.columns))}')
    if origins.origin_index.isna().any() or origins.origin_index.duplicated().any() or origins.timestamp.duplicated().any():
        raise ValueError('Origin identities must be complete and unique')
    for name in ('timestamp', 'label_available_at', 'cutoff'):
        if not isinstance(origins[name].dtype, pd.DatetimeTZDtype) or origins[name].isna().any():
            raise ValueError(f'Complete timezone-aware values are required: {name}')
    if origins.year.isna().any() or origins.reason.isna().any():
        raise ValueError('Missing origin year or label reason')
    for arm in ARMS:
        rates(origins, origins[arm])


def build_signals(origins, cutoffs):
    validate_origins(origins)
    if (cutoffs.evaluation_block.isna().any() or cutoffs.evaluation_block.duplicated().any()
            or cutoffs.cutoff.isna().any() or cutoffs.evaluation_year.isna().any()
            or not isinstance(cutoffs.cutoff.dtype, pd.DatetimeTZDtype)):
        raise ValueError('Cutoff identities must be unique and timezone aware')
    records, members = [], []
    for cutoff in cutoffs.sort_values(['cutoff', 'evaluation_block']).itertuples(index=False):
        for arm in ARMS:
            for bucket in BUCKETS:
                identity = dict(evaluation_block=int(cutoff.evaluation_block), cutoff=cutoff.cutoff,
                    evaluation_year=int(cutoff.evaluation_year), arm=arm, bucket=bucket)
                selected = origins.loc[prefix_mask(origins, cutoff.cutoff, cutoff.evaluation_block, arm, bucket)]
                stats = rates(selected, selected[arm])
                records.append(dict(identity, **stats, support_sufficient=stats['past_blocks'] >= 6,
                    direction=screen(stats)))
                member = selected.loc[:, list(MEMBER_COLUMNS)].copy()
                member['prediction'] = selected[arm].to_numpy()
                for key, value in identity.items():
                    member[key] = value
                members.append(member)
    signals = pd.DataFrame(records)
    prefix_members = pd.concat(members, ignore_index=True) if members else pd.DataFrame(
        columns=list(MEMBER_COLUMNS) + ['prediction'] + list(IDENTITY))
    return signals, prefix_members


def current_diagnostics(origins, signals):
    validate_origins(origins)
    records = []
    for signal in signals.to_dict('records'):
        identity = {key: signal[key] for key in IDENTITY}
        selected = origins.loc[origins.continuity_block.eq(signal['evaluation_block'])
            & bucket_mask(origins[signal['arm']], signal['bucket'])]
        stats = rates(selected, selected[signal['arm']])
        if not stats['past_rows']:
            direction = 'current_empty'
        elif stats['row_certain_early'] > .5:
            direction = 'raise'
        elif stats['row_certain_late'] > .5:
            direction = 'lower'
        else:
            direction = 'unknown'
        if direction == 'current_empty':
            agreement = 'current_empty'
        elif signal['direction'] == 'insufficient':
            agreement = 'insufficient_history'
        elif signal['direction'] == 'none':
            agreement = 'no_signal'
        elif direction == 'unknown':
            agreement = 'unknown'
        else:
            agreement = 'consistent' if direction == signal['direction'] else 'opposite'
        records.append(dict(identity, current_rows=stats['past_rows'], current_blocks=stats['past_blocks'],
            **{key: stats[key] for key in RATE_COLUMNS}, current_direction=direction, agreement=agreement))
    return pd.DataFrame(records)


def summary_leaf(signals, current):
    if len(signals) != len(current) or not signals.loc[:, list(IDENTITY)].reset_index(drop=True).equals(
            current.loc[:, list(IDENTITY)].reset_index(drop=True)):
        raise ValueError('Summary signal and diagnostic identities differ')
    n = len(signals)
    direction = {key: int(signals.direction.eq(key).sum()) for key in DIRECTIONS}
    current_direction = {key: int(current.current_direction.eq(key).sum()) for key in CURRENT_DIRECTIONS}
    agreement = {key: int(current.agreement.eq(key).sum()) for key in AGREEMENTS}
    if sum(direction.values()) != n or sum(current_direction.values()) != n or sum(agreement.values()) != n:
        raise ValueError('Unknown direction or agreement category')
    signal_n = direction['raise'] + direction['lower']
    comparable = agreement['consistent'] + agreement['opposite']
    fraction = lambda count: count / n if n else None
    return dict(combinations=n, direction_counts=direction, signal_combinations=signal_n,
        signal_coverage=fraction(signal_n), support_sufficient_combinations=int(signals.support_sufficient.sum()),
        empty_prefix_combinations=int(signals.past_rows.eq(0).sum()),
        insufficient_history_combinations=direction['insufficient'], no_signal_combinations=direction['none'],
        current_empty_combinations=current_direction['current_empty'], current_direction_counts=current_direction,
        agreement_counts=agreement, comparable_combinations=comparable, comparable_coverage=fraction(comparable),
        consistent_fraction_all=fraction(agreement['consistent']), opposite_fraction_all=fraction(agreement['opposite']),
        unknown_fraction_all=fraction(agreement['unknown']),
        consistent_fraction_comparable=agreement['consistent'] / comparable if comparable else None)


def summarize(signals, current):
    def leaf(mask):
        return summary_leaf(signals.loc[mask], current.loc[mask])
    return dict(all=summary_leaf(signals, current),
        by_arm={arm: leaf(signals.arm.eq(arm)) for arm in ARMS},
        by_arm_bucket={arm: {bucket: leaf(signals.arm.eq(arm) & signals.bucket.eq(bucket))
            for bucket in BUCKETS} for arm in ARMS},
        by_year={str(year): leaf(signals.evaluation_year.eq(year)) for year in (2024, 2025)},
        primary=leaf(signals.arm.eq(PRIMARY_ARM)))


def prepare_inputs(parent_report):
    evaluation = pd.read_parquet(PARENT / 'evaluation.parquet')
    predictions = pd.read_parquet(PARENT / 'predictions.parquet')
    if len(evaluation) != 701 or not np.array_equal(evaluation.eval_row, np.arange(701)):
        raise ValueError('The fixed parent 701-row evaluation identity differs')
    pd.testing.assert_frame_equal(evaluation, predictions.loc[:, evaluation.columns], check_exact=True)
    origins = predictions.loc[:, ['eval_row', 'timestamp', 'year', 'continuity_block', 'cutoff',
        'label_available_at', 'lower_bound_hours', 'upper_bound_hours', 'exact_hours', 'reason'] + list(ARMS)].copy()
    origins.rename(columns={'eval_row': 'origin_index'}, inplace=True)
    validate_origins(origins)
    if not np.array_equal(origins.timestamp.dt.tz_convert('Asia/Shanghai').dt.year, origins.year):
        raise ValueError('Parent issue years differ from local timestamps')
    grouped = origins.groupby('continuity_block', sort=True)
    if grouped.cutoff.nunique().ne(1).any() or grouped.year.nunique().ne(1).any():
        raise ValueError('An evaluation block has inconsistent cutoff or issue year')
    cutoffs = grouped.agg(cutoff=('cutoff', 'first'), evaluation_year=('year', 'first')).reset_index()
    cutoffs.rename(columns={'continuity_block': 'evaluation_block'}, inplace=True)
    if len(cutoffs) != 32 or origins.timestamp.lt(origins.cutoff).any():
        raise ValueError('Fixed 32 evaluation cutoffs differ')
    annual_origins = {str(year): int(origins.year.eq(year).sum()) for year in (2024, 2025)}
    annual_blocks = {str(year): int(cutoffs.evaluation_year.eq(year).sum()) for year in (2024, 2025)}
    lower, upper = origins.lower_bound_hours.to_numpy(), origins.upper_bound_hours.to_numpy()
    labels = dict(exact=int((lower == upper).sum()),
        finite_interval=int((np.isfinite(upper) & (upper > lower)).sum()), right_censored=int(np.isposinf(upper).sum()))
    if (annual_origins != {'2024': 383, '2025': 318} or annual_blocks != {'2024': 15, '2025': 17}
            or labels != {'exact': 674, 'finite_interval': 27, 'right_censored': 0}
            or parent_report['counts']['evaluation_rows'] != len(origins)
            or parent_report['counts']['evaluation_blocks'] != len(cutoffs)):
        raise ValueError('Parent cohort or label counts differ from the fixed protocol')
    counts = dict(origin_rows=len(origins), evaluation_blocks=len(cutoffs), arms=len(ARMS), buckets=len(BUCKETS),
        combinations_expected=len(cutoffs) * len(ARMS) * len(BUCKETS), annual_origins=annual_origins,
        annual_blocks=annual_blocks, label_counts=labels)
    return origins, cutoffs, counts


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as target:
        json.dump(value, target, indent=2, ensure_ascii=True, allow_nan=False)
        target.write('\n')


def execute(manifests, parent_report, environment, origins, cutoffs, counts):
    started = time.perf_counter()
    assert_locks(manifests)
    check_environment(environment)
    run = new_run_directory()
    report = dict(schema_version=1, status='execution_failed', run_dir=str(run), years=[2024, 2025],
        **manifests, arms=list(ARMS), buckets=list(BUCKETS), primary_arm=PRIMARY_ARM,
        configuration=CONFIGURATION, scope=SCOPE, limits=LIMITS, fit_performed=False,
        predict_performed=False, calibration_performed=False, deployed=False, production_ready=False,
        descriptive_screen_only=True, historical_shadow_evaluation=True, blind_test=False)
    try:
        shutil.copyfile(HERE / 'EXECUTION_RECORD.md', run / 'execution_record_at_start.md')
        write_json(run / 'environment.json', environment)
        signals, members = build_signals(origins, cutoffs)
        signals.to_parquet(run / 'signals.parquet', index=False)
        members.to_parquet(run / 'prefix_members.parquet', index=False)
        current = current_diagnostics(origins, signals)
        current.to_parquet(run / 'current_diagnostics.parquet', index=False)
        counts = dict(counts, signal_rows=len(signals), current_diagnostic_rows=len(current), prefix_member_rows=len(members))
        if len(signals) != counts['combinations_expected'] or len(current) != len(signals):
            raise ValueError('Full combination coverage differs from the protocol')
        assert_locks(manifests)
        check_environment(environment)
        report.update(status='pending_independent_verification', counts=counts,
            summary=summarize(signals, current), protected_unchanged=True, parent_unchanged=True,
            elapsed_seconds=time.perf_counter() - started)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error),
            protected_unchanged=False, parent_unchanged=False)
        raise
    finally:
        report['output_sha256'] = {name: sha(run / name) for name in file_inventory(run)}
        write_json(run / 'report.json', report)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    manifests, parent_report, environment = provenance()
    origins, cutoffs, counts = prepare_inputs(parent_report)
    assert_locks(manifests)
    if args.preflight_only:
        print(json.dumps(dict(status='preflight_passed', counts=counts,
            frozen_files=len(manifests['frozen_sha256']), parent_files=len(manifests['parent_sha256']),
            fit_performed=False, predict_performed=False), ensure_ascii=True))
        return 0
    run = execute(manifests, parent_report, environment, origins, cutoffs, counts)
    print(json.dumps(dict(status='pending_independent_verification', run_dir=str(run)), ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
