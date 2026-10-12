"""Fixed f072 strategy comparison; fail closed before creating any output."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha
from scripts.build_gefs_72h_features import FIELDS, require, validated_snapshots

AGES = ['forecast_known_age_hours', 'forecast_run_age_hours', 'forecast_remaining_lead_hours']
MATCHING_CHECKS = {'origin_order', 'latest_eligible_selection', 'all_fields_and_ages',
                   'missing_rows_retained', 'source_hashes'}


def required_audit_paths(root):
    base = root / 'artifacts/task01_recovery_head'
    folder = base / 'gefs_72h_features_v1'
    return [folder / name for name in ('features.parquet', 'snapshots.parquet', 'report.json')] + [
        base / 'interval_business_labels_v1/origins.parquet',
        root / 'scripts/audit_gefs_72h_matching.py']


def verify_hashes(mapping, required=()):
    paths = {Path(path): value for path, value in mapping.items()}
    require(set(required) <= set(paths), 'required input hashes missing')
    require(bool(paths), 'empty hash inventory')
    for path, value in paths.items():
        require(path.is_absolute(), 'hash paths must be absolute')
        require(sha(path) == value, f'stale or changed input: {path}')
    return {str(path): value for path, value in paths.items()}


def require_matching_audit(root):
    path = root / 'artifacts/task01_recovery_head/gefs_72h_features_v1/matching_audit.json'
    audit = json.loads(path.read_bytes())
    require(audit.get('passed') is True, 'matching audit did not pass')
    require(audit.get('origin_rows') == 7890 and audit.get('snapshot_rows') == 292,
            'matching audit population incomplete')
    require(MATCHING_CHECKS <= set(audit.get('checks', [])), 'matching audit checks incomplete')
    inputs = verify_hashes(audit.get('input_sha256', {}), required_audit_paths(root))
    inputs[str(path)] = sha(path)
    return inputs


def attach_forecast(frame, forecast, extra):
    require(not frame.timestamp.duplicated().any(), 'duplicate origin timestamps')
    require(not forecast.timestamp.duplicated().any(), 'duplicate forecast timestamps')
    indexed = forecast.set_index('timestamp')
    require(frame.timestamp.isin(indexed.index).all(), 'missing forecast origin rows')
    result = frame.copy()
    result[extra] = indexed.loc[frame.timestamp, extra].to_numpy()
    pd.testing.assert_series_equal(result.timestamp, frame.timestamp)
    return result


def compare_cohort(frame, prediction):
    prediction = np.asarray(prediction, dtype=float)
    require(prediction.shape == (len(frame),) and np.isfinite(prediction).all(),
            'prediction must be finite and aligned')
    require(frame.score_allowed.dtype == bool and frame.score_allowed.any(),
            'invalid score_allowed mask')
    require(np.isfinite(frame[['forecast_aft', 'history_counts_aft']].to_numpy(dtype=float)).all(),
            'control prediction must be finite on every origin')
    scored = frame.loc[frame.score_allowed]
    candidate = metrics(scored, prediction[frame.score_allowed])
    control = metrics(scored, scored.forecast_aft.to_numpy())
    reference = metrics(scored, scored.history_counts_aft.to_numpy())
    keys = ['row_mae', 'block_mae', 'tail_24h_block_mae']
    require(all(np.isfinite(values[key]) for values in (candidate, control, reference)
                for key in keys), 'unscorable development cohort')
    return {'candidate': candidate, 'f024_control': control, 'history_reference': reference,
            'passed': bool(candidate['row_mae'] < control['row_mae']
                           and candidate['block_mae'] < control['block_mae']
                           and candidate['tail_24h_block_mae'] <= control['tail_24h_block_mae'])}


def verified_artifact(folder, inputs, with_protocol=True):
    report_path = folder / 'report.json'
    report = json.loads(report_path.read_bytes())
    inputs[str(report_path)] = sha(report_path)
    outputs = report['output_sha256']
    required = {'features.parquet', 'snapshots.parquet'} if not with_protocol else {
        'protocol.json', 'model.joblib', 'predictions.parquet', 'expanded_predictions.parquet'}
    require(required <= set(outputs), 'incomplete artifact outputs')
    require(all(Path(name).name == name for name in outputs), 'invalid output filename')
    inputs.update(verify_hashes({str(folder / name): value for name, value in outputs.items()}))
    if with_protocol:
        protocol = json.loads((folder / 'protocol.json').read_bytes())
        inputs.update(verify_hashes(protocol['input_sha256']))
        return protocol
    inputs.update(verify_hashes(report['input_sha256']))
    return report


def prepare(root=ROOT):
    inputs = require_matching_audit(root)
    base = root / 'artifacts/task01_recovery_head'
    snapshots, raw_inputs = validated_snapshots(root)
    inputs.update(raw_inputs)
    folder = base / 'gefs_72h_features_v1'
    report = verified_artifact(folder, inputs, with_protocol=False)
    require(report['fields'] == FIELDS and report['origins'] == 7890
            and report['snapshots'] == 292, 'unexpected feature construction population')
    pd.testing.assert_frame_equal(snapshots, pd.read_parquet(folder / 'snapshots.parquet'))
    forecasts = pd.read_parquet(folder / 'features.parquet')
    origins = pd.read_parquet(base / 'interval_business_labels_v1/origins.parquet')
    pd.testing.assert_series_equal(forecasts.timestamp, origins.timestamp)
    old = base / 'reusable_gefs_aft_v1'
    dev = base / 'interval_history_counts_v1'
    old_protocol = verified_artifact(old, inputs)
    history_protocol = verified_artifact(dev, inputs)
    history_names = history_protocol['features']
    old_names = old_protocol['features']
    require(len(history_names) == 92 and len(old_names) == 104
            and old_names[:92] == history_names
            and old_names[92:] == [name.replace('_f072', '_f024') for name in FIELDS] + AGES,
            'baseline feature schema changed')
    require(old_protocol['seed'] == history_protocol['seed'] == 42
            and old_protocol['rounds'] == history_protocol['rounds'] == 500,
            'baseline model settings changed')
    # Reconstruct the old training matrix from exactly its hash-verified sources.
    train_path = base / 'interval_business_aft_v1/training_rows.parquet'
    history_path = base / 'trajectory_reset_audit_v1/origins.parquet'
    old_forecast_path = base / 'reusable_gefs_snapshots_v1/features.parquet'
    verify_hashes(old_protocol['input_sha256'], [train_path, history_path, old_forecast_path,
                                               dev / 'predictions.parquet'])
    train = pd.read_parquet(train_path)
    history = pd.read_parquet(history_path)
    train = attach_forecast(train, history, history_names[87:])
    old_train = attach_forecast(train, pd.read_parquet(old_forecast_path), old_names[92:])
    val = pd.read_parquet(old / 'predictions.parquet')
    expanded = pd.read_parquet(old / 'expanded_predictions.parquet')
    history_val = pd.read_parquet(dev / 'predictions.parquet')
    history_expanded = pd.read_parquet(dev / 'expanded_predictions.parquet')
    pd.testing.assert_frame_equal(val[history_val.columns], history_val)
    pd.testing.assert_frame_equal(expanded[history_expanded.columns], history_expanded)
    pd.testing.assert_series_equal(val.timestamp, expanded.timestamp)
    require(len(train) == old_protocol['training_rows'] == 6770 and len(val) == 519,
            'fixed training or validation population changed')
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    require(train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all(),
            'training crosses cutoff')
    require(not set(train.continuity_block) & set(val.continuity_block), 'training block leakage')
    candidate_train = attach_forecast(train, forecasts, FIELDS + AGES)
    candidate_val = attach_forecast(history_val, forecasts, FIELDS + AGES)
    pd.testing.assert_frame_equal(candidate_train[history_names], old_train[history_names])
    pd.testing.assert_frame_equal(candidate_val[history_names], val[history_names])
    # Replay rather than retrain the frozen controls before fitting any candidate.
    np.testing.assert_array_equal(joblib.load(old / 'model.joblib').predict(val[old_names]),
                                  val.forecast_aft.to_numpy())
    np.testing.assert_array_equal(joblib.load(dev / 'model.joblib').predict(val[history_names]),
                                  val.history_counts_aft.to_numpy())
    for path in [Path(__file__), root / 'scripts/benchmark_observed_business_aft.py',
                 root / 'scripts/build_gefs_72h_features.py',
                 root / 'gefs_72h_comparison_protocol_20260906.md']:
        inputs[str(path)] = sha(path)
    verify_hashes(inputs)
    return inputs, history_names + FIELDS + AGES, old_names, candidate_train, candidate_val, old_train, val, expanded


def main(root=ROOT):
    out = root / 'artifacts/task01_recovery_head/gefs_72h_aft_v1'
    if out.exists():
        raise FileExistsError(out)
    inputs, names, old_names, train, inference, old_train, original, expanded = prepare(root)
    protocol = {'input_sha256': inputs, 'features': names, 'f024_features': old_names,
                'seed': 42, 'rounds': 500, 'training_rows': len(train),
                'validation_rows': len(inference), 'train_cutoff': '2023-01-01 Asia/Shanghai',
                'training_forecast_rows': int(train[FIELDS[0]].notna().sum()),
                'validation_forecast_rows': int(inference[FIELDS[0]].notna().sum()),
                'training_matrix_evidence': 'Reconstructed from frozen f024 sources and algorithm; first 92 columns identical.',
                'gate': 'Both cohorts: row/block MAE below f024, tail block MAE no worse.',
                'limitations': ['Reused 2023 development year; no deployment authorization.',
                                'Strategy replacement changes lead, availability, age and coverage.',
                                'Archive Last-Modified is not first-publication proof.',
                                'Missing forecasts retained; no imputation.']}
    out.mkdir(parents=True)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train.to_parquet(out / 'training_rows.parquet', index=False)
    inference.to_parquet(out / 'inference_rows.parquet', index=False)
    old_train[['timestamp'] + old_names].to_parquet(out / 'f024_training_inputs.parquet', index=False)
    original[['timestamp'] + old_names].to_parquet(out / 'f024_inference_inputs.parquet', index=False)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
        train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    prediction = model.predict(inference[names])
    np.testing.assert_array_equal(joblib.load(out / 'model.joblib').predict(inference[names]), prediction)
    results = {}
    for name, frame in [('original', original), ('expanded', expanded)]:
        results[name] = compare_cohort(frame, prediction)
        frame['f072_aft'] = prediction
        filename = 'predictions.parquet' if name == 'original' else 'expanded_predictions.parquet'
        frame.to_parquet(out / filename, index=False)
    verify_hashes(inputs)
    report = {'cohorts': results, 'passed': all(item['passed'] for item in results.values()),
              'inputs_unchanged': True, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
