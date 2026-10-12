"""Fill only frozen 2021 training forecast gaps after complete independent audit."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.forecast_snapshot_reuse import reuse_forecast_snapshots
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.audit_gefs_72h_matching import verify_matches
from scripts.audit_gefs_2021_f024 import FIELDS as PRODUCTS
from scripts.audit_gefs_2021_f024 import verify_protocol
from scripts.benchmark_gefs_72h_aft import AGES, compare_cohort, prepare, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha
from scripts.build_gefs_72h_features import require
from scripts.collect_gefs_2021_f024 import verify_complete

PREFIXES = {'geavg': 'mean', 'gespr': 'spread', 'gec00': 'control'}
FIELDS = [f'gefs_{PREFIXES[p]}_{field}_f024' for p, fields in PRODUCTS.items() for field in fields]
FORECAST = FIELDS + AGES


def validate_audit(audit, runs, protocol_sha, script_sha):
    require(audit.get('verified_runs') == audit.get('planned_runs') == len(runs) == 101
            and audit.get('verified_grib_messages') == 909
            and audit.get('all_planned_runs_verified') is True, 'audit population incomplete')
    require(audit.get('protocol_sha256') == protocol_sha
            and audit.get('audit_script_sha256') == script_sha, 'stale audit identity')
    records = audit.get('verified', [])
    require(len(records) == 101 and sorted(r['run_at'] for r in records) == sorted(runs),
            'audit run population mismatch')
    return {r['run_at']: r['complete_sha256'] for r in records}


def load_snapshots(root=ROOT):
    base = root / 'artifacts/task01_recovery_head/gefs_2021_f024_collection_v1'
    source = root / 'artifacts/task01_recovery_head/interval_business_labels_v1/origins.parquet'
    binary = root / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    auditor = root / 'scripts/audit_gefs_2021_f024.py'
    protocol_path, audit_path = base / 'protocol.json', base / 'independent_audit.json'
    paths = [protocol_path, audit_path, source, binary, auditor, Path(__file__),
             root / 'scripts/collect_gefs_2021_f024.py', root / 'scripts/audit_gefs_72h_matching.py',
             root / 'src/haze_forecast/forecast_snapshot_reuse.py',
             root / 'gefs_2021_training_extension_protocol_20260907.md']
    inputs = {str(path): sha(path) for path in paths}
    runs = verify_protocol(root, base, binary, source)
    protocol = json.loads(protocol_path.read_bytes())
    for name, value in protocol['helper_sha256'].items():
        inputs[str(root / name)] = value
    audited = validate_audit(json.loads(audit_path.read_bytes()), runs,
                             inputs[str(protocol_path)], inputs[str(auditor)])
    require(len(list(base.glob('*/complete.json'))) == 101, 'completion population mismatch')
    rows = []
    for run_text in runs:
        run = pd.Timestamp(run_text)
        folder = base / run.strftime('%Y%m%d%H')
        marker = folder / 'complete.json'
        require(sha(marker) == audited[run_text], 'stale audited completion marker')
        record = verify_complete(folder, run)
        inputs[str(marker)] = audited[run_text]
        inputs.update({str(folder / name): value for name, value in record['output_sha256'].items()})
        row = {'run_at': run, 'valid_at': pd.Timestamp(record['valid_at']),
               'known_at': pd.Timestamp(record['archive_available_at_proxy'])}
        for product, fields in PRODUCTS.items():
            for field in fields:
                row[f'gefs_{PREFIXES[product]}_{field}_f024'] = record['products'][product]['fields'][field]['value']
        rows.append(row)
    snapshots = pd.DataFrame(rows).sort_values('known_at').reset_index(drop=True)
    verify_hashes(inputs)
    return snapshots, inputs


def match_2021(snapshots, origins):
    features = reuse_forecast_snapshots(snapshots, origins, FIELDS)
    verify_matches(snapshots, features, origins, FIELDS)
    return features


def extend_training(train, features, expected_rows=498):
    require(train.timestamp.is_unique and features.timestamp.is_unique, 'duplicate timestamp')
    mask = train.timestamp.dt.tz_convert('Asia/Shanghai').dt.year.eq(2021)
    require(int(mask.sum()) == len(features) == expected_rows, '2021 population mismatch')
    require(set(train.loc[mask, 'timestamp']) == set(features.timestamp), '2021 timestamps mismatch')
    require(train.loc[mask, FORECAST].isna().all().all(), 'original 2021 forecast is not empty')
    result = train.copy(deep=True)
    result.loc[mask, FORECAST] = features.set_index('timestamp').loc[train.loc[mask, 'timestamp'], FORECAST].to_numpy()
    pd.testing.assert_frame_equal(result.loc[~mask], train.loc[~mask])
    pd.testing.assert_frame_equal(result.drop(columns=FORECAST), train.drop(columns=FORECAST))
    return result


def frozen_inference(original, expanded, names):
    pd.testing.assert_series_equal(original.timestamp, expanded.timestamp)
    shared = [name for name in names if name in expanded]
    pd.testing.assert_frame_equal(original[shared], expanded[shared])
    return original[['timestamp'] + names].copy(deep=True)


def main(root=ROOT):
    out = root / 'artifacts/task01_recovery_head/gefs_2021_extension_v1'
    if out.exists():
        raise FileExistsError(out)
    snapshots, raw_inputs = load_snapshots(root)
    inputs, _, names, _, _, old_train, original, expanded = prepare(root)
    inputs.update(raw_inputs)
    require(len(names) == 104 and names[92:] == FORECAST, 'frozen feature schema changed')
    source = root / 'artifacts/task01_recovery_head/interval_business_labels_v1/origins.parquet'
    origins = pd.read_parquet(source).timestamp
    origins = origins.loc[origins.dt.tz_convert('Asia/Shanghai').dt.year.eq(2021)]
    require(len(origins) == 498, 'source origin population mismatch')
    features = match_2021(snapshots, origins)
    train = extend_training(old_train, features)
    inference = frozen_inference(original, expanded, names)
    pd.testing.assert_frame_equal(inference[names], original[names])
    inputs[str(root / 'src/haze_forecast/observed_aft.py')] = sha(root / 'src/haze_forecast/observed_aft.py')
    verify_hashes(inputs)
    covered = train[FIELDS[0]].notna()
    mask = train.timestamp.dt.tz_convert('Asia/Shanghai').dt.year.eq(2021)
    coverage = {'original_forecast_rows': int(old_train[FIELDS[0]].notna().sum()),
                'extended_forecast_rows': int(covered.sum()),
                '2021_origin_rows': int(mask.sum()), '2021_forecast_rows': int((covered & mask).sum()),
                '2021_forecast_blocks': int(train.loc[covered & mask, 'continuity_block'].nunique()),
                'all_forecast_blocks': int(train.loc[covered, 'continuity_block'].nunique())}
    protocol = {'input_sha256': inputs, 'features': names, 'seed': 42, 'rounds': 500,
                'training_rows': len(train), 'validation_rows': len(inference), 'coverage': coverage,
                'train_cutoff': '2023-01-01 Asia/Shanghai', 'sample_weight': None,
                'change': 'Only twelve previously empty forecast inputs on original 2021 training rows.',
                'gate': 'Both original and expanded cohorts: row/block MAE lower, tail block MAE no worse than f024.',
                'limitations': ['2023 is repeatedly reused development data, not final independent validation.',
                                'Archive Last-Modified is not first-publication certification.',
                                'No deployment authorization; cross-year confirmation still required.']}
    out.mkdir(parents=True)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    snapshots.to_parquet(out / 'snapshots.parquet', index=False)
    features.to_parquet(out / '2021_features.parquet', index=False)
    train.to_parquet(out / 'training_rows.parquet', index=False)
    old_train.to_parquet(out / 'f024_training_rows.parquet', index=False)
    inference.to_parquet(out / 'inference_rows.parquet', index=False)
    verify_hashes(inputs)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
        train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    prediction = model.predict(inference[names])
    np.testing.assert_array_equal(joblib.load(out / 'model.joblib').predict(inference[names]), prediction)
    cohorts = {}
    for name, frame in [('original', original), ('expanded', expanded)]:
        cohorts[name] = compare_cohort(frame, prediction)
        saved = frame.copy(deep=True)
        saved['extended_2021_aft'] = prediction
        filename = 'predictions.parquet' if name == 'original' else 'expanded_predictions.parquet'
        saved.to_parquet(out / filename, index=False)
    verify_hashes(inputs)
    report = {'cohorts': cohorts, 'coverage': coverage,
              'passed': all(c['passed'] for c in cohorts.values()), 'inputs_unchanged': True,
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
