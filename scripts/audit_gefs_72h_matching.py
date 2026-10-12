"""Check every f072 match independently, without the production selector."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGES = ['forecast_known_age_hours', 'forecast_run_age_hours', 'forecast_remaining_lead_hours']
TIMES = ['known_at', 'run_at', 'valid_at']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_matches(snapshots, features, origins, fields):
    origins = pd.DatetimeIndex(origins)
    require(origins.tz is not None and not origins.hasnans and origins.is_unique, 'invalid origins')
    require(features.timestamp.tolist() == origins.tolist(), 'origin order or count mismatch')
    require(len(set(fields)) == len(fields) > 0, 'invalid fields')
    require(not snapshots.known_at.duplicated().any(), 'ambiguous release times')
    records = snapshots.to_dict('records')
    for item in records:
        require(all(pd.notna(item[t]) and item[t].tzinfo is not None for t in TIMES), 'invalid time')
        require(item['run_at'] <= item['known_at'] and item['valid_at'] > item['run_at'], 'invalid chronology')
        require(np.isfinite([item[f] for f in fields]).all(), 'invalid snapshot values')
    for actual in features.to_dict('records'):
        origin = actual['timestamp']
        eligible = [r for r in records if r['known_at'] <= origin < r['valid_at']]
        if not eligible:
            require(all(pd.isna(actual[k]) for k in fields + AGES + TIMES), 'missing forecast mismatch')
            continue
        expected = max(eligible, key=lambda r: r['known_at'])
        require(all(actual[t] == expected[t] for t in TIMES), 'selected snapshot mismatch')
        values = [expected[f] for f in fields] + [
            (origin - expected['known_at']).total_seconds() / 3600,
            (origin - expected['run_at']).total_seconds() / 3600,
            (expected['valid_at'] - origin).total_seconds() / 3600,
        ]
        require(np.allclose([actual[k] for k in fields + AGES], values, rtol=0, atol=1e-10),
                'field or age mismatch')
    return len(features)


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    folder = base / 'gefs_72h_features_v1'
    report_path = folder / 'report.json'
    report = json.loads(report_path.read_bytes())
    inputs = {str(report_path): sha(report_path), str(Path(__file__)): sha(Path(__file__))}
    for path, value in report['input_sha256'].items():
        require(sha(Path(path)) == value, f'stale source: {path}')
        inputs[path] = value
    for name in ['features.parquet', 'snapshots.parquet']:
        path = folder / name
        require(sha(path) == report['output_sha256'][name], f'changed output: {name}')
        inputs[str(path)] = sha(path)
    origin_path = base / 'interval_business_labels_v1/origins.parquet'
    require(str(origin_path) in inputs, 'origin source hash missing')
    origins = pd.read_parquet(origin_path).timestamp
    features = pd.read_parquet(folder / 'features.parquet')
    snapshots = pd.read_parquet(folder / 'snapshots.parquet')
    require(len(origins) == 7890 and len(snapshots) == 292, 'incomplete population')
    expected_fields = [f'gefs_{prefix}_{field}_f072' for prefix in ['mean', 'spread']
                       for field in ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']]
    expected_fields += ['gefs_control_boundary_layer_height_m_f072']
    require(report['fields'] == expected_fields, 'wrong forecast fields')
    count = verify_matches(snapshots, features, origins, expected_fields)
    require(all(sha(Path(p)) == value for p, value in inputs.items()), 'inputs changed during audit')
    audit = {'passed': True, 'origin_rows': count, 'snapshot_rows': len(snapshots),
             'input_sha256': inputs,
             'checks': ['origin_order', 'latest_eligible_selection', 'all_fields_and_ages',
                        'missing_rows_retained', 'source_hashes'],
             'limitations': ['Matching consistency only, not accuracy or deployment approval.']}
    (folder / 'matching_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'origin_rows': count, 'snapshot_rows': len(snapshots)}))


if __name__ == '__main__':
    main()
