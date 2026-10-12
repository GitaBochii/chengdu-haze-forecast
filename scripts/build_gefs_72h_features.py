"""Build f072 strategy inputs only after complete independent collection audit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.forecast_snapshot_reuse import reuse_forecast_snapshots

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = {
    'geavg': ('mean', ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']),
    'gespr': ('spread', ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']),
    'gec00': ('control', ['boundary_layer_height_m']),
}
FIELDS = [f'gefs_{prefix}_{field}_f072' for prefix, fields in DEFINITIONS.values()
          for field in fields]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    result = pd.Timestamp(value)
    require(not pd.isna(result) and result.tzinfo is not None, 'timezone-aware time required')
    return result.tz_convert('UTC')


def validated_snapshots(root, expected_runs=292):
    base = root / 'artifacts/task01_recovery_head'
    collection = base / 'gefs_72h_collection_v1'
    inputs = {}

    def read(path):
        data = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def check(path, expected):
        actual = digest(path)
        require(actual == expected, f'stale or changed input: {path}')
        inputs[str(path)] = actual

    protocol_path = collection / 'protocol.json'
    protocol = read(protocol_path)
    runs = [timestamp(run) for run in protocol['runs']]
    require(len(runs) == expected_runs and len(set(runs)) == expected_runs,
            f'exactly {expected_runs} unique planned runs required')
    source = base / 'reusable_gefs_snapshots_v1/snapshots.parquet'
    check(source, protocol['source_sha256'])
    source_times = pd.read_parquet(source).run_at
    source_runs = {timestamp(run) for run in source_times}
    require(source_runs == set(runs), 'source run population mismatch')
    check(root / 'tools/wgrib2/v3.1.3/wgrib2.exe', protocol['decoder_sha256'])
    check(root / 'scripts/extract_gefs_72h_batch.py', protocol['collector_sha256'])
    audit = read(collection / 'independent_audit.json')
    require(audit['protocol_sha256'] == inputs[str(protocol_path)], 'stale audit protocol')
    require(audit['verified_runs'] == audit['planned_runs'] == expected_runs
            and audit['verified_grib_messages'] == expected_runs * 9, 'incomplete audit counts')
    check(root / 'scripts/audit_gefs_72h_collection.py', audit['audit_script_sha256'])
    require(set(audit['helper_sha256']) == {'grib.py', 'forecast_sources.py'},
            'incomplete audit helper hashes')
    for name, value in audit['helper_sha256'].items():
        check(root / 'src/haze_forecast' / name, value)
    verified = audit['verified']
    require(len(verified) == expected_runs, 'incomplete audit coverage')
    audit_runs = {timestamp(item['run_at']): item['complete_sha256'] for item in verified}
    require(set(audit_runs) == set(runs), 'audit run coverage mismatch')
    markers = {collection / run.strftime('%Y%m%d%H') / 'complete.json' for run in runs}
    require(set(collection.glob('*/complete.json')) == markers, 'incomplete completion markers')
    rows = []
    for run in sorted(runs):
        folder = collection / run.strftime('%Y%m%d%H')
        marker = folder / 'complete.json'
        check(marker, audit_runs[run])
        record = read(marker)
        require(timestamp(record['run_at']) == run, 'marker run mismatch')
        valid = timestamp(record['valid_at'])
        require((valid - run).total_seconds() == 72 * 3600, 'wrong forecast lead')
        require(set(record['products']) == set(DEFINITIONS), 'incomplete products')
        expected_files = {f'{product}.idx' for product in DEFINITIONS}
        values, availability = {}, []
        for product, (prefix, fields) in DEFINITIONS.items():
            items = record['products'][product]['fields']
            require(set(items) == set(fields), 'incomplete forecast fields')
            for field in fields:
                item = items[field]
                name = f'{product}_{field}.grb2'
                expected_files.add(name)
                require(item['sha256'] == record['output_sha256'][name], 'field hash mismatch')
                value = float(item['value'])
                require(np.isfinite(value), 'nonfinite forecast value')
                if product != 'geavg':
                    require(value >= 0, 'negative spread or boundary layer height')
                if product == 'geavg' and field == 'relative_humidity_pct':
                    require(0 <= value <= 100, 'invalid relative humidity')
                # The collector already converted the mean temperature to Celsius.
                values[f'gefs_{prefix}_{field}_f072'] = value
                availability.append(timestamp(item['last_modified']))
        require(set(record['output_sha256']) == expected_files, 'raw output inventory mismatch')
        for name in sorted(expected_files):
            check(folder / name, record['output_sha256'][name])
        known = max(availability)
        require(known == timestamp(record['archive_available_at_proxy']), 'availability mismatch')
        require(run <= known, 'availability precedes initialization')
        rows.append({'known_at': known, 'run_at': run, 'valid_at': valid, **values})
    snapshots = pd.DataFrame(rows)
    require(not snapshots.known_at.duplicated().any(), 'ambiguous known times')
    require(all(digest(Path(path)) == value for path, value in inputs.items()),
            'inputs changed during verification')
    return snapshots, inputs


def match_origins(snapshots, origins):
    result = reuse_forecast_snapshots(snapshots, origins, FIELDS)
    require(result.timestamp.tolist() == list(origins), 'origin order changed')
    return result


def build(root=ROOT):
    base = root / 'artifacts/task01_recovery_head'
    output = base / 'gefs_72h_features_v1'
    if output.exists():
        raise FileExistsError(output)
    snapshots, inputs = validated_snapshots(root)
    origin_path = base / 'interval_business_labels_v1/origins.parquet'
    for path in [origin_path, Path(__file__), root / 'src/haze_forecast/forecast_snapshot_reuse.py',
                 root / 'gefs_72h_comparison_protocol_20260906.md']:
        inputs[str(path)] = digest(path)
    origins = pd.read_parquet(origin_path)
    require(len(origins) == 7890, 'exactly 7890 origins required')
    result = match_origins(snapshots, origins.timestamp)
    yearly = {str(year): {'origins': len(frame),
                         'available': int(frame.known_at.notna().sum())}
              for year, frame in result.groupby(result.timestamp.dt.year)}
    require(all(digest(Path(path)) == value for path, value in inputs.items()),
            'inputs changed during construction')
    output.mkdir(parents=True)
    snapshots.to_parquet(output / 'snapshots.parquet', index=False)
    result.to_parquet(output / 'features.parquet', index=False)
    report = {'input_sha256': inputs, 'fields': FIELDS, 'snapshots': len(snapshots),
              'origins': len(result), 'yearly': yearly, 'formal_model_evaluation_allowed': False,
              'limitations': ['Archive Last-Modified proxy is not first-publication proof.',
                              'Feature construction only; no accuracy or training authorization.',
                              'Independent origin selection and split audits remain required.'],
              'output_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'snapshots': len(snapshots), 'yearly': yearly}, indent=2))


if __name__ == '__main__':
    build()
