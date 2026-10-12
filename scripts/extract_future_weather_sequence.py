"""Download the fixed, audited two-origin sequence pilot; retain every failure."""

import argparse
import hashlib
import json
import subprocess

import pandas as pd
import requests

from scripts.benchmark_recovery_selection import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.extract_future_weather_pilot import extract_object


def select_records(rows, autumn=False):
    return [row for row in rows if row.get('quarter') == 4] if autumn else rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--autumn', action='store_true')
    args = parser.parse_args()
    source_name = 'future_weather_seasonal_probe_v1' if args.autumn else 'future_weather_sequence_probe_v1'
    source = ROOT / 'artifacts/task01_recovery_head' / source_name / 'report.json'
    source_bytes = source.read_bytes()
    planned = select_records(json.loads(source_bytes)['records'], args.autumn)
    if len(planned) != 48 or any(row['state'] != 'available' for row in planned):
        raise ValueError('expected all 48 fixed candidates to pass the archive proxy')
    output_name = 'future_weather_autumn_fields_v1' if args.autumn else 'future_weather_sequence_fields_v1'
    output = ROOT / 'artifacts/task01_recovery_head' / output_name
    output.mkdir(exist_ok=False)
    records, table = [], []
    for row in planned:
        record = {'probe': row}
        values = {key: row[key] for key in ('issue_utc', 'run_utc', 'valid_utc', 'ahead_hours')}
        try:
            record.update(extract_object(row, output, ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'))
            record['status'] = 'pass'
            values.update({name: field['value'] for name, field in record['fields'].items()})
        except (ValueError, OSError, requests.RequestException, subprocess.SubprocessError) as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        values['status'] = record['status']
        records.append(record)
        table.append(values)
        write_json(output / 'progress.json', records)
        print(f"{len(records)}/48: {record['status']}", flush=True)
    pd.DataFrame(table).to_csv(output / 'sequence.csv', index=False)
    result = {'records': records, 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
              'source_unchanged': source.read_bytes() == source_bytes,
              'objects_passed': sum(r['status'] == 'pass' for r in records),
              'fields_passed': sum(len(r.get('fields', {})) for r in records),
              'limitations': ['two_fixed_origins_only', 'three_hourly_not_hourly',
                              'single_grid_ensemble_mean', 'archive_proxy_not_first_publication',
                              'no_model_training_or_accuracy_claim']}
    write_json(output / 'report.json', result)
    if result['objects_passed'] != 48 or not result['source_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
