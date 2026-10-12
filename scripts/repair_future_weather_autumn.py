"""One bounded retry of failed autumn objects, preserving the original report."""

import copy
import hashlib
import json

import pandas as pd

from scripts.benchmark_recovery_selection import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.extract_future_weather_pilot import extract_object


def main():
    folder = ROOT / 'artifacts/task01_recovery_head/future_weather_autumn_fields_v1'
    original_bytes = (folder / 'report.json').read_bytes()
    report = copy.deepcopy(json.loads(original_bytes))
    source = ROOT / 'artifacts/task01_recovery_head/future_weather_seasonal_probe_v1/report.json'
    if hashlib.sha256(source.read_bytes()).hexdigest() != report['source_sha256']:
        raise ValueError('source changed')
    retry = folder / 'repair_01'
    retry.mkdir(exist_ok=False)
    for record in report['records']:
        if record['status'] == 'pass':
            continue
        result = extract_object(record['probe'], retry, ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe')
        record['original_error'] = record.pop('error')
        record.update(result, status='pass')
    rows = []
    for record in report['records']:
        rows.append({**{k: record['probe'][k] for k in ('issue_utc', 'run_utc', 'valid_utc', 'ahead_hours')},
                     **{name: field['value'] for name, field in record['fields'].items()}, 'status': record['status']})
    report['original_report_sha256'] = hashlib.sha256(original_bytes).hexdigest()
    report['objects_passed'] = sum(r['status'] == 'pass' for r in report['records'])
    report['fields_passed'] = sum(len(r['fields']) for r in report['records'])
    report['source_unchanged'] = hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256']
    pd.DataFrame(rows).to_csv(folder / 'sequence_repaired.csv', index=False)
    write_json(folder / 'report_repaired.json', report)
    print(f"objects={report['objects_passed']}, fields={report['fields_passed']}")


if __name__ == '__main__':
    main()
