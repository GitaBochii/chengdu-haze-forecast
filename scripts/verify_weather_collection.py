"""Re-read every completed collected object, while retaining partial coverage."""

import hashlib
import json
import math
import subprocess

import pandas as pd

from haze_forecast.grib import parse_wgrib2_point_output
from haze_forecast.operational_gefs_availability import availability_state
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.extract_future_weather_pilot import FIELDS, validate_inventory, validate_units


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    folder = base / 'future_weather_collection_v1'
    manifest = base / 'future_weather_fallback_v1/resolved_candidate_requests.csv'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = pd.read_csv(manifest)
    passed, failed, total_bytes = set(), set(), 0
    for path in (folder / 'results').glob('*.json'):
        record = json.loads(path.read_bytes())
        assert record['manifest_sha256'] == digest
        row = record['probe']
        match = plan.loc[plan.url.eq(row['url']) & plan.issue_utc.eq(row['issue_utc'])]
        assert len(match) == 1
        assert all(match.iloc[0][key] == row[key] for key in ('run_utc', 'valid_utc', 'lead_hours', 'cycles_back'))
        if record['status'] != 'pass':
            failed.add(row['url'])
            continue
        issue, run, valid = [pd.Timestamp(row[key]) for key in ('issue_utc', 'run_utc', 'valid_utc')]
        assert availability_state(record['head'], issue, run)[0] == 'available'
        assert record['head']['etag'] == row['etag']
        assert set(record['fields']) == set(FIELDS)
        for name, (variable, level, raw_unit, unit) in FIELDS.items():
            field = record['fields'][name]
            grib = (ROOT / field['path']).resolve()
            assert grib.is_relative_to(base.resolve())
            payload = grib.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == field['sha256']
            assert len(payload) == field['bytes'] == field['byte_range'][1] - field['byte_range'][0] + 1
            binary = str(ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe')
            result = subprocess.run([binary, str(grib), '-s', '-vt', '-lon', '104.0667', '30.6667'],
                                    check=True, capture_output=True, text=True, timeout=30).stdout
            validate_inventory(result, run, valid, row['lead_hours'], variable, level)
            point = parse_wgrib2_point_output(result)
            assert point.longitude == field['longitude'] == 104.0
            assert point.latitude == field['latitude'] == 30.5
            assert point.value == field['raw_value']
            converted = point.value - 273.15 if variable == 'TMP' else point.value
            assert math.isclose(converted, field['value'], rel_tol=0, abs_tol=1e-10)
            verbose = subprocess.run([binary, str(grib), '-v2'], check=True,
                                     capture_output=True, text=True, timeout=30).stdout.strip()
            validate_units(verbose, raw_unit)
            assert field['unit'] == unit and field['raw_unit'] == raw_unit
            total_bytes += len(payload)
        passed.add(row['url'])
    completed = plan.assign(present=plan.url.isin(passed)).groupby('issue_utc').present.all()
    report = {'status': 'verified_collected_subset', 'passed_objects': len(passed),
              'passed_fields': len(passed) * 4, 'failed_objects': len(failed),
              'pending_objects': int(plan.url.nunique()) - len(passed) - len(failed),
              'complete_origins': int(completed.sum()), 'expected_origins': len(completed),
              'verified_grib_bytes': total_bytes, 'manifest_sha256': digest,
              'limitations': ['partial_collection_not_training_ready', 'archive_proxy_not_first_publication',
                              'no_model_training_or_accuracy_claim']}
    write_json(folder / 'offline_verification.json', report)
    print(report)


if __name__ == '__main__':
    main()
