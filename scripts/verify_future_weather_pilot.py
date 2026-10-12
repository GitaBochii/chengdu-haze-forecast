"""Offline re-read of persisted pilot GRIBs, independent of download pass flags."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from scripts.benchmark_regional_air_models import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', action='store_true')
    parser.add_argument('--autumn', action='store_true')
    parser.add_argument('--repaired', action='store_true')
    args = parser.parse_args()
    if args.autumn:
        args.sequence = True
    name = 'future_weather_sequence_fields_v1' if args.sequence else 'future_weather_fields_v1'
    if args.autumn:
        name = 'future_weather_autumn_fields_v1'
    folder = ROOT / 'artifacts/task01_recovery_head' / name
    report_name = 'report_repaired.json' if args.repaired else 'report.json'
    report = json.loads((folder / report_name).read_text())
    if args.repaired:
        assert hashlib.sha256((folder / 'report.json').read_bytes()).hexdigest() == report['original_report_sha256']
    if args.sequence:
        source = ROOT / 'artifacts/task01_recovery_head/future_weather_sequence_probe_v1/report.json'
        if args.autumn:
            source = ROOT / 'artifacts/task01_recovery_head/future_weather_seasonal_probe_v1/report.json'
        assert report['source_unchanged']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256']
        planned = json.loads(source.read_bytes())['records']
        if args.autumn:
            planned = [r for r in planned if r['quarter'] == 4]
        assert [r['probe'] for r in report['records']] == planned
    expected = {
        'temperature_c': ('TMP', '2 m above ground', 'K', 'degC'),
        'relative_humidity_pct': ('RH', '2 m above ground', '%', '%'),
        'wind_u_ms': ('UGRD', '10 m above ground', 'm/s', 'm/s'),
        'wind_v_ms': ('VGRD', '10 m above ground', 'm/s', 'm/s'),
    }
    records = []
    count = 48 if args.sequence else 4
    assert len(report['records']) == count
    assert len({(r['probe']['issue_utc'], r['probe']['lead_hours']) for r in report['records']}) == count
    if args.sequence:
        issues = {r['probe']['issue_utc'] for r in report['records']}
        assert len(issues) == 2
        for issue in issues:
            assert sorted(r['probe']['ahead_hours'] for r in report['records']
                          if r['probe']['issue_utc'] == issue) == list(range(3, 73, 3))
    for record in report['records']:
        probe = record['probe']
        issue = pd.Timestamp(probe['issue_utc'])
        run = pd.Timestamp(probe['run_utc'])
        valid = pd.Timestamp(probe['valid_utc'])
        lead = probe['lead_hours']
        assert lead in (range(12, 82, 3) if args.sequence else (12, 30))
        assert run == issue.floor('6h') - pd.Timedelta(hours=6)
        assert valid == run + pd.Timedelta(hours=lead)
        assert (valid - issue).total_seconds() / 3600 == probe['ahead_hours']
        assert run <= pd.Timestamp(record['head']['last_modified']) <= issue
        assert record['head']['etag'] == probe['etag']
        assert set(record['fields']) == set(expected)
        values = {}
        for name, (variable, level, raw_unit, unit) in expected.items():
            field = record['fields'][name]
            path = (ROOT / field['path']).resolve()
            assert path.is_relative_to(folder.resolve())
            payload = path.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == field['sha256']
            assert len(payload) == field['bytes'] == field['byte_range'][1] - field['byte_range'][0] + 1
            assert payload[:4] == b'GRIB' and payload[-4:] == b'7777'
            result = subprocess.run([
                str(ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'), str(path),
                '-s', '-vt', '-lon', '104.0667', '30.6667',
            ], check=True, capture_output=True, text=True, timeout=30).stdout.strip()
            assert len(result.splitlines()) == 1
            tokens = result.split(':')
            for token in (f'd={run:%Y%m%d%H}', f'vt={valid:%Y%m%d%H}',
                          variable, level, f'{lead} hour fcst', 'ens mean'):
                assert token in tokens, (name, token, result)
            verbose = subprocess.run([
                str(ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'), str(path), '-v2',
            ], check=True, capture_output=True, text=True, timeout=30).stdout
            assert f'[{raw_unit}]' in verbose
            point = next(token for token in tokens if token.startswith('lon='))
            numeric = dict(part.split('=') for part in point.split(','))
            assert float(numeric['lon']) == field['longitude'] == 104.0
            assert float(numeric['lat']) == field['latitude'] == 30.5
            raw = float(numeric['val'])
            assert math.isfinite(raw) and abs(raw) < 1e10
            assert raw == field['raw_value']
            converted = raw - 273.15 if variable == 'TMP' else raw
            assert math.isclose(converted, field['value'], abs_tol=1e-10)
            assert field['raw_unit'] == raw_unit and field['unit'] == unit
            values[name] = converted
        records.append({'issue_utc': issue.isoformat(), 'valid_utc': valid.isoformat(),
                        'ahead_hours': (valid - issue).total_seconds() / 3600, **values})
    if args.sequence:
        table = pd.read_csv(folder / ('sequence_repaired.csv' if args.repaired else 'sequence.csv'))
        assert len(table) == count and table.status.eq('pass').all()
        assert not table.duplicated(['issue_utc', 'valid_utc']).any()
        for row in records:
            matched = table.loc[table.issue_utc.eq(row['issue_utc']) & table.valid_utc.eq(row['valid_utc'])]
            assert len(matched) == 1
            for name in (*expected, 'ahead_hours'):
                assert math.isclose(float(matched.iloc[0][name]), row[name], rel_tol=0, abs_tol=1e-10)
    baseline = json.loads((ROOT / 'artifacts/task01_recovery_head/future_weather_fields_v1/report.json').read_text())
    for name, digest in baseline['deployed_before'].items():
        path = ROOT / 'artifacts/cnemc_2025_deployment_candidate' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    result = {'status': 'pass', 'objects_verified': count, 'fields_verified': count * 4,
              'grib_bytes': sum(f['bytes'] for r in report['records'] for f in r['fields'].values()),
              'deployment_unchanged': True, 'records': records,
              'limitations': report['limitations']}
    write_json(folder / 'offline_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
