"""Offline full-envelope verification of all nine received f024 messages."""

import argparse
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from haze_forecast.forecast_sources import parse_gfs_index, range_for_gfs_index_entry
from haze_forecast.grib import query_grib_point
from haze_forecast.operational_gefs import GEFS_FIELD_DEFINITIONS, gefs_data_url
from scripts.collect_gefs_verified_cycle import FIELDS, ROOT, validate_inventory, validate_value


def validate_grid_units(text, field, valid):
    units = {'temperature_2m': 'K', 'relative_humidity_2m': '%',
             'u_wind_10m': 'm/s', 'v_wind_10m': 'm/s', 'pblh': 'm'}
    first = text.splitlines()[0]
    if f'[{units[field]}]' not in first or f':vt={valid:%Y%m%d%H}:' not in first:
        raise ValueError('Wrong GRIB unit or decoded valid time')
    expected = ['grid_template=0:winds(N/S)', '(720 x 361)',
                'lat 90.000000 to -90.000000 by 0.500000',
                'lon 0.000000 to 359.500000 by 0.500000 #points=259920']
    if not all(value in text for value in expected):
        raise ValueError('Wrong grid geometry or wind reference')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path):
    report = json.loads(path.read_bytes())
    run = datetime.fromisoformat(report['run_at'])
    valid = datetime.fromisoformat(report['valid_at'])
    usable = datetime.fromisoformat(report['usable_after'])
    if report['status'] != 'validated' or not run < usable < valid or valid != run + timedelta(hours=24):
        raise ValueError('Invalid cycle status or chronology')
    expected = {(p, f) for p in ('geavg', 'gespr') for f in FIELDS} | {('gec00', 'pblh')}
    if len(report['messages']) != 9 or {(m['product'], m['field']) for m in report['messages']} != expected:
        raise ValueError('Incomplete or duplicate message identities')
    if len(report['indexes']) != 3 or {i['product'] for i in report['indexes']} != {'geavg', 'gespr', 'gec00'}:
        raise ValueError('Incomplete index set')
    decoder = ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    if sha(decoder) != report['decoder_sha256']:
        raise ValueError('Decoder changed')
    dbpath = Path(report['ledger']) / 'receipts.sqlite3'
    db = sqlite3.connect(dbpath.resolve().as_uri() + '?mode=ro', uri=True)
    envelopes = []
    try:
        for item in report['indexes'] + report['messages']:
            rec = item.get('receipt', item)
            if sha(rec['object_path']) != rec['sha256']:
                raise ValueError('Object hash mismatch')
            row = db.execute('SELECT url,status,sha256,completed_at,response_headers FROM attempts WHERE id=?',
                             (rec['attempt'],)).fetchone()
            if row is None or row[:4] != (item['source_url'], 'received_unvalidated', rec['sha256'], rec['completed_at']):
                raise ValueError('Receipt binding mismatch')
            first = db.execute('SELECT first_received_at FROM versions WHERE url=? AND sha256=?',
                               (item['source_url'], rec['sha256'])).fetchone()
            if not first or first[0] != rec['first_received_at']:
                raise ValueError('Version receipt mismatch')
            if not run <= datetime.fromisoformat(rec['completed_at']) <= usable:
                raise ValueError('Receipt chronology mismatch')
            body = Path(rec['object_path']).read_bytes()
            if len(body) != rec['bytes']:
                raise ValueError('Receipt byte count mismatch')
            if 'receipt' not in item:
                continue
            p, field = item['product'], item['field']
            if item['source_url'] != gefs_data_url(run, product=p, lead_hours=24):
                raise ValueError('Wrong source URL')
            index = next(i for i in report['indexes'] if i['product'] == p)
            if index['source_url'] != item['source_url'] + '.idx':
                raise ValueError('Wrong index URL')
            entries = parse_gfs_index(Path(index['object_path']).read_text())
            variable, level = GEFS_FIELD_DEFINITIONS[field]
            matches = [e for e in entries if e.variable == variable and e.level == level]
            if len(matches) != 1 or list(range_for_gfs_index_entry(entries, matches[0])) != item['byte_range']:
                raise ValueError('Index range mismatch')
            start, end = item['byte_range']
            headers = json.loads(row[4])
            if (len(body) != end - start + 1 or not body.startswith(b'GRIB') or not body.endswith(b'7777')
                    or not headers['content-range'].startswith(f'bytes {start}-{end}/')):
                raise ValueError('Range or framing mismatch')
            def decode(*flags, object_path=rec['object_path']):
                return subprocess.run([str(decoder), object_path, *flags], check=True,
                                      capture_output=True, text=True, timeout=30).stdout.strip()
            inventory = decode('-s')
            validate_inventory(inventory, run, p, field)
            if inventory != item['inventory']:
                raise ValueError('Inventory replay mismatch')
            verbose = decode('-V', '-grid')
            validate_grid_units(verbose, field, valid)
            point = query_grib_point(Path(rec['object_path']), 104.06, 30.67, decoder)
            validate_value(p, field, point.value)
            if ([point.longitude, point.latitude] != report['grid'] or report['grid'] != [104., 30.5]
                    or point.value != item['raw_value']):
                raise ValueError('Grid or raw value mismatch')
            converted = point.value - 273.15 if p == 'geavg' and field == 'temperature_2m' else point.value
            if converted != item['converted_value'] or converted != report['features'][item['feature_name']]:
                raise ValueError('Unit conversion mismatch')
            envelopes.append({'product': p, 'field': field, 'verbose_inventory': verbose})
    finally:
        db.close()
    metadata = json.loads((ROOT / 'artifacts/task01_recovery_head/reusable_gefs_aft_v1/protocol.json').read_bytes())
    expected_names = {n for n in metadata['features'] if n.startswith('gefs_')}
    if set(report['features']) != expected_names:
        raise ValueError('Frozen f024 model feature-name mismatch')
    return {'status': 'PASS', 'messages': 9, 'receipts': 12, 'report_sha256': sha(path),
            'decoder_sha256': sha(decoder), 'verifier_sha256': sha(__file__),
            'envelopes': envelopes, 'deployment_authorized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.report)
    output = args.output or args.report.with_name('full_verification.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(f'PASS: 9 fields, 12 receipts, full grid/units/time/schema; {output}')
