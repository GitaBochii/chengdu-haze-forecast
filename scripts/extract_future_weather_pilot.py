"""Extract four earlier-cycle GEFS mean objects without changing model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime

import pandas as pd
import requests

from haze_forecast.forecast_sources import parse_gfs_index, range_for_gfs_index_entry
from haze_forecast.grib import query_grib_point
from haze_forecast.operational_gefs_availability import availability_state
from scripts.benchmark_recovery_selection import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.probe_future_weather_cycles import cycle_candidate

FIELDS = {
    'temperature_c': ('TMP', '2 m above ground', 'K', 'degC'),
    'relative_humidity_pct': ('RH', '2 m above ground', '%', '%'),
    'wind_u_ms': ('UGRD', '10 m above ground', 'm/s', 'm/s'),
    'wind_v_ms': ('VGRD', '10 m above ground', 'm/s', 'm/s'),
}


def checked_payload(response, start, end, total, etag):
    if not 0 <= start <= end < total:
        raise ValueError('range outside object bounds')
    if (response.status_code != 206
            or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}'
            or response.headers.get('ETag') != etag):
        raise ValueError('range response or object identity mismatch')
    expected = end - start + 1
    if not 0 < expected <= 2_000_000:
        raise ValueError('range outside pilot byte budget')
    payload = bytearray()
    for chunk in response.iter_content(chunk_size=65536):
        if len(payload) + len(chunk) > expected:
            raise ValueError('overlong range body')
        payload.extend(chunk)
    if len(payload) != expected:
        raise ValueError('truncated range body')
    return bytes(payload)


def bounded_index(response, limit=256_000):
    if response.status_code != 200:
        raise ValueError('index response must be HTTP 200')
    payload = bytearray()
    for chunk in response.iter_content(chunk_size=16384):
        if len(payload) + len(chunk) > limit:
            raise ValueError('index exceeds byte budget')
        payload.extend(chunk)
    return payload.decode('utf-8')


def validate_inventory(text, run, valid, lead, variable, level):
    lines = text.strip().splitlines()
    tokens = lines[0].split(':') if len(lines) == 1 else []
    required = [f'd={run:%Y%m%d%H}', variable, level,
                f'{lead} hour fcst', f'vt={valid:%Y%m%d%H}']
    if not all(token in tokens for token in required):
        raise ValueError(f'GRIB field or time identity mismatch: {text}')


def validate_units(text, unit):
    if len(text.strip().splitlines()) != 1 or f'[{unit}]' not in text or 'ens mean' not in text.split(':'):
        raise ValueError('GRIB unit or ensemble product mismatch')


def validate_candidate(row):
    issue, run, valid = (pd.Timestamp(row[key]) for key in ('issue_utc', 'run_utc', 'valid_utc'))
    lead = row['lead_hours']
    back = row.get('cycles_back', 1)
    if back not in (1, 2, 3, 4):
        raise ValueError('candidate cycle fallback outside audited bounds')
    expected = cycle_candidate(issue, issue.floor('6h'), lead - 6 * back, back)
    if expected != (run, valid, lead, row['url']):
        raise ValueError('probe URL or time mismatch')
    return issue, run, valid, lead


def extract_object(row, output, wgrib2, *, http=None):
    http = requests if http is None else http
    issue, run, valid, lead = validate_candidate(row)
    url = row['url']
    folder = output / f'{run:%Y%m%d%H}_f{lead:03d}'
    folder.mkdir(exist_ok=False)
    with http.head(url, timeout=30, allow_redirects=False) as response:
        evidence = {'url': url, 'checked_at_utc': datetime.now(UTC).isoformat(),
                    'http_status': response.status_code, 'etag': response.headers.get('ETag'),
                    'last_modified': response.headers.get('Last-Modified'),
                    'content_length': response.headers.get('Content-Length')}
    write_json(folder / 'head.json', evidence)
    state, reason = availability_state(evidence, issue, run)
    if state != 'available' or evidence['etag'] != row['etag']:
        raise ValueError('archive proxy or original object identity no longer passes')
    with http.get(url + '.idx', timeout=30, allow_redirects=False, stream=True) as response:
        index_text = bounded_index(response)
    (folder / 'index.txt').write_text(index_text, encoding='utf-8')
    entries = parse_gfs_index(index_text)
    fields = {}
    for name, (variable, level, raw_unit, unit) in FIELDS.items():
        matches = [entry for entry in entries if entry.variable == variable and entry.level == level]
        if len(matches) != 1:
            raise ValueError(f'expected one index entry for {name}')
        start, end = range_for_gfs_index_entry(entries, matches[0])
        with http.get(url, headers={'Range': f'bytes={start}-{end}',
                          'If-Match': evidence['etag'], 'Accept-Encoding': 'identity'},
                          stream=True, timeout=30, allow_redirects=False) as response:
            payload = checked_payload(response, start, end, int(evidence['content_length']), evidence['etag'])
        if payload[:4] != b'GRIB' or payload[-4:] != b'7777':
            raise ValueError('not a complete GRIB message')
        path = folder / f'{name}.grib2'
        path.write_bytes(payload)
        inventory = subprocess.run([str(wgrib2), str(path), '-s', '-vt'], check=True,
                                   capture_output=True, text=True, timeout=30).stdout
        validate_inventory(inventory, run, valid, lead, variable, level)
        verbose = subprocess.run([str(wgrib2), str(path), '-v2'], check=True,
                                 capture_output=True, text=True, timeout=30).stdout
        validate_units(verbose.strip(), raw_unit)
        (folder / f'{name}.inventory.txt').write_text(inventory + verbose, encoding='utf-8')
        point = query_grib_point(path, 104.0667, 30.6667, wgrib2)
        if (point.longitude, point.latitude) != (104.0, 30.5):
            raise ValueError('unexpected nearest grid')
        if not math.isfinite(point.value) or abs(point.value) >= 1e10:
            raise ValueError('invalid or missing point value')
        value = point.value - 273.15 if variable == 'TMP' else point.value
        fields[name] = {'value': value, 'unit': unit, 'raw_value': point.value,
                        'raw_unit': raw_unit, 'longitude': point.longitude, 'latitude': point.latitude,
                        'byte_range': [start, end], 'bytes': len(payload),
                        'sha256': hashlib.sha256(payload).hexdigest(),
                        'path': str(path.relative_to(ROOT)), 'inventory': inventory.strip()}
        write_json(folder / 'fields_progress.json', fields)
    return {'head': evidence, 'state': state, 'reason': reason, 'fields': fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    probe = ROOT / 'artifacts/task01_recovery_head/future_weather_cycle_probe_v1/report.json'
    probe_bytes = probe.read_bytes()
    rows = [row for row in json.loads(probe_bytes)['records'] if row['cycles_back'] == 1]
    if len(rows) != 4:
        raise ValueError('pilot expects exactly four previously selected objects')
    deployed = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in deployed.iterdir() if path.is_file()}
    records = []
    for row in rows:
        record = {'probe': row}
        try:
            record.update(extract_object(row, output, ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe'))
            record['status'] = 'pass'
        except (ValueError, OSError, requests.RequestException, subprocess.SubprocessError) as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        records.append(record)
        write_json(output / 'progress.json', records)
        print(f"{row['year']} f{row['lead_hours']:03d}: {record['status']}", flush=True)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in deployed.iterdir() if path.is_file()}
    report = {'records': records, 'source_probe_sha256': hashlib.sha256(probe_bytes).hexdigest(),
              'source_unchanged': probe.read_bytes() == probe_bytes,
              'deployed_before': hashes, 'deployed_after': after, 'deployed_unchanged': hashes == after,
              'limitations': ['two_origins_only', 'archive_proxy_not_first_publication',
                              'actual_ahead_3_and_21_hours_not_6_and_24',
                              'single_grid_mean_only_not_station_fields', 'no_training_or_accuracy_claim']}
    write_json(output / 'report.json', report)
    if any(row['status'] != 'pass' for row in records) or hashes != after or not report['source_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
