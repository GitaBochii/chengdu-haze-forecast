"""Offline, full-coverage GRIB verification before any formal weather training."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = 'f7a77f53e8b8278c3cb650846867d66dadb13d8dd29b96235b05c7e972cd0dc2'
FIELDS = {'temperature_c': ('TMP', '2 m above ground', 'K', 'degC'),
          'relative_humidity_pct': ('RH', '2 m above ground', '%', '%'),
          'wind_u_ms': ('UGRD', '10 m above ground', 'm/s', 'm/s'),
          'wind_v_ms': ('VGRD', '10 m above ground', 'm/s', 'm/s')}
INPUTS = ('station_hourly.parquet', 'weather_proxy.parquet', 'city_hourly.parquet')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc(value):
    value = pd.Timestamp(value)
    require(not pd.isna(value) and value.tzinfo is not None
            and value.utcoffset().total_seconds() == 0, 'expected UTC timestamp')
    return value


def contained(root, value, parent):
    path = (root / value).resolve()
    require(path.is_relative_to(parent.resolve()), f'path outside evidence tree: {value}')
    return path


def check_plan(plan, counts=(4992, 208, 4926)):
    require((len(plan), plan.issue_utc.nunique(), plan.url.nunique()) == counts, 'manifest coverage mismatch')
    require(not plan.duplicated(['issue_utc', 'valid_utc']).any(), 'duplicate origin/valid relationship')
    for issue, group in plan.groupby('issue_utc'):
        issue = utc(issue)
        require(issue == issue.floor('3h'), 'origin not three-hour aligned')
        require(sorted(group.ahead_hours.tolist()) == list(range(3, 73, 3)), 'incomplete 72-hour sequence')
        require(group.run_utc.nunique() == 1 and group.cycles_back.nunique() == 1, 'mixed cycles within origin')
        for row in group.to_dict('records'):
            run, valid = utc(row['run_utc']), utc(row['valid_utc'])
            back, lead = row['cycles_back'], row['lead_hours']
            require(back in (1, 2, 3, 4), 'unaudited cycle fallback')
            require(run == issue.floor('6h') - pd.Timedelta(hours=6 * back), 'incorrect fallback cycle')
            require(valid == issue + pd.Timedelta(hours=row['ahead_hours']), 'incorrect issue-relative valid time')
            require(valid == run + pd.Timedelta(hours=lead) and lead % 3 == 0, 'incorrect forecast lead')
            url = (f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run:%Y%m%d}/{run:%H}/atmos/'
                   f'pgrb2ap5/geavg.t{run:%H}z.pgrb2a.0p50.f{lead:03d}')
            require(row['url'] == url, 'URL identity mismatch')
    for _, group in plan.groupby('url'):
        require(len(group[['run_utc', 'valid_utc', 'lead_hours']].drop_duplicates()) == 1,
                'shared object has conflicting forecast identity')


def check_head(head, row):
    require(head.get('url') == row['url'] and head.get('http_status') == 200, 'HEAD URL/status mismatch')
    require(re.fullmatch(r'"[0-9a-fA-F]{32}"', str(head.get('etag', ''))) is not None,
            'unknown or multipart ETag')
    modified = utc(parsedate_to_datetime(head['last_modified']))
    require(utc(row['run_utc']) <= modified <= utc(row['issue_utc']), 'archive timestamp unavailable at origin')
    require(int(head['content_length']) > 0, 'invalid object length')


def check_field(field, name, row, root, binary, runner=subprocess.run):
    base = root / 'artifacts/task01_recovery_head'
    path = contained(root, field['path'], base)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == field['sha256'], 'GRIB hash mismatch')
    start, end = field['byte_range']
    require(isinstance(start, int) and isinstance(end, int) and 0 <= start <= end,
            'invalid byte range')
    require(len(payload) == field['bytes'] == end - start + 1, 'GRIB length mismatch')
    require(payload[:4] == b'GRIB' and payload[-4:] == b'7777', 'invalid GRIB envelope')
    variable, level, raw_unit, unit = FIELDS[name]
    run, valid, lead = utc(row['run_utc']), utc(row['valid_utc']), row['lead_hours']
    def read(*options):
        return runner([str(binary), str(path), *options], check=True, capture_output=True,
                      text=True, timeout=30).stdout.strip()
    inventory = read('-s', '-vt', '-lon', '104.0667', '30.6667')
    require(len(inventory.splitlines()) == 1, 'multiple GRIB messages')
    tokens = inventory.split(':')
    for token in (f'd={run:%Y%m%d%H}', f'vt={valid:%Y%m%d%H}', variable,
                  level, f'{lead} hour fcst', 'ens mean'):
        require(token in tokens, f'GRIB identity mismatch: {token}')
    verbose = read('-v2')
    require(len(verbose.splitlines()) == 1 and f'[{raw_unit}]' in verbose
            and 'ens mean' in verbose.split(':'), 'GRIB unit/product mismatch')
    points = [token for token in tokens if token.startswith('lon=')]
    require(len(points) == 1, 'missing or duplicate GRIB point')
    point = dict(part.split('=') for part in points[0].split(','))
    require(float(point['lon']) == field['longitude'] == 104.0
            and float(point['lat']) == field['latitude'] == 30.5, 'grid mismatch')
    raw = float(point['val'])
    require(math.isfinite(raw) and abs(raw) < 1e10 and raw == field['raw_value'], 'raw value mismatch')
    value = raw - 273.15 if name == 'temperature_c' else raw
    require(math.isclose(value, field['value'], rel_tol=0, abs_tol=1e-10), 'converted value mismatch')
    require(field['raw_unit'] == raw_unit and field['unit'] == unit, 'saved unit mismatch')
    return str(path.relative_to(root)), digest, len(payload)


def evidence_for_plan(root, plan, tracked):
    base = root / 'artifacts/task01_recovery_head'
    fallback_path = base / 'future_weather_fallback_v1/report.json'
    fallback = json.loads(fallback_path.read_bytes())
    tracked[str(fallback_path.relative_to(root))] = sha(fallback_path)
    require(fallback['resolved_manifest_sha256'] == MANIFEST_SHA256, 'fallback manifest mismatch')
    evidence = {}
    for path in (base / 'future_weather_training_audit_v1/results').glob('*.json'):
        record = json.loads(path.read_bytes())
        if record['state'] == 'available':
            evidence[record['url']] = record['evidence']
    evidence.update({row['url']: row['evidence'] for row in fallback['selected']})
    verified = {}
    for url, group in plan.groupby('url'):
        source = evidence[url]
        path = contained(root, source['cache_path'], base)
        digest = sha(path)
        require(digest == source['cache_sha256'], 'original HEAD cache hash mismatch')
        tracked[str(path.relative_to(root))] = digest
        head = json.loads(path.read_bytes())
        for row in group.to_dict('records'):
            check_head(head, row)
        verified[url] = head
    return verified


def verify(root=ROOT, workers=8):
    require(1 <= workers <= 8, 'workers must be 1-8')
    base = root / 'artifacts/task01_recovery_head'
    collection = base / 'future_weather_collection_v1'
    gate = collection / 'training_readiness_gate.json'
    # Retain the previous attestation, but never expose it as current during a recheck.
    if gate.exists():
        gate.replace(collection / f'training_readiness_gate.previous_{uuid4().hex}.json')
    manifest = base / 'future_weather_fallback_v1/resolved_candidate_requests.csv'
    payload = manifest.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == MANIFEST_SHA256, 'fixed manifest hash changed')
    plan = pd.read_csv(io.BytesIO(payload))
    check_plan(plan)
    inputs = root / 'data/processed/regional_air_models_full_history'
    input_hashes = {name: sha(inputs / name) for name in INPUTS}
    binary = root / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    binary_hash, verifier_hash = sha(binary), sha(Path(__file__))
    tracked = {str(manifest.relative_to(root)): MANIFEST_SHA256}
    evidence = evidence_for_plan(root, plan, tracked)
    paths = sorted((collection / 'results').glob('*.json'))
    require(len(paths) == 4926, 'pending or extra result objects')
    result_hashes, records = {}, {}
    for path in paths:
        content = path.read_bytes()
        result_hashes[path.name] = hashlib.sha256(content).hexdigest()
        record = json.loads(content)
        url = record['probe']['url']
        require(path.name == hashlib.sha256(url.encode()).hexdigest() + '.json', 'result filename URL mismatch')
        require(url not in records and url in evidence, 'duplicate or unplanned result')
        require(record['manifest_sha256'] == MANIFEST_SHA256 and record['status'] == 'pass',
                'current failed result or manifest mismatch')
        records[url] = record
    require(set(records) == set(plan.url), 'incomplete result URL coverage')
    jobs = []
    for url, group in plan.groupby('url'):
        record = records[url]
        probe = record['probe']
        matches = group.loc[group.issue_utc.eq(probe['issue_utc'])]
        require(len(matches) == 1, 'cached origin not in manifest')
        row = matches.iloc[0].to_dict()
        for key in ('run_utc', 'valid_utc', 'lead_hours', 'ahead_hours', 'cycles_back'):
            require(probe[key] == row[key], f'cached probe mismatch: {key}')
        require(record['head']['etag'] == evidence[url]['etag'] == probe['etag'], 'cached object ETag conflict')
        for shared in group.to_dict('records'):
            check_head(record['head'], shared)
        require(set(record['fields']) == set(FIELDS), 'missing or extra fields')
        if 'reused_from' in record:
            source = contained(base, record['reused_from'], base)
            require(sha(source) == record['reused_report_sha256'], 'reused report hash mismatch')
            tracked[str(source.relative_to(root))] = sha(source)
            originals = [r for r in json.loads(source.read_bytes())['records'] if r['probe']['url'] == url]
            require(len(originals) == 1 and originals[0]['head'] == record['head']
                    and originals[0]['fields'] == record['fields'], 'reused record conflicts with source')
        for field in record['fields'].values():
            require(field['byte_range'][1] < int(record['head']['content_length']), 'range exceeds object length')
        jobs.append((record, row))
    def verify_object(job):
        record, row = job
        return [check_field(field, name, row, root, binary) for name, field in record['fields'].items()]
    gribs, total_bytes = {}, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(verify_object, job) for job in jobs]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                for path, digest, size in future.result():
                    require(path not in gribs, 'different objects reuse a GRIB file path')
                    gribs[path] = digest
                    total_bytes += size
                if count % 256 == 0:
                    print(f'{count}/4926 objects independently verified', flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    require({p.name: sha(p) for p in (collection / 'results').glob('*.json')} == result_hashes,
            'results changed during verification')
    for path, digest in {**tracked, **gribs}.items():
        require(sha(root / path) == digest, f'evidence changed during verification: {path}')
    require({name: sha(inputs / name) for name in INPUTS} == input_hashes, 'training inputs changed')
    require(sha(binary) == binary_hash and sha(Path(__file__)) == verifier_hash, 'verification tool changed')
    summary = {'status': 'ready_for_formal_training', 'manifest_sha256': MANIFEST_SHA256,
               'passed_objects': 4926, 'failed_objects': 0, 'pending_objects': 0, 'complete_origins': 208,
               'verified_fields': len(gribs), 'verified_grib_bytes': total_bytes,
               'results_sha256': result_hashes, 'input_data_sha256': input_hashes,
               'verifier_sha256': verifier_hash, 'wgrib2_sha256': binary_hash,
               'evidence_sha256': tracked, 'grib_sha256': gribs,
               'verified_at_utc': datetime.now(UTC).isoformat(),
               'limitations': ['archive_timestamp_proxy_not_historical_first_publication',
                               'three_hourly_single_grid_ensemble_mean_not_hourly_station_weather',
                               'readiness_is_data_integrity_not_model_accuracy_or_deployment_approval']}
    temporary = collection / f'training_readiness_gate.{uuid4().hex}.tmp'
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(gate)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    report = verify(workers=parser.parse_args().workers)
    print(json.dumps({key: report[key] for key in ('status', 'passed_objects', 'failed_objects',
                                                   'pending_objects', 'complete_origins')}))


if __name__ == '__main__':
    main()
