"""Independently verify every expansion GRIB before expanded-weather training."""

import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import pandas as pd

from scripts import verify_weather_training_ready as original
from scripts.verify_weather_training_ready import (
    FIELDS,
    INPUTS,
    ROOT,
    check_field,
    check_head,
    check_plan,
    contained,
    require,
    sha,
)

MANIFEST_SHA256 = 'f33a2adce9a13719b3590f79e9539cd89aa23ec551f12f8ac1fa8a6f6e359ef3'
COUNTS = (4272, 178, 4184)


def tracked_json(path, root, tracked):
    payload = path.read_bytes()
    tracked[str(path.relative_to(root))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def evidence_for_plan(root, plan, tracked):
    base = root / 'artifacts/task01_recovery_head'
    source = base / 'future_weather_expansion_resolved_v1/accepted_evidence.json'
    records = tracked_json(source, root, tracked)
    by_key = {(r['issue_utc'], r['ahead_hours']): r for r in records}
    require(len(by_key) == len(records) == len(plan), 'evidence relationship coverage mismatch')
    expected_keys = set(zip(plan.issue_utc, plan.ahead_hours, strict=True))
    require(set(by_key) == expected_keys, 'evidence relationship keys mismatch')
    heads = {}
    for row in plan.to_dict('records'):
        record = by_key[(row['issue_utc'], row['ahead_hours'])]
        require(all(record.get(k) == v for k, v in row.items()), 'evidence identity mismatch')
        require(record['state'] == 'available', 'original evidence not available')
        evidence = record['evidence']
        path = contained(root, evidence['cache_path'], base)
        head = tracked_json(path, root, tracked)
        require(tracked[str(path.relative_to(root))] == evidence['cache_sha256'],
                'original HEAD cache hash mismatch')
        require(all(head.get(k) == v for k, v in evidence.items()
                    if k not in ('cache_path', 'cache_sha256')), 'cached HEAD value mismatch')
        check_head(head, row)
        if row['url'] in heads:
            previous = heads[row['url']]
            require(all(head[k] == previous[k] for k in ('etag', 'last_modified', 'content_length')),
                    'shared URL original HEAD conflict')
        heads[row['url']] = head
    return heads


def load_results(collection, plan, evidence):
    hashes, records = {}, {}
    paths = sorted((collection / 'results').glob('*.json'))
    require(len(paths) == plan.url.nunique(), 'pending or extra result objects')
    for path in paths:
        payload = path.read_bytes()
        hashes[path.name] = hashlib.sha256(payload).hexdigest()
        record = json.loads(payload)
        url = record['probe']['url']
        require(path.name == hashlib.sha256(url.encode()).hexdigest() + '.json',
                'result filename URL mismatch')
        require(url not in records and url in evidence, 'duplicate or unplanned result')
        require(record['manifest_sha256'] == MANIFEST_SHA256 and record['status'] == 'pass',
                'failed result or manifest mismatch')
        require('reused_from' not in record, 'expansion reuse not independently audited')
        records[url] = record
    require(set(records) == set(plan.url), 'incomplete result URL coverage')
    return hashes, records


def object_jobs(plan, records, evidence):
    jobs = []
    for url, group in plan.groupby('url'):
        record = records[url]
        probe = record['probe']
        matches = group.loc[group.issue_utc.eq(probe['issue_utc'])]
        require(len(matches) == 1, 'cached origin not in manifest')
        row = matches.iloc[0].to_dict()
        require(all(probe.get(k) == v for k, v in row.items()), 'cached probe identity mismatch')
        require(record['head']['etag'] == evidence[url]['etag'] == probe['etag'], 'object ETag conflict')
        require(int(record['head']['content_length']) == int(evidence[url]['content_length']),
                'object size changed since original HEAD')
        for shared in group.to_dict('records'):
            check_head(record['head'], shared)
        require(set(record['fields']) == set(FIELDS), 'missing or extra fields')
        ranges = []
        for field in record['fields'].values():
            start, end = field['byte_range']
            require(isinstance(start, int) and isinstance(end, int)
                    and 0 <= start <= end < int(record['head']['content_length']), 'invalid object range')
            ranges.append((start, end))
        ranges.sort()
        require(all(left[1] < right[0] for left, right in pairwise(ranges)),
                'overlapping field ranges')
        jobs.append((record, row))
    return jobs


def verify(root=ROOT, workers=8):
    require(1 <= workers <= 8, 'workers must be 1-8')
    base = root / 'artifacts/task01_recovery_head'
    collection = base / 'future_weather_expansion_collection_v1'
    gate = collection / 'training_readiness_gate.json'
    if gate.exists():
        gate.replace(collection / f'training_readiness_gate.previous_{uuid4().hex}.json')
    manifest = base / 'future_weather_expansion_resolved_v1/accepted_manifest.csv'
    payload = manifest.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == MANIFEST_SHA256, 'fixed manifest hash changed')
    plan = pd.read_csv(io.BytesIO(payload))
    check_plan(plan, COUNTS)
    require(set(plan.year) == {2020, 2021}, 'unexpected expansion years')
    for row in plan.to_dict('records'):
        require(pd.Timestamp(row['issue_utc']).tz_convert('Asia/Shanghai').year == row['year'],
                'expansion year identity mismatch')
    tracked = {str(manifest.relative_to(root)): MANIFEST_SHA256}
    formal = tracked_json(base / 'future_weather_formal_v1/report.json', root, tracked)
    require(formal['status'] == 'completed_train_validation_only', 'original formal training not complete')
    inputs = root / 'data/processed/regional_air_models_full_history'
    input_hashes = {name: sha(inputs / name) for name in INPUTS}
    require(input_hashes == formal['inputs_before'], 'historical data differs from original formal training')
    binary = root / 'tools/wgrib2/v3.1.3/wgrib2.exe'
    binary_hash = sha(binary)
    verifier_hash = sha(Path(__file__))
    shared_hash = sha(Path(original.__file__))
    evidence = evidence_for_plan(root, plan, tracked)
    result_hashes, records = load_results(collection, plan, evidence)
    jobs = object_jobs(plan, records, evidence)

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
                    print(f'{count}/{COUNTS[2]} objects independently verified', flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    require(len(gribs) == COUNTS[2] * len(FIELDS), 'incomplete field coverage')
    require({p.name: sha(p) for p in (collection / 'results').glob('*.json')} == result_hashes,
            'results changed during verification')
    for path, digest in {**tracked, **gribs}.items():
        require(sha(root / path) == digest, f'evidence changed during verification: {path}')
    require({name: sha(inputs / name) for name in INPUTS} == input_hashes, 'training inputs changed')
    require(sha(binary) == binary_hash and sha(Path(__file__)) == verifier_hash
            and sha(Path(original.__file__)) == shared_hash, 'verification tool changed')
    summary = {'status': 'ready_for_formal_training', 'manifest_sha256': MANIFEST_SHA256,
               'passed_objects': COUNTS[2], 'failed_objects': 0, 'pending_objects': 0,
               'complete_origins': COUNTS[1], 'manifest_relationships': COUNTS[0],
               'verified_fields': len(gribs), 'verified_grib_bytes': total_bytes,
               'results_sha256': result_hashes, 'input_data_sha256': input_hashes,
               'verifier_sha256': verifier_hash, 'shared_verifier_sha256': shared_hash,
               'wgrib2_sha256': binary_hash, 'evidence_sha256': tracked, 'grib_sha256': gribs,
               'verified_at_utc': datetime.now(UTC).isoformat(),
               'limitations': ['archive_timestamp_proxy_not_historical_first_publication',
                               'three_hourly_single_grid_ensemble_mean_not_hourly_station_weather',
                               'readiness_is_integrity_not_accuracy_or_deployment_approval']}
    temporary = gate.with_suffix('.tmp')
    temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    temporary.replace(gate)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    report = verify(workers=args.workers)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_sha256')}, indent=2))


if __name__ == '__main__':
    main()
