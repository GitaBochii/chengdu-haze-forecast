"""Isolated bounded collection for the verified expansion manifest."""

import argparse
import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import pandas as pd
import requests

from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.collect_weather_training_fields import verify_cached, worker_http_client
from scripts.extract_future_weather_pilot import extract_object, validate_candidate
from scripts.resolve_weather_expansion import verify_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    assert 0 <= args.limit <= 512 and 1 <= args.workers <= 8
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'future_weather_expansion_resolved_v1'
    manifest = source / 'accepted_manifest.csv'
    payload = manifest.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == 'f33a2adce9a13719b3590f79e9539cd89aa23ec551f12f8ac1fa8a6f6e359ef3'
    plan = pd.read_csv(manifest)
    assert len(plan) == 4272 and plan.issue_utc.nunique() == 178
    evidence = json.loads((source / 'accepted_evidence.json').read_bytes())
    by_key = {(row['issue_utc'], row['ahead_hours']): row for row in evidence}
    assert len(by_key) == len(evidence) == len(plan)
    for row in plan.to_dict('records'):
        cached = by_key[(row['issue_utc'], row['ahead_hours'])]
        assert all(cached[key] == value for key, value in row.items())
        validate_candidate(row)
        verify_record(cached)
        assert cached['state'] == 'available'
    rows = plan.sort_values(['issue_utc', 'ahead_hours']).drop_duplicates('url').to_dict('records')
    for row in rows:
        row['etag'] = by_key[(row['issue_utc'], row['ahead_hours'])]['evidence']['etag']
    urls = {row['url']: row for row in rows}
    output = base / 'future_weather_expansion_collection_v1'
    results = output / 'results'
    results.mkdir(parents=True, exist_ok=True)
    done = {}
    for path in results.glob('*.json'):
        record = json.loads(path.read_bytes())
        url = record['probe']['url']
        assert record['manifest_sha256'] == digest and url in urls and url not in done
        assert path.stem == hashlib.sha256(url.encode()).hexdigest()
        if record['status'] == 'pass':
            verify_cached(record, urls[url])
        done[url] = record
    selected = [row for row in rows if (done.get(row['url'], {}).get('status') == 'failed'
                if args.retry_failed else row['url'] not in done)][:args.limit]
    local = threading.local()
    clients = []

    def collect(row):
        if not hasattr(local, 'http'):
            local.http = worker_http_client()
            clients.append(local.http)
        key = hashlib.sha256(row['url'].encode()).hexdigest()
        attempt = output / 'objects' / key / uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        record = {'probe': row, 'manifest_sha256': digest,
                  'attempt_path': str(attempt.relative_to(ROOT))}
        try:
            record.update(extract_object(row, attempt, ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe', http=local.http))
            verify_cached(record, row)
            record['status'] = 'pass'
        except (ValueError, OSError, requests.RequestException, subprocess.SubprocessError, AssertionError) as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        return record

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for number, future in enumerate(as_completed([pool.submit(collect, row) for row in selected]), 1):
                record = future.result()
                url = record['probe']['url']
                if url in done:
                    previous = done[url]
                    record['previous_attempts'] = [*previous.get('previous_attempts', []),
                        {key: value for key, value in previous.items() if key != 'previous_attempts'}]
                key = hashlib.sha256(url.encode()).hexdigest()
                temporary = results / f'{key}.tmp'
                write_json(temporary, record)
                temporary.replace(results / f'{key}.json')
                done[url] = record
                if number % 16 == 0:
                    print(f'{number}/{len(selected)} objects finished', flush=True)
    finally:
        for client in clients:
            client.close()
    assert manifest.read_bytes() == payload
    passed = {url for url, record in done.items() if record['status'] == 'pass'}
    complete = plan.assign(passed=plan.url.isin(passed)).groupby('issue_utc').passed.all()
    report = {'manifest_sha256': digest, 'expected_objects': len(rows), 'expected_origins': 178,
              'passed_objects': len(passed), 'failed_objects': sum(r['status'] == 'failed' for r in done.values()),
              'pending_objects': len(rows) - len(done), 'complete_origins': int(complete.sum()),
              'new_attempts': len(selected), 'limitations': ['no_training', 'independent_full_gate_still_required']}
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
