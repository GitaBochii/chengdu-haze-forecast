"""Bounded, resumable, URL-deduplicated weather field collection."""

import argparse
import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from haze_forecast.operational_gefs_availability import availability_state
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.extract_future_weather_pilot import FIELDS, extract_object, validate_candidate


def worker_http_client():
    client = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({'HEAD', 'GET'}), respect_retry_after_header=True,
                  raise_on_status=False)
    client.mount('https://', HTTPAdapter(max_retries=retry))
    return client


def verify_cached(record, row):
    assert record['head']['url'] == row['url'] and record['head']['etag'] == row['etag']
    assert availability_state(record['head'], pd.Timestamp(row['issue_utc']),
                              pd.Timestamp(row['run_utc']))[0] == 'available'
    assert set(record['fields']) == set(FIELDS)
    for field in record['fields'].values():
        path = (ROOT / field['path']).resolve()
        assert path.is_relative_to((ROOT / 'artifacts/task01_recovery_head').resolve())
        content = path.read_bytes()
        assert len(content) == field['bytes']
        assert hashlib.sha256(content).hexdigest() == field['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.limit <= 512 or not 1 <= args.workers <= 8:
        raise ValueError('limit 0-512 and workers 1-8 required')
    base = ROOT / 'artifacts/task01_recovery_head'
    fallback = json.loads((base / 'future_weather_fallback_v1/report.json').read_bytes())
    manifest = base / 'future_weather_fallback_v1/resolved_candidate_requests.csv'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest == fallback['resolved_manifest_sha256']
    plan = pd.read_csv(manifest)
    assert len(plan) == 4992 and plan.url.nunique() == 4926
    evidence = {}
    for path in (base / 'future_weather_training_audit_v1/results').glob('*.json'):
        record = json.loads(path.read_bytes())
        if record['state'] == 'available':
            evidence[record['url']] = record['evidence']
    evidence.update({row['url']: row['evidence'] for row in fallback['selected']})
    rows = plan.sort_values('issue_utc').drop_duplicates('url').to_dict('records')
    for row in rows:
        validate_candidate(row)
        head = evidence[row['url']]
        payload = ROOT.joinpath(head['cache_path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == head['cache_sha256']
        raw = json.loads(payload)
        assert raw['url'] == row['url']
        assert availability_state(raw, pd.Timestamp(row['issue_utc']), pd.Timestamp(row['run_utc']))[0] == 'available'
        row['etag'] = raw['etag']
    by_url = {row['url']: row for row in rows}
    output = base / 'future_weather_collection_v1'
    results = output / 'results'
    results.mkdir(parents=True, exist_ok=True)
    done = {}

    def persist(record):
        key = hashlib.sha256(record['probe']['url'].encode()).hexdigest()
        temporary = results / f'{key}.tmp'
        write_json(temporary, record)
        temporary.replace(results / f'{key}.json')

    for path in results.glob('*.json'):
        record = json.loads(path.read_bytes())
        assert record['manifest_sha256'] == digest
        url = record['probe']['url']
        if record['status'] == 'pass':
            verify_cached(record, by_url[url])
        done[url] = record
    for relative in ('future_weather_sequence_fields_v1/report.json',
                     'future_weather_autumn_fields_v1/report_repaired.json'):
        source = base / relative
        payload = source.read_bytes()
        for record in json.loads(payload)['records']:
            url = record['probe']['url']
            if url in done or url not in by_url:
                continue
            assert record['status'] == 'pass'
            verify_cached(record, by_url[url])
            imported = {**record, 'probe': by_url[url], 'manifest_sha256': digest,
                        'reused_from': relative, 'reused_report_sha256': hashlib.sha256(payload).hexdigest()}
            persist(imported)
            done[url] = imported
    if args.retry_failed:
        selected = [row for row in rows if done.get(row['url'], {}).get('status') == 'failed'][:args.limit]
    else:
        selected = [row for row in rows if row['url'] not in done][:args.limit]

    worker_state = threading.local()
    sessions = []

    def collect(row):
        if not hasattr(worker_state, 'http'):
            worker_state.http = worker_http_client()
            sessions.append(worker_state.http)
        key = hashlib.sha256(row['url'].encode()).hexdigest()
        attempt = output / 'objects' / key / uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        record = {'probe': row, 'manifest_sha256': digest, 'attempt_path': str(attempt.relative_to(ROOT))}
        try:
            record.update(extract_object(row, attempt, ROOT / 'tools/wgrib2/v3.1.3/wgrib2.exe',
                                         http=worker_state.http))
            verify_cached(record, row)
            record['status'] = 'pass'
        except (ValueError, OSError, requests.RequestException, subprocess.SubprocessError, AssertionError) as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect, row) for row in selected]
        for number, future in enumerate(as_completed(futures), 1):
            record = future.result()
            previous = done.get(record['probe']['url'])
            if previous is not None:
                record['previous_attempts'] = [*previous.get('previous_attempts', []),
                                               {k: v for k, v in previous.items() if k != 'previous_attempts'}]
            persist(record)
            done[record['probe']['url']] = record
            if number % 16 == 0:
                print(f'{number}/{len(selected)} new objects finished', flush=True)
    for session in sessions:
        session.close()
    passed = {url for url, record in done.items() if record['status'] == 'pass'}
    counts = plan.assign(collected=plan.url.isin(passed)).groupby('issue_utc').collected.all()
    report = {'manifest_sha256': digest, 'expected_unique_objects': len(rows),
              'new_objects_attempted': len(selected), 'passed_objects': len(passed),
              'failed_objects': sum(r['status'] == 'failed' for r in done.values()),
              'pending_objects': len(rows) - len(done), 'complete_origins': int(counts.sum()),
              'expected_origins': len(counts), 'passed_fields': len(passed) * 4,
              'reused_objects': sum('reused_from' in r for r in done.values()),
              'limitations': ['partial_until_pending_and_failures_resolved', 'archive_proxy_not_first_publication',
                              'no_model_training', 'failed_objects_require_explicit_repair']}
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
