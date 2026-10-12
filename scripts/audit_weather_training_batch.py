"""Resume bounded HEAD batches against the frozen weather training manifest."""

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime

import pandas as pd
import requests

from haze_forecast.operational_gefs_availability import availability_state, head_evidence
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.probe_future_weather_sequence import sequence_candidates


def pending_rows(planned, done, limit):
    return [row for row in planned if row['url'] not in done][:limit]


def summarize(planned, done):
    counts = Counter(done[row['url']]['state'] if row['url'] in done else 'pending' for row in planned)
    groups = defaultdict(list)
    for row in planned:
        groups[row['issue_utc']].append(done.get(row['url'], {}).get('state', 'pending'))
    return {**{key: counts[key] for key in ('available', 'not_yet_available', 'unknown', 'pending')},
            'expected_objects': len(planned), 'expected_origins': len(groups),
            'complete_origins': sum(len(states) == 24 and all(state == 'available' for state in states)
                                    for states in groups.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=256)
    args = parser.parse_args()
    if not 0 <= args.limit <= 512:
        raise ValueError('batch limit must be 0-512; zero only summarizes')
    source_dir = ROOT / 'artifacts/task01_recovery_head/future_weather_training_smoke_v1'
    payload = (source_dir / 'candidate_weather_requests.csv').read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = json.loads((source_dir / 'report.json').read_text(encoding='utf-8'))['planned_requests_sha256']
    if digest != expected:
        raise ValueError('candidate manifest changed')
    planned = pd.read_csv(io.BytesIO(payload)).to_dict('records')
    if len(planned) != 4992 or len({row['url'] for row in planned}) != len(planned):
        raise ValueError('candidate count or unique URL contract failed')
    for row in planned:
        issue = pd.Timestamp(row['issue_utc'])
        candidates = sequence_candidates(issue)
        candidate = next(r for r in candidates if r['ahead_hours'] == row['ahead_hours'])
        if any(candidate[key] != row[key] for key in candidate):
            raise ValueError('candidate URL or time alignment mismatch')
    output = ROOT / 'artifacts/task01_recovery_head/future_weather_training_audit_v1'
    output.mkdir(exist_ok=True)
    results = output / 'results'
    results.mkdir(exist_ok=True)
    done = {}
    for path in results.glob('*.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['manifest_sha256'] != digest:
            raise ValueError('cached result belongs to another manifest')
        cached = record['evidence']
        evidence_bytes = ROOT.joinpath(cached['cache_path']).read_bytes()
        if hashlib.sha256(evidence_bytes).hexdigest() != cached['cache_sha256']:
            raise ValueError('cached HEAD evidence changed')
        done[record['url']] = record
    selected = pending_rows(planned, done, args.limit)
    with requests.Session() as session:
        for number, row in enumerate(selected, 1):
            evidence = head_evidence(row['url'], output / 'head_cache', head=session.head, timeout_s=15, attempts=2)
            state, reason = availability_state(evidence, pd.Timestamp(row['issue_utc']), pd.Timestamp(row['run_utc']))
            record = {**row, 'state': state, 'reason': reason, 'evidence': evidence, 'manifest_sha256': digest}
            key = hashlib.sha256(row['url'].encode()).hexdigest()
            write_json(results / f'{key}.json', record)
            done[row['url']] = record
            if number % 24 == 0:
                print(f'{number}/{len(selected)} new objects checked', flush=True)
    summary = summarize(planned, done)
    report = {'manifest_sha256': digest, 'checked_at_utc': datetime.now(UTC).isoformat(),
              'new_objects_checked': len(selected), **summary,
              'limitations': ['HEAD_only_no_new_GRIB_fields', 'archive_proxy_not_first_publication',
                              'pending_not_available_and_not_failure', 'unknown_requires_explicit_retry',
                              'no_model_training_or_performance_claim']}
    write_json(output / 'report.json', report)
    write_json(output / f"batch_{datetime.now(UTC):%Y%m%dT%H%M%S%f}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
