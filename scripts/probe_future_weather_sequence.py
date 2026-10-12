"""Bounded HEAD-only audit of issue-relative 72-hour GEFS pilot sequences."""

import argparse
import hashlib
import json
from datetime import UTC, datetime

import pandas as pd
import requests

from haze_forecast.operational_gefs_availability import availability_state
from scripts.benchmark_recovery_selection import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.probe_future_weather_cycles import cycle_candidate


def sequence_candidates(issue):
    if issue.tzinfo is None:
        raise ValueError('issue must be timezone aware')
    issue = issue.tz_convert('UTC')
    if issue != issue.floor('3h'):
        raise ValueError('pilot requires three-hour-aligned issue')
    latest = issue.floor('6h')
    offset = int((issue - latest).total_seconds() / 3600)
    records = []
    for ahead in range(3, 73, 3):
        run, valid, lead, url = cycle_candidate(issue, latest, ahead + offset, 1)
        records.append({'issue_utc': issue.isoformat(), 'run_utc': run.isoformat(),
                        'valid_utc': valid.isoformat(), 'lead_hours': lead,
                        'ahead_hours': ahead, 'url': url})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'artifacts/task01_recovery_head/future_weather_cycle_probe_v1/report.json'
    source_bytes = source.read_bytes()
    issues = sorted({r['issue_utc'] for r in json.loads(source_bytes)['records']})
    if len(issues) != 2:
        raise ValueError('expected the two fixed pilot origins')
    planned = [r for issue in issues for r in sequence_candidates(pd.Timestamp(issue))]
    write_json(output / 'plan.json', planned)
    records = []
    for row in planned:
        evidence = {'http_status': None, 'etag': None, 'last_modified': None,
                    'checked_at_utc': datetime.now(UTC).isoformat()}
        try:
            with requests.head(row['url'], timeout=15, allow_redirects=False) as response:
                evidence.update(http_status=response.status_code,
                                etag=response.headers.get('ETag'),
                                last_modified=response.headers.get('Last-Modified'),
                                content_length=response.headers.get('Content-Length'))
        except requests.RequestException as exc:
            evidence['error'] = str(exc)
        state, reason = availability_state(evidence, pd.Timestamp(row['issue_utc']),
                                           pd.Timestamp(row['run_utc']))
        records.append({**row, **evidence, 'state': state, 'reason': reason})
        write_json(output / 'progress.json', records)
        print(f"{len(records)}/48 ahead={row['ahead_hours']} state={state}", flush=True)
    summary = []
    for issue in issues:
        subset = [r for r in records if r['issue_utc'] == issue]
        summary.append({'issue_utc': issue, 'expected': 24,
                        'available': sum(r['state'] == 'available' for r in subset),
                        'failed_ahead_hours': [r['ahead_hours'] for r in subset if r['state'] != 'available']})
    write_json(output / 'report.json', {
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'source_unchanged': source.read_bytes() == source_bytes,
        'summary': summary, 'records': records,
        'limitations': ['HEAD_only_no_field_values', 'two_fixed_origins_only',
                        'archive_timestamp_proxy_not_first_publication',
                        'three_hourly_not_hourly', 'no_model_training'],
    })
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
