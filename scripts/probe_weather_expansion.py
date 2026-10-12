"""Bounded endpoint HEAD preflight for frozen 2020-2021 training origins."""

import argparse
import hashlib
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from haze_forecast.operational_gefs_availability import availability_state, head_evidence
from scripts.probe_future_weather_sequence import sequence_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    source = Path('artifacts/task01_recovery_head/future_weather_supervision_audit_v1/causal_training_candidates.csv')
    output = Path('artifacts/task01_recovery_head') / (
        'future_weather_expansion_full_head_v1' if args.full else 'future_weather_expansion_probe_v1')
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    candidates = pd.read_csv(source)
    candidates = candidates.loc[candidates.year.isin([2020, 2021])].sort_values('timestamp')
    assert len(candidates) == 255 and candidates.timestamp.is_unique
    planned = []
    for row in candidates.itertuples(index=False):
        sequence = sequence_candidates(pd.Timestamp(row.timestamp))
        for entry in (sequence if args.full else (sequence[0], sequence[-1])):
            planned.append({**entry, 'year': int(row.year)})
    expected = 255 * (24 if args.full else 2)
    assert len(planned) == expected and len({row['url'] for row in planned}) == expected
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(planned, indent=2), encoding='utf-8')
    (output / 'results').mkdir()
    local = threading.local()
    clients = []

    def probe(row):
        if not hasattr(local, 'client'):
            local.client = requests.Session()
            clients.append(local.client)
        evidence = head_evidence(row['url'], output / 'head_cache', head=local.client.head,
                                 timeout_s=15, attempts=1)
        state, reason = availability_state(evidence, pd.Timestamp(row['issue_utc']), pd.Timestamp(row['run_utc']))
        return {**row, 'state': state, 'reason': reason, 'evidence': evidence}

    records = []
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(probe, row) for row in planned]
            for future in as_completed(futures):
                record = future.result()
                key = hashlib.sha256(record['url'].encode()).hexdigest()
                (output / 'results' / f'{key}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
                records.append(record)
                if len(records) % 50 == 0:
                    print(f'{len(records)}/{expected} HEAD objects checked', flush=True)
    finally:
        for client in clients:
            client.close()
    assert source.read_bytes() == payload
    rows = pd.DataFrame([{key: record[key] for key in ('issue_utc', 'year', 'ahead_hours', 'state')}
                         | {'http_status': record['evidence']['http_status']} for record in records])
    rows.to_csv(output / 'endpoint_states.csv', index=False)
    summary = {'status': 'completed_full_head_preflight' if args.full else 'completed_endpoint_preflight',
               'source_sha256': digest,
               'origins': 255, 'checked_objects': len(records),
               'states': dict(Counter(row['state'] for row in records)),
               'http_statuses': dict(Counter(str(row['evidence']['http_status']) for row in records)),
               'origins_all_checked_steps_available': int(rows.groupby('issue_utc').state.apply(
                   lambda values: len(values) == (24 if args.full else 2) and values.eq('available').all()).sum()),
               'checked_steps_per_origin': 24 if args.full else 2,
               'limitations': ['fixed_one_cycle_back_no_fallback',
                               'HEAD_only_no_field_validation', 'archive_timestamp_proxy_not_first_publication',
                               '404_does_not_prove_absence_under_other_archive_layouts', 'no_training']}
    if not args.full:
        summary['limitations'].append('only_ahead_3_and_72_not_full_24_steps')
    (output / 'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
