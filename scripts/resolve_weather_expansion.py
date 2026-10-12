"""Verify expansion evidence and try bounded whole-cycle repairs, never mix cycles."""

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from haze_forecast.operational_gefs_availability import availability_state, head_evidence
from scripts.probe_future_weather_sequence import sequence_candidates
from scripts.resolve_weather_fallback import fallback_candidates


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_record(record):
    evidence = record['evidence']
    path = Path(evidence['cache_path'])
    assert sha(path) == evidence['cache_sha256']
    raw = json.loads(path.read_text(encoding='utf-8'))
    assert all(raw[key] == value for key, value in evidence.items()
               if key not in ('cache_path', 'cache_sha256'))
    assert raw['url'] == record['url']
    state, reason = availability_state(raw, pd.Timestamp(record['issue_utc']), pd.Timestamp(record['run_utc']))
    assert (state, reason) == (record['state'], record['reason'])


def main():
    base = Path('artifacts/task01_recovery_head')
    source = base / 'future_weather_expansion_full_head_v1'
    output = base / 'future_weather_expansion_resolved_v1'
    plan = json.loads((source / 'plan.json').read_text(encoding='utf-8'))
    expected = {row['url']: row for row in plan}
    assert len(expected) == len(plan) == 6120
    groups = defaultdict(list)
    source_hashes = {str(source / 'plan.json'): sha(source / 'plan.json')}
    result_files = list((source / 'results').glob('*.json'))
    assert len(result_files) == 6120
    seen = set()
    for path in result_files:
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['url'] not in seen
        seen.add(record['url'])
        assert path.stem == hashlib.sha256(record['url'].encode()).hexdigest()
        assert all(record[key] == value for key, value in expected[record['url']].items())
        verify_record(record)
        source_hashes[str(path)] = sha(path)
        groups[record['issue_utc']].append(record)
    assert seen == set(expected) and len(groups) == 255
    output.mkdir(parents=True, exist_ok=False)
    selected, unresolved, repairs = [], [], []
    for issue, rows in sorted(groups.items()):
        originals = {r['ahead_hours']: r for r in sequence_candidates(pd.Timestamp(issue))}
        assert len(rows) == 24 and {r['ahead_hours'] for r in rows} == set(originals)
        assert all(all(row[key] == value for key, value in originals[row['ahead_hours']].items()) for row in rows)
        if all(row['state'] == 'available' for row in rows):
            selected.extend([{**row, 'cycles_back': 1} for row in rows])
        elif all(row['evidence']['http_status'] == 404 for row in rows):
            unresolved.append({'issue_utc': issue, 'reason': 'all_current_layout_objects_404; older_layout_not_audited'})
        else:
            repairs.append(issue)
    print(f'Verified 6120 original records; {len(repairs)} mixed-result origins need fallback', flush=True)
    attempts = []

    def probe(row):
        with requests.Session() as client:
            evidence = head_evidence(row['url'], output / 'head_cache', head=client.head,
                                     timeout_s=15, attempts=2)
        state, reason = availability_state(evidence, pd.Timestamp(row['issue_utc']), pd.Timestamp(row['run_utc']))
        return {**row, 'year': pd.Timestamp(row['issue_utc']).tz_convert('Asia/Shanghai').year,
                'evidence': evidence, 'state': state, 'reason': reason}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for issue in repairs:
            accepted = False
            for back in (2, 3, 4):
                records = list(pool.map(probe, fallback_candidates(pd.Timestamp(issue), back)))
                for record in records:
                    verify_record(record)
                attempts.extend(records)
                (output / 'attempts.json').write_text(json.dumps(attempts, indent=2), encoding='utf-8')
                accepted = all(row['state'] == 'available' for row in records)
                print(f'{issue}: cycles_back={back}, complete={accepted}', flush=True)
                if accepted:
                    selected.extend(records)
                    break
            if not accepted:
                unresolved.append({'issue_utc': issue, 'reason': 'no_complete_cycle_within_four_cycle_limit'})
    assert all(sha(Path(path)) == value for path, value in source_hashes.items())
    table = pd.DataFrame([{key: row[key] for key in ('issue_utc', 'run_utc', 'valid_utc',
                          'ahead_hours', 'lead_hours', 'url', 'cycles_back', 'year')} for row in selected])
    assert not table.duplicated(['issue_utc', 'ahead_hours']).any()
    assert table.groupby('issue_utc').size().eq(24).all()
    assert table.groupby('issue_utc').run_utc.nunique().eq(1).all()
    table.sort_values(['issue_utc', 'ahead_hours']).to_csv(output / 'accepted_manifest.csv', index=False)
    (output / 'accepted_evidence.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    (output / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=2), encoding='utf-8')
    report = {'status': 'verified_HEAD_only', 'original_records_verified': 6120,
              'accepted_origins': int(table.issue_utc.nunique()), 'accepted_rows': len(table),
              'accepted_origins_by_year': table.groupby('year').issue_utc.nunique().to_dict(),
              'fallback_origins_attempted': len(repairs), 'fallback_objects_checked': len(attempts),
              'unresolved': unresolved, 'manifest_sha256': sha(output / 'accepted_manifest.csv'),
              'limitations': ['HEAD_only_not_ready_for_training', 'archive_timestamp_proxy',
                              'all_404_origins_not_subject_to_cycle_fallback', 'no_model_changes']}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'unresolved'}
                     | {'unresolved_origins': len(unresolved)}, indent=2))


if __name__ == '__main__':
    main()
