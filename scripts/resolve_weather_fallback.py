"""Resolve incomplete origins using the newest complete audited earlier cycle."""

import hashlib
import json

import pandas as pd
import requests

from haze_forecast.operational_gefs_availability import availability_state, head_evidence
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.probe_future_weather_cycles import cycle_candidate
from scripts.probe_future_weather_sequence import sequence_candidates


def fallback_candidates(issue, back):
    if back not in (2, 3, 4):
        raise ValueError('fallback limited to two through four cycles back')
    baseline = sequence_candidates(issue)
    latest = issue.tz_convert('UTC').floor('6h')
    result = []
    for row in baseline:
        original_lead = int((pd.Timestamp(row['valid_utc']) - latest).total_seconds() / 3600)
        run, valid, lead, url = cycle_candidate(issue, latest, original_lead, back)
        result.append({**row, 'run_utc': run.isoformat(), 'valid_utc': valid.isoformat(),
                       'lead_hours': lead, 'url': url, 'cycles_back': back})
    return result


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    audited = base / 'future_weather_training_audit_v1'
    manifest = base / 'future_weather_training_smoke_v1/candidate_weather_requests.csv'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    prior = json.loads((audited / 'offline_verification.json').read_text(encoding='utf-8'))
    assert prior['status'] == 'pass' and prior['objects_verified'] == 4992
    assert prior['manifest_sha256'] == digest
    planned = pd.read_csv(manifest)
    table = pd.read_csv(audited / 'verified_availability.csv')
    issues = sorted(table.loc[table.state != 'available', 'issue_utc'].unique())
    assert len(issues) == 3
    output = base / 'future_weather_fallback_v1'
    output.mkdir(exist_ok=False)
    attempts, selected, summaries = [], [], []
    with requests.Session() as session:
        for issue_text in issues:
            issue = pd.Timestamp(issue_text)
            accepted = []
            for back in (2, 3, 4):
                current = []
                for row in fallback_candidates(issue, back):
                    evidence = head_evidence(row['url'], output / 'head_cache', head=session.head,
                                             timeout_s=15, attempts=2)
                    state, reason = availability_state(evidence, issue, pd.Timestamp(row['run_utc']))
                    current.append({**row, 'evidence': evidence, 'state': state, 'reason': reason})
                attempts.extend(current)
                write_json(output / 'attempts.json', attempts)
                complete = all(row['state'] == 'available' for row in current)
                print(f'{issue_text} cycles_back={back}: complete={complete}', flush=True)
                if complete:
                    accepted = current
                    selected.extend(current)
                    break
            summaries.append({'issue_utc': issue_text, 'resolved': bool(accepted),
                              'selected_run_utc': accepted[0]['run_utc'] if accepted else None,
                              'cycles_back': accepted[0]['cycles_back'] if accepted else None})
    repaired = planned.copy()
    repaired['cycles_back'] = 1
    for row in selected:
        match = repaired.issue_utc.eq(row['issue_utc']) & repaired.valid_utc.eq(row['valid_utc'])
        assert match.sum() == 1
        for key in ('run_utc', 'lead_hours', 'url', 'cycles_back'):
            repaired.loc[match, key] = row[key]
    assert len(repaired) == len(planned) and not repaired.duplicated(['issue_utc', 'valid_utc']).any()
    assert repaired[['issue_utc', 'valid_utc', 'ahead_hours']].equals(planned[['issue_utc', 'valid_utc', 'ahead_hours']])
    repaired.to_csv(output / 'resolved_candidate_requests.csv', index=False)
    report = {'source_manifest_sha256': digest,
              'source_unchanged': hashlib.sha256(manifest.read_bytes()).hexdigest() == digest,
              'resolved_manifest_sha256': hashlib.sha256((output / 'resolved_candidate_requests.csv').read_bytes()).hexdigest(),
              'summary': summaries, 'selected': selected, 'candidate_rows_checked': len(attempts),
              'selected_replacement_rows': len(selected),
              'limitations': ['HEAD_only', 'archive_proxy_not_first_publication',
                              'maximum_four_cycles_back', 'one_consistent_cycle_per_origin',
                              'no_field_download_or_model_training']}
    write_json(output / 'report.json', report)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
