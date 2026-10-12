"""Verify frozen keys, earlier-cycle evidence and final coverage without network."""

import hashlib
import json

import pandas as pd

from haze_forecast.operational_gefs_availability import availability_state
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    folder = base / 'future_weather_fallback_v1'
    report = json.loads((folder / 'report.json').read_bytes())
    source = base / 'future_weather_training_smoke_v1/candidate_weather_requests.csv'
    resolved = folder / 'resolved_candidate_requests.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_manifest_sha256']
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == report['resolved_manifest_sha256']
    before, after = pd.read_csv(source), pd.read_csv(resolved)
    assert len(before) == len(after) == 4992
    pd.testing.assert_frame_equal(before[['issue_utc', 'valid_utc', 'ahead_hours', 'partition']],
                                  after[['issue_utc', 'valid_utc', 'ahead_hours', 'partition']])
    attempts = json.loads((folder / 'attempts.json').read_bytes())
    for row in attempts:
        evidence = row['evidence']
        path = ROOT.joinpath(evidence['cache_path']).resolve()
        assert path.is_relative_to((folder / 'head_cache').resolve())
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == evidence['cache_sha256']
        raw = json.loads(payload)
        assert all(evidence[key] == value for key, value in raw.items())
        assert raw['url'] == row['url']
        issue, run, valid = [pd.Timestamp(row[key]) for key in ('issue_utc', 'run_utc', 'valid_utc')]
        assert run == issue.floor('6h') - pd.Timedelta(hours=6 * row['cycles_back'])
        assert (valid - run).total_seconds() / 3600 == row['lead_hours']
        assert (valid - issue).total_seconds() / 3600 == row['ahead_hours']
        assert availability_state(raw, issue, run) == (row['state'], row['reason'])
    selected = {(r['issue_utc'], r['valid_utc']): r for r in report['selected']}
    assert len(selected) == 72 and all(r['state'] == 'available' for r in selected.values())
    baseline = pd.read_csv(base / 'future_weather_training_audit_v1/verified_availability.csv')
    states = {(r.issue_utc, r.valid_utc): r.state for r in baseline.itertuples()}
    for row in after.to_dict('records'):
        key = (row['issue_utc'], row['valid_utc'])
        if key in selected:
            assert all(row[k] == selected[key][k] for k in ('url', 'run_utc', 'lead_hours', 'cycles_back'))
            states[key] = selected[key]['state']
        else:
            original = before.loc[before.issue_utc.eq(key[0]) & before.valid_utc.eq(key[1])].iloc[0]
            assert row['url'] == original.url and row['run_utc'] == original.run_utc
    assert len(states) == 4992 and all(state == 'available' for state in states.values())
    for summary in report['summary']:
        for back in range(2, summary['cycles_back']):
            prior = [r for r in attempts if r['issue_utc'] == summary['issue_utc'] and r['cycles_back'] == back]
            assert len(prior) == 24 and any(r['state'] != 'available' for r in prior)
    result = {'status': 'pass', 'final_available_rows': 4992, 'origins': 208,
              'unique_object_urls': int(after.url.nunique()), 'replacement_rows': len(selected),
              'candidate_rows_verified': len(attempts), 'original_keys_preserved': True,
              'limitations': ['archive_timestamp_proxy_only', 'no_field_extraction_or_model_training']}
    write_json(folder / 'offline_verification.json', result)
    print(result)


if __name__ == '__main__':
    main()
