"""Independently reconcile the complete fixed manifest against saved HEAD bytes."""

import hashlib
import json

import pandas as pd

from haze_forecast.operational_gefs_availability import availability_state
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json


def main():
    source = ROOT / 'artifacts/task01_recovery_head/future_weather_training_smoke_v1'
    manifest = source / 'candidate_weather_requests.csv'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest == json.loads((source / 'report.json').read_text(encoding='utf-8'))['planned_requests_sha256']
    planned = pd.read_csv(manifest)
    folder = ROOT / 'artifacts/task01_recovery_head/future_weather_training_audit_v1'
    records = []
    for path in (folder / 'results').glob('*.json'):
        result = json.loads(path.read_text(encoding='utf-8'))
        assert result['manifest_sha256'] == digest
        evidence = result['evidence']
        cached_path = ROOT.joinpath(evidence['cache_path']).resolve()
        assert cached_path.is_relative_to((folder / 'head_cache').resolve())
        raw = cached_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == evidence['cache_sha256']
        raw_evidence = json.loads(raw)
        assert all(evidence[key] == value for key, value in raw_evidence.items())
        assert raw_evidence['url'] == result['url']
        issue, run, valid = [pd.Timestamp(result[key]) for key in ('issue_utc', 'run_utc', 'valid_utc')]
        state, reason = availability_state(raw_evidence, issue, run)
        assert (state, reason) == (result['state'], result['reason'])
        assert run == issue.floor('6h') - pd.Timedelta(hours=6)
        assert (valid - issue).total_seconds() / 3600 == result['ahead_hours']
        records.append({key: result[key] for key in (*planned.columns, 'state')} | {
            'last_modified': raw_evidence['last_modified'], 'http_status': raw_evidence['http_status']})
    actual = pd.DataFrame(records)
    assert len(actual) == len(planned) == 4992
    assert not actual.duplicated('url').any()
    pd.testing.assert_frame_equal(actual[list(planned.columns)].sort_values('url').reset_index(drop=True),
                                  planned.sort_values('url').reset_index(drop=True))
    origins = actual.groupby('issue_utc').agg(objects=('url', 'size'),
                                            available=('state', lambda s: s.eq('available').sum()))
    assert len(origins) == 208 and origins.objects.eq(24).all()
    actual['year'] = pd.to_datetime(actual.issue_utc, utc=True).dt.year
    by_year = actual.groupby(['year', 'state']).size().rename('objects').reset_index()
    actual.to_csv(folder / 'verified_availability.csv', index=False)
    report = {'status': 'pass', 'objects_verified': len(actual), 'origins_verified': len(origins),
              'complete_available_origins': int(origins.available.eq(24).sum()),
              'incomplete_available_origins': int(origins.available.ne(24).sum()),
              'states': actual.state.value_counts().to_dict(), 'by_year': by_year.to_dict('records'),
              'manifest_sha256': digest,
              'limitations': ['HEAD_only', 'archive_proxy_not_first_publication', 'no_new_field_extraction']}
    write_json(folder / 'offline_verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
