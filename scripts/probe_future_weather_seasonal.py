"""Fixed quarterly pollution-origin availability audit, without test-year selection."""

import hashlib
import io
import json
from datetime import UTC, datetime

import pandas as pd
import requests

from haze_forecast.operational_gefs_availability import availability_state
from scripts.benchmark_recovery_selection import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.probe_future_weather_sequence import sequence_candidates


def select_seasonal_origins(frame):
    frame = frame.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
    frame['year'] = frame.timestamp.dt.year
    frame['quarter'] = frame.timestamp.dt.quarter
    rows = []
    for year in (2022, 2023):
        for quarter in range(1, 5):
            candidates = frame.loc[(frame.year == year) & (frame.quarter == quarter)
                                   & frame.eligible.fillna(False)].sort_values('timestamp')
            rows.append({'year': year, 'quarter': quarter, 'eligible_count': len(candidates),
                         'issue_utc': candidates.iloc[0].timestamp.isoformat() if len(candidates) else None})
    return rows


def main():
    output = ROOT / 'artifacts/task01_recovery_head/future_weather_seasonal_probe_v1'
    output.mkdir(exist_ok=False)
    source = ROOT / 'artifacts/task01_recovery_head/future_weather_coverage_v1/origin_coverage.parquet'
    payload = source.read_bytes()
    selected = select_seasonal_origins(pd.read_parquet(io.BytesIO(payload)))
    write_json(output / 'selection.json', selected)
    planned = [{**r, 'year': sample['year'], 'quarter': sample['quarter']}
               for sample in selected if sample['issue_utc'] is not None
               for r in sequence_candidates(pd.Timestamp(sample['issue_utc']))]
    write_json(output / 'plan.json', planned)
    records = []
    with requests.Session() as session:
        for row in planned:
            evidence = {'http_status': None, 'etag': None, 'last_modified': None,
                        'checked_at_utc': datetime.now(UTC).isoformat()}
            try:
                with session.head(row['url'], timeout=15, allow_redirects=False) as response:
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
            if len(records) % 24 == 0:
                print(f'{len(records)}/{len(planned)} objects checked', flush=True)
    summaries = []
    for sample in selected:
        subset = [r for r in records if r['issue_utc'] == sample['issue_utc']]
        summaries.append({**sample, 'expected_objects': 24,
                          'checked_objects': len(subset),
                          'available_objects': sum(r['state'] == 'available' for r in subset),
                          'failed_ahead_hours': [r['ahead_hours'] for r in subset if r['state'] != 'available']})
    report = {'selection_rule': 'first_eligible_origin_per_UTC_calendar_quarter_2022_2023',
              'source_sha256': hashlib.sha256(payload).hexdigest(),
              'source_unchanged': source.read_bytes() == payload,
              'summary': summaries, 'records': records,
              'limitations': ['quarterly_sentinel_not_population_coverage', 'HEAD_only',
                              'archive_proxy_not_first_publication', 'no_training_or_test_year_selection',
                              'origins_from_existing_2024_cutoff_panel_inventory']}
    write_json(output / 'report.json', report)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
