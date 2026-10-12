"""Small HEAD-only probe of earlier GEFS cycles at unchanged forecast valid times."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pandas as pd
import requests

from haze_forecast.operational_gefs_availability import availability_state
from scripts.benchmark_recovery_selection import ROOT
from scripts.benchmark_regional_air_models import write_json


def cycle_candidate(issue, latest_run, original_lead, back):
    run = latest_run - pd.Timedelta(hours=6 * back)
    valid = latest_run + pd.Timedelta(hours=original_lead)
    lead = int((valid - run).total_seconds() / 3600)
    if run > issue or valid <= issue or lead % 3:
        raise ValueError('invalid cycle or target-valid alignment')
    url = (f'https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run:%Y%m%d}/{run:%H}/atmos/'
           f'pgrb2ap5/geavg.t{run:%H}z.pgrb2a.0p50.f{lead:03d}')
    return run, valid, lead, url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    coverage = pd.read_parquet(ROOT / 'artifacts/task01_recovery_head/future_weather_coverage_v1/origin_coverage.parquet')
    samples = coverage.loc[coverage.matched & coverage.year.isin([2022, 2023])].sort_values('timestamp').groupby('year').head(1)
    records = []
    for row in samples.itertuples(index=False):
        issue, latest = pd.Timestamp(row.timestamp), pd.Timestamp(row.gefs_run_utc)
        for original_lead in (6, 24):
            for back in (0, 1):
                run, valid, lead, url = cycle_candidate(issue, latest, original_lead, back)
                evidence = {'url': url, 'checked_at_utc': datetime.now(UTC).isoformat(),
                            'http_status': None, 'etag': None, 'last_modified': None}
                try:
                    response = requests.head(url, timeout=15, allow_redirects=False)
                    evidence.update(http_status=response.status_code, etag=response.headers.get('ETag'),
                                    last_modified=response.headers.get('Last-Modified'),
                                    content_length=response.headers.get('Content-Length'))
                except requests.RequestException as exc:
                    evidence['error'] = str(exc)
                state, reason = availability_state(evidence, issue, run)
                records.append({'year': row.year, 'issue_utc': issue.isoformat(),
                                'run_utc': run.isoformat(), 'valid_utc': valid.isoformat(),
                                'lead_hours': lead, 'cycles_back': back,
                                'ahead_hours': (valid - issue).total_seconds() / 3600,
                                'state': state, 'reason': reason, **evidence})
                write_json(output / 'progress.json', records)
    write_json(output / 'report.json', {'selection': 'first_matched_neural_origin_per_year_2022_2023',
                                        'records': records, 'limitations': [
                                            'HEAD_only_no_GRIB_values_or_field_validation',
                                            'archive_proxy_not_historical_first_publication',
                                            'two_origins_not_full_coverage_or_performance_evidence']})
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
