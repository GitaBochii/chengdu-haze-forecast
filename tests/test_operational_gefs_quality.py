import json

import numpy as np
import pandas as pd
import pytest

from haze_forecast.operational_gefs import gefs_data_url
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES
from haze_forecast.operational_gefs_quality import consolidate_verified_shards


def valid_row(hour=1):
    run = pd.Timestamp('2023-01-03T00:00:00Z')
    messages = []
    for lead in (6, 24):
        for product in ('geavg', 'gespr', 'gec00'):
            fields = ('pblh',) if product == 'gec00' else (
                'temperature_2m', 'relative_humidity_2m', 'u_wind_10m', 'v_wind_10m')
            url = gefs_data_url(run.to_pydatetime(), product=product, lead_hours=lead)
            for field in fields:
                messages.append({'product': product, 'field': field, 'lead_hours': lead,
                                 'source_url': url, 'index_url': url + '.idx',
                                 'sha256': 'a' * 64, 'byte_range': [0, 99]})
    return dict(timestamp=run + pd.Timedelta(hour, unit='h'), event_id=1,
                gefs_run_utc=run, gefs_forecast_valid_utc_f006=run + pd.Timedelta(6, unit='h'),
                gefs_forecast_valid_utc_f024=run + pd.Timedelta(24, unit='h'),
                gefs_source_manifest_json=json.dumps({'run_utc': run.isoformat(), 'messages': messages}),
                gefs_extraction_error='', **{key: 1.0 for key in OPERATIONAL_GEFS_FEATURES})


def test_success_retry_wins_and_utc_keys_are_normalized():
    good = valid_row()
    failed = {**good, OPERATIONAL_GEFS_FEATURES[0]: np.nan, 'gefs_extraction_error': 'timeout'}
    good['timestamp'] = good['timestamp'].tz_convert('Asia/Shanghai')
    result = consolidate_verified_shards([pd.DataFrame([failed]), pd.DataFrame([good])])
    assert len(result) == 1
    assert result.gefs_row_verified.all()
    assert str(result.timestamp.dt.tz) == 'UTC'


@pytest.mark.parametrize('error', ['', np.nan, None])
def test_empty_error_requires_finite_features_and_complete_manifest(error):
    row = valid_row()
    row['gefs_extraction_error'] = error
    assert consolidate_verified_shards([pd.DataFrame([row])]).gefs_row_verified.all()
    row[OPERATIONAL_GEFS_FEATURES[0]] = np.inf
    assert not consolidate_verified_shards([pd.DataFrame([row])]).gefs_row_verified.any()
    row[OPERATIONAL_GEFS_FEATURES[0]] = 1.
    row['gefs_source_manifest_json'] = '{}'
    assert not consolidate_verified_shards([pd.DataFrame([row])]).gefs_row_verified.any()


@pytest.mark.parametrize('change', ['value', 'sha'])
def test_conflicting_successful_retries_are_rejected(change):
    first, second = valid_row(), valid_row()
    if change == 'value':
        second[OPERATIONAL_GEFS_FEATURES[0]] = 2.
    else:
        manifest = json.loads(second['gefs_source_manifest_json'])
        manifest['messages'][0]['sha256'] = 'b' * 64
        second['gefs_source_manifest_json'] = json.dumps(manifest)
    with pytest.raises(ValueError, match='conflicting'):
        consolidate_verified_shards([pd.DataFrame([first, second])])


def test_message_order_and_equivalent_success_duplicates_are_allowed():
    first, second = valid_row(), valid_row()
    manifest = json.loads(second['gefs_source_manifest_json'])
    manifest['messages'].reverse()
    second['gefs_source_manifest_json'] = json.dumps(manifest)
    assert len(consolidate_verified_shards([pd.DataFrame([first, second])])) == 1


@pytest.mark.parametrize('change', ['duplicate', 'url', 'sha', 'range', 'run', 'valid'])
def test_invalid_source_identity_or_timing_never_counts_complete(change):
    row = valid_row()
    manifest = json.loads(row['gefs_source_manifest_json'])
    if change == 'duplicate':
        manifest['messages'][0] = manifest['messages'][1]
    elif change == 'url':
        manifest['messages'][0]['source_url'] = 'https://example.org/not-noaa'
    elif change == 'sha':
        manifest['messages'][0]['sha256'] = 'invalid'
    elif change == 'range':
        manifest['messages'][0]['byte_range'] = [100, 2]
    elif change == 'run':
        manifest['run_utc'] = '2023-01-02T00:00Z'
    else:
        row['gefs_forecast_valid_utc_f006'] = row['timestamp']
    row['gefs_source_manifest_json'] = json.dumps(manifest)
    result = consolidate_verified_shards([pd.DataFrame([row])])
    assert not result.gefs_row_verified.any()
    assert result.gefs_quality_error.str.len().gt(0).all()


def test_naive_or_duplicate_invalid_keys_fail_before_quality_checks():
    row = valid_row()
    row['timestamp'] = row['timestamp'].tz_localize(None)
    with pytest.raises(ValueError, match='timezone-aware'):
        consolidate_verified_shards([pd.DataFrame([row])])
    row = valid_row()
    row['event_id'] = 1.5
    with pytest.raises(ValueError, match='positive integers'):
        consolidate_verified_shards([pd.DataFrame([row])])


def test_repair_retries_only_failed_cycle_and_preserves_attempts(tmp_path):
    from scripts.repair_operational_gefs_features import repair_year
    good, failed = valid_row(1), valid_row(2)
    failed['gefs_extraction_error'] = 'timeout'
    failed[OPERATIONAL_GEFS_FEATURES[0]] = np.nan
    expected = pd.DataFrame([good, valid_row(2)])
    calls, sleeps = [], []

    def extract(rows):
        calls.append(rows.timestamp.tolist())
        return pd.DataFrame([failed if len(calls) == 1 else valid_row(2)])

    report = repair_year(expected, [pd.DataFrame([good, failed])], year=2023,
                         output_dir=tmp_path / 'shards', manifest_dir=tmp_path / 'audit',
                         extractor=extract, sleeper=sleeps.append, backoff_seconds=1)
    assert len(calls) == 2 and all(len(call) == 1 for call in calls)
    assert sleeps == [1]
    assert report['verified_keys'] == report['expected_keys'] == 2
    assert len(report['successful_runs']) == 1 and report['failed_runs'] == []
    assert len(list((tmp_path / 'shards').glob('*.parquet'))) == 2
    first_report = report.copy()
    repair_year(expected, [pd.DataFrame([good, failed])], year=2023,
                output_dir=tmp_path / 'shards', manifest_dir=tmp_path / 'audit',
                extractor=lambda rows: pd.DataFrame([valid_row(2)]), sleeper=sleeps.append)
    assert len(list((tmp_path / 'shards').glob('*.parquet'))) == 3
    assert first_report['verified_keys'] == 2


def test_repair_caps_attempts_and_does_not_claim_failed_rows_complete(tmp_path):
    from scripts.repair_operational_gefs_features import repair_year
    calls, sleeps = [], []

    def fail(rows):
        calls.append(len(rows))
        raise OSError('offline')

    report = repair_year(pd.DataFrame([valid_row()]), [], year=2023,
                         output_dir=tmp_path / 'shards', manifest_dir=tmp_path / 'audit',
                         extractor=fail, sleeper=sleeps.append, backoff_seconds=2)
    assert calls == [1, 1, 1] and sleeps == [2, 4]
    assert report['verified_keys'] == 0 and len(report['failed_runs']) == 1
    assert len(list((tmp_path / 'shards').glob('*.parquet'))) == 3
    with pytest.raises(ValueError, match='attempts'):
        repair_year(pd.DataFrame([valid_row()]), [], year=2023, output_dir=tmp_path,
                    manifest_dir=tmp_path, extractor=fail, max_attempts=4)
