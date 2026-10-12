import json

import numpy as np
import pandas as pd
import pytest

from scripts.extract_operational_gefs_event_features import select_predictable_event_rows


def test_selection_keeps_only_labeled_particle_event_rows_in_the_requested_year():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-12-31 23:00", "2024-01-01 00:00", "2024-01-01 01:00"], utc=True
            ),
            "event_id": [1, 2, 0],
            "remaining_hours": [3.0, 2.0, 1.0],
            "particle_primary": [True, True, True],
        }
    )

    selected = select_predictable_event_rows(frame, year=2024)

    assert len(selected) == 2
    assert selected["timestamp"].dt.tz_convert('Asia/Shanghai').dt.year.eq(2024).all()
    assert selected["event_id"].gt(0).all()
    assert selected["remaining_hours"].notna().all()
    assert selected["particle_primary"].all()


def _extracted_row():
    from haze_forecast.operational_gefs import gefs_data_url
    from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES
    run = pd.Timestamp('2024-01-01T00:00Z')
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
    return {
        'timestamp': run, 'event_id': 1, 'gefs_run_utc': run,
        'gefs_forecast_valid_utc_f006': run + pd.Timedelta(6, unit='h'),
        'gefs_forecast_valid_utc_f024': run + pd.Timedelta(24, unit='h'),
        'gefs_source_manifest_json': json.dumps({'run_utc': run.isoformat(), 'messages': messages}),
        'gefs_extraction_error': '', **{key: 1.0 for key in OPERATIONAL_GEFS_FEATURES},
    }


@pytest.mark.parametrize('failure', ['nan', 'error', 'manifest'])
def test_failed_extraction_is_not_complete_even_with_all_requested_rows(failure):
    from scripts.extract_operational_gefs_event_features import extraction_quality_summary
    row = _extracted_row()
    expected = pd.DataFrame([row])
    if failure == 'nan':
        row['gefs_mean_temperature_c_f006'] = np.nan
    elif failure == 'error':
        row['gefs_extraction_error'] = 'download failed'
    else:
        row['gefs_source_manifest_json'] = '{}'
    summary = extraction_quality_summary(pd.DataFrame([row]), expected, expected)
    assert summary['coverage']['complete_rows'] == 0
    assert summary['coverage']['is_complete'] is False
    assert summary['run_valid_order_safe'] is True
    assert summary['publication_availability_verified'] is False
    assert 'timing_safe' not in summary


def test_verified_shard_is_complete_only_when_it_covers_the_full_source():
    from scripts.extract_operational_gefs_event_features import extraction_quality_summary
    frame = pd.DataFrame([_extracted_row()])
    complete = extraction_quality_summary(frame, frame, frame)
    assert complete['coverage'] == {'complete_rows': 1, 'is_complete': True}
    source = pd.concat([frame, frame.assign(event_id=2)], ignore_index=True)
    partial = extraction_quality_summary(frame, frame, source)
    assert partial['coverage'] == {'complete_rows': 1, 'is_complete': False}


def test_missing_or_wrong_produced_keys_cannot_be_complete():
    from scripts.extract_operational_gefs_event_features import extraction_quality_summary
    expected = pd.DataFrame([_extracted_row()])
    wrong = expected.assign(event_id=2)
    result = extraction_quality_summary(wrong, expected, expected)
    assert result['coverage'] == {'complete_rows': 0, 'is_complete': False}
