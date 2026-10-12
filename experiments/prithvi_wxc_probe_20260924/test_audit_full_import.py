import pytest

from audit_full_import import derive_requirements, completed


def candidate():
    return {'timestamp': '2024-01-01T00:00:00Z',
            'support_latest_utc': '2023-12-31T18:00:00Z',
            'support_earlier_utc': '2023-12-31T12:00:00Z',
            'support_end_utc': '2023-12-31T19:00:00Z'}


def test_calendar_and_both_inputs():
    result = derive_requirements([candidate()])
    assert len(result['times']) == 2
    assert result['climate'] == {'climatology/climate_surface_doy001_hour00.nc',
                                 'climatology/climate_vertical_doy001_hour00.nc'}
    assert result['times'][1]['source_right_utc'].endswith('18:30:00+00:00')


def test_rejects_stale_alignment():
    row = candidate()
    row['support_latest_utc'] = '2024-01-01T00:00:00Z'
    with pytest.raises(ValueError):
        derive_requirements([row])


def test_rejects_duplicate_and_empty_origins():
    for rows in ([], [candidate(), candidate()]):
        with pytest.raises(ValueError):
            derive_requirements(rows)


def test_metadata_and_missing_never_complete():
    assert not completed(True, [], 10, 10, 5, 5)
    assert not completed(False, [], 9, 10, 5, 5)
    assert not completed(False, ['invalid'], 10, 10, 5, 5)
    assert completed(False, [], 10, 10, 5, 5)


def test_leap_year_cap_and_day_boundary():
    row = {'timestamp': '2024-12-31T19:00:00Z',
           'support_latest_utc': '2024-12-31T18:00:00Z',
           'support_earlier_utc': '2024-12-31T12:00:00Z',
           'support_end_utc': '2024-12-31T19:00:00Z'}
    assert 'climatology/climate_surface_doy001_hour00.nc' in derive_requirements([row])['climate']
    row.update(timestamp='2024-12-31T13:00:00Z', support_latest_utc='2024-12-31T12:00:00Z',
               support_earlier_utc='2024-12-31T06:00:00Z', support_end_utc='2024-12-31T13:00:00Z')
    assert 'climatology/climate_surface_doy365_hour18.nc' in derive_requirements([row])['climate']


def test_midnight_averaged_uses_prior_date():
    row = {'timestamp': '2024-01-01T07:00:00Z',
           'support_latest_utc': '2024-01-01T06:00:00Z',
           'support_earlier_utc': '2024-01-01T00:00:00Z',
           'support_end_utc': '2024-01-01T07:00:00Z'}
    mapping = derive_requirements([row])['mappings'][0]
    assert 'M2T1NXLND/2023-12-31' in mapping['dynamic']
    assert 'M2T1NXLND/2024-01-01' in mapping['dynamic']
    assert 'M2I1NXASM/2023-12-31' not in mapping['dynamic']
