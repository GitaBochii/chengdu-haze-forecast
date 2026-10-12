import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.verify_weather_training_ready import check_field, check_head, check_plan, verify


def plan():
    issue = pd.Timestamp('2022-01-02T03:00Z')
    run = pd.Timestamp('2022-01-01T18:00Z')
    return pd.DataFrame([{'issue_utc': issue.isoformat(), 'run_utc': run.isoformat(),
                         'valid_utc': (issue + pd.Timedelta(hours=ahead)).isoformat(),
                         'ahead_hours': ahead, 'lead_hours': ahead + 9, 'cycles_back': 1,
                         'url': ('https://noaa-gefs-pds.s3.amazonaws.com/gefs.20220101/18/atmos/'
                                 f'pgrb2ap5/geavg.t18z.pgrb2a.0p50.f{ahead+9:03d}')}
                        for ahead in range(3, 73, 3)])


def test_complete_issue_relative_sequence():
    frame = plan()
    check_plan(frame, (24, 1, 24))
    frame.loc[0, 'ahead_hours'] = 6
    with pytest.raises(ValueError, match='72-hour'):
        check_plan(frame, (24, 1, 24))


@pytest.mark.parametrize('key,value', [('run_utc', '2022-01-01T12:00Z'),
                                      ('valid_utc', '2022-01-02T07:00Z'),
                                      ('lead_hours', 15), ('cycles_back', 5), ('url', 'wrong')])
def test_reject_manifest_time_and_identity_mismatch(key, value):
    frame = plan()
    frame.loc[0, key] = value
    with pytest.raises(ValueError):
        check_plan(frame, (24, 1, 24))


def test_shared_object_must_be_available_at_each_origin():
    row = plan().iloc[0].to_dict()
    head = {'url': row['url'], 'http_status': 200, 'etag': '"' + 'a' * 32 + '"',
            'last_modified': 'Sun, 02 Jan 2022 00:00:00 GMT', 'content_length': '100'}
    check_head(head, row)
    earlier = {**row, 'issue_utc': '2022-01-01T21:00Z'}
    with pytest.raises(ValueError, match='unavailable'):
        check_head(head, earlier)
    with pytest.raises(ValueError, match='ETag'):
        check_head({**head, 'etag': '"abcd-2"'}, row)


def fixture_field(tmp_path):
    path = tmp_path / 'artifacts/task01_recovery_head/object/temperature.grib2'
    path.parent.mkdir(parents=True)
    payload = b'GRIBdata7777'
    path.write_bytes(payload)
    field = {'path': str(path.relative_to(tmp_path)), 'bytes': len(payload),
             'sha256': hashlib.sha256(payload).hexdigest(), 'byte_range': [10, 10 + len(payload) - 1],
             'longitude': 104., 'latitude': 30.5, 'raw_value': 280., 'value': 6.85,
             'raw_unit': 'K', 'unit': 'degC'}
    inventory = '1:0:d=2022010118:TMP:2 m above ground:12 hour fcst:ens mean:vt=2022010206:lon=104,lat=30.5,val=280'
    verbose = '1:0:TMP [K]:ens mean:'
    return field, inventory, verbose


def test_independent_grib_reread(tmp_path):
    field, inventory, verbose = fixture_field(tmp_path)
    def runner(command, **kwargs):
        return SimpleNamespace(stdout=verbose if '-v2' in command else inventory)
    output = check_field(field, 'temperature_c', plan().iloc[0].to_dict(), tmp_path,
                         tmp_path / 'wgrib2.exe', runner)
    assert output[1] == field['sha256']
    with pytest.raises(ValueError, match='converted'):
        check_field({**field, 'value': 99.}, 'temperature_c', plan().iloc[0].to_dict(),
                    tmp_path, tmp_path / 'wgrib2.exe', runner)


@pytest.mark.parametrize('old,new', [('TMP', 'RH'), ('2 m above ground', '10 m above ground'),
                                    ('d=2022010118', 'd=2022010112'), ('vt=2022010206', 'vt=2022010209'),
                                    ('12 hour fcst', '15 hour fcst'), ('ens mean', 'ens spread'),
                                    ('[K]', '[C]'), ('val=280', 'val=281')])
def test_reject_grib_metadata_unit_or_value_errors(tmp_path, old, new):
    field, inventory, verbose = fixture_field(tmp_path)
    def runner(command, **kwargs):
        text = verbose if '-v2' in command else inventory
        return SimpleNamespace(stdout=text.replace(old, new))
    with pytest.raises(ValueError):
        check_field(field, 'temperature_c', plan().iloc[0].to_dict(), tmp_path,
                    tmp_path / 'wgrib2.exe', runner)


def test_failed_reverification_cannot_leave_old_ready_gate(tmp_path):
    collection = tmp_path / 'artifacts/task01_recovery_head/future_weather_collection_v1'
    collection.mkdir(parents=True)
    gate = collection / 'training_readiness_gate.json'
    gate.write_text(json.dumps({'status': 'ready_for_formal_training'}))
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)
    assert not gate.exists()
    previous = list(collection.glob('training_readiness_gate.previous_*.json'))
    assert len(previous) == 1
    assert json.loads(previous[0].read_text())['status'] == 'ready_for_formal_training'
