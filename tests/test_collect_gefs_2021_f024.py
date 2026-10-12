import json
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from scripts import collect_gefs_2021_f024 as collector


def test_plan_uses_local_year_and_utc_four_hour_lag():
    frame = pd.DataFrame({'timestamp': pd.to_datetime([
        '2020-12-31 23:00+08:00', '2021-01-01 00:00+08:00',
        '2021-01-02 08:00+08:00', '2021-01-02 08:00+08:00',
        '2022-01-01 00:00+08:00'])})
    assert collector.plan_runs(frame) == [
        pd.Timestamp('2020-12-31 12:00Z'), pd.Timestamp('2021-01-01 18:00Z')]


def test_protocol_mismatch_never_overwrites(tmp_path):
    collector.freeze_protocol(tmp_path, {'runs': ['a']})
    with pytest.raises(ValueError, match='protocol'):
        collector.freeze_protocol(tmp_path, {'runs': ['b']})
    assert json.loads((tmp_path / 'protocol.json').read_text()) == {'runs': ['a']}


def test_network_failure_retries_and_keeps_each_log(tmp_path):
    calls = []
    sleeps = []

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ReadTimeout('transient')

    collector.with_retries(operation, tmp_path, max_retries=3, sleep=sleeps.append)
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert len(list(tmp_path.glob('attempt-*.json'))) == 2


@pytest.mark.parametrize('error', [AssertionError('bad range'), ValueError('bad metadata')])
def test_validation_failure_does_not_retry(tmp_path, error):
    calls = []

    def operation():
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        collector.with_retries(operation, tmp_path, max_retries=3, sleep=lambda _: None)
    assert len(calls) == 1
    assert len(list(tmp_path.glob('attempt-*.json'))) == 1


def test_network_retry_exhaustion_is_failure(tmp_path):
    def operation():
        raise requests.exceptions.ConnectionError('offline')

    with pytest.raises(requests.exceptions.ConnectionError):
        collector.with_retries(operation, tmp_path, max_retries=2, sleep=lambda _: None)
    assert len(list(tmp_path.glob('attempt-*.json'))) == 3


def complete_fixture(folder):
    run = pd.Timestamp('2021-01-02T00:00Z')
    products = {product: {'fields': {field: {} for field in fields}}
                for product, fields in collector.PRODUCT_FIELDS.items()}
    for name in collector.expected_files():
        (folder / name).write_bytes(b'fixture')
    report = {'run_at': run.isoformat(), 'valid_at': (run + pd.Timedelta('24h')).isoformat(),
              'products': products,
              'output_sha256': {name: collector.sha(folder / name)
                                for name in collector.expected_files()}}
    collector.atomic_json(folder / 'complete.json', report)
    return run


def test_resume_requires_exact_file_and_product_sets(tmp_path):
    run = complete_fixture(tmp_path)
    collector.verify_complete(tmp_path, run)
    (tmp_path / 'geavg.idx').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        collector.verify_complete(tmp_path, run)


def test_resume_rejects_incomplete_manifest(tmp_path):
    run = complete_fixture(tmp_path)
    path = tmp_path / 'complete.json'
    report = json.loads(path.read_text())
    report['products']['geavg']['fields'].pop('temperature_c')
    collector.atomic_json(path, report)
    with pytest.raises(ValueError):
        collector.verify_complete(tmp_path, run)


def test_resume_rejects_changed_file(tmp_path):
    run = complete_fixture(tmp_path)
    (tmp_path / 'geavg.idx').write_bytes(b'changed')
    with pytest.raises(ValueError):
        collector.verify_complete(tmp_path, run)


def mock_extraction(monkeypatch, defect=None):
    definitions = list(collector.DEFINITIONS.values()) + [('HPBL', 'surface')]
    entries = [SimpleNamespace(variable=v, level=l) for v, l in definitions]
    monkeypatch.setattr(collector.forecast_sources, 'parse_gfs_index', lambda _: entries)
    monkeypatch.setattr(collector.forecast_sources, 'range_for_gfs_index_entry',
                        lambda _entries, _match: (0, 19))
    content = b'GRIB' + b'\x00' * 3 + b'\x02' + (20).to_bytes(8, 'big') + b'7777'

    def get(url, **_kwargs):
        if url.endswith('.idx'):
            return SimpleNamespace(status_code=200, content=b'index', text='index',
                                   raise_for_status=lambda: None)
        return SimpleNamespace(status_code=200 if defect == 'status' else 206,
                               content=content, raise_for_status=lambda: None,
                               headers={'Content-Range': 'bytes 0-19/100', 'ETag': 'one',
                                        'Last-Modified': 'Sat, 02 Jan 2021 04:10:00 GMT'})

    def decode(command, **_kwargs):
        name = command[1].split('\\')[-1].split('/')[-1]
        product, field = name.removesuffix('.grb2').split('_', 1)
        variable, level = collector.PRODUCT_FIELDS[product][field]
        identity = {'geavg': 'ens mean', 'gespr': 'ens std dev', 'gec00': 'ENS=low-res ctl'}[product]
        lead = 72 if defect == 'lead' else 24
        return SimpleNamespace(stdout=f'1:0:d=2021010200:{variable}:{level}:{lead} hour fcst:'
                               f'{identity}:vt=2021010300:')

    def point(path, *_args):
        product, field = path.stem.split('_', 1)
        value = 280 if product == 'geavg' and field == 'temperature_c' else 10
        return SimpleNamespace(value=value, longitude=0 if defect == 'grid' else 104,
                               latitude=30.5)

    monkeypatch.setattr(collector.requests, 'get', get)
    monkeypatch.setattr(collector.subprocess, 'run', decode)
    monkeypatch.setattr(collector.grib, 'query_grib_point', point)


def test_extraction_seals_nine_fields_and_correct_units(tmp_path, monkeypatch):
    mock_extraction(monkeypatch)
    run = pd.Timestamp('2021-01-02 00:00Z')
    collector.extract(run, tmp_path, tmp_path / 'decoder')
    report = collector.verify_complete(tmp_path, run)
    assert report['products']['geavg']['fields']['temperature_c']['value'] == pytest.approx(6.85)
    assert report['products']['gespr']['fields']['temperature_c']['value'] == 10
    assert report['archive_available_at_proxy'] == '2021-01-02T04:10:00+00:00'
    assert len(report['output_sha256']) == 12


@pytest.mark.parametrize('defect', ['status', 'lead', 'grid'])
def test_extraction_validation_never_seals_bad_batch(tmp_path, monkeypatch, defect):
    mock_extraction(monkeypatch, defect)
    with pytest.raises(ValueError):
        collector.extract(pd.Timestamp('2021-01-02 00:00Z'), tmp_path, tmp_path / 'decoder')
    assert not (tmp_path / 'complete.json').exists()
