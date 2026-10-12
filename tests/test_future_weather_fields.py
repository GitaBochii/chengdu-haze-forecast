from typing import ClassVar

import pandas as pd
import pytest

from scripts.extract_future_weather_pilot import (
    bounded_index,
    checked_payload,
    extract_object,
    validate_inventory,
    validate_units,
)


def test_extraction_uses_injected_http_client(tmp_path, monkeypatch):
    class HeadResponse:
        status_code = 500
        headers: ClassVar[dict] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Client:
        called = False

        def head(self, *args, **kwargs):
            self.called = True
            return HeadResponse()

    def forbidden(*args, **kwargs):
        raise AssertionError('global client used')

    monkeypatch.setattr('scripts.extract_future_weather_pilot.requests.head', forbidden)
    client = Client()
    row = {'issue_utc': '2022-01-02T03:00Z', 'run_utc': '2022-01-01T18:00Z',
           'valid_utc': '2022-01-02T06:00Z', 'lead_hours': 12, 'etag': 'tag',
           'url': 'https://noaa-gefs-pds.s3.amazonaws.com/gefs.20220101/18/atmos/pgrb2ap5/geavg.t18z.pgrb2a.0p50.f012'}
    with pytest.raises(ValueError, match='archive proxy'):
        extract_object(row, tmp_path, tmp_path / 'unused.exe', http=client)
    assert client.called


class Response:
    status_code = 206
    headers: ClassVar[dict] = {'Content-Range': 'bytes 10-13/100', 'ETag': 'tag'}

    def iter_content(self, chunk_size):
        yield b'ab'
        yield b'cd'


@pytest.mark.parametrize('start,end,total', [(-1, 2, 100), (98, 101, 100), (0, 2000000, 3000000)])
def test_reject_out_of_object_ranges(start, end, total):
    response = Response()
    response.headers = {'Content-Range': f'bytes {start}-{end}/{total}', 'ETag': 'tag'}
    with pytest.raises(ValueError):
        checked_payload(response, start, end, total, 'tag')


def test_index_has_streamed_byte_budget():
    response = Response()
    response.status_code = 200
    assert bounded_index(response, limit=4) == 'abcd'
    with pytest.raises(ValueError):
        bounded_index(response, limit=3)
    response.status_code = 302
    with pytest.raises(ValueError):
        bounded_index(response)


def test_exact_bounded_range():
    assert checked_payload(Response(), 10, 13, 100, 'tag') == b'abcd'


@pytest.mark.parametrize('status,headers', [
    (200, Response.headers),
    (206, {'Content-Range': 'bytes 10-14/100', 'ETag': 'tag'}),
    (206, {'Content-Range': 'bytes 10-13/101', 'ETag': 'tag'}),
    (206, {'Content-Range': 'bytes 10-13/100', 'ETag': 'changed'}),
])
def test_reject_wrong_range_or_object(status, headers):
    response = Response()
    response.status_code, response.headers = status, headers
    with pytest.raises(ValueError):
        checked_payload(response, 10, 13, 100, 'tag')


@pytest.mark.parametrize('payload', [b'abc', b'abcde'])
def test_reject_wrong_body_length(payload):
    response = Response()
    response.iter_content = lambda chunk_size: iter([payload])
    with pytest.raises(ValueError):
        checked_payload(response, 10, 13, 100, 'tag')


@pytest.mark.parametrize('line,ok', [
    ('1:0:d=2022010118:TMP:2 m above ground:12 hour fcst:vt=2022010206:', True),
    ('1:0:d=2022010118:TMP:2 m above ground:30 hour fcst:vt=2022010206:', False),
    ('1:0:d=2022010118:RH:2 m above ground:12 hour fcst:vt=2022010206:', False),
    ('1:0:d=2022010118:TMP:2 m above ground:12 hour fcst:vt=2022010209:', False),
])
def test_grib_inventory_identity(line, ok):
    args = (line, pd.Timestamp('2022-01-01T18:00Z'),
            pd.Timestamp('2022-01-02T06:00Z'), 12, 'TMP', '2 m above ground')
    if ok:
        validate_inventory(*args)
    else:
        with pytest.raises(ValueError):
            validate_inventory(*args)


@pytest.mark.parametrize('text,ok', [
    ('1:0:TMP Temperature [K]:2 m above ground:ens mean', True),
    ('1:0:TMP Temperature [degC]:2 m above ground:ens mean', False),
    ('1:0:TMP Temperature [K]:2 m above ground:ens std dev', False),
])
def test_units_and_ensemble_product(text, ok):
    if ok:
        validate_units(text, 'K')
    else:
        with pytest.raises(ValueError):
            validate_units(text, 'K')
