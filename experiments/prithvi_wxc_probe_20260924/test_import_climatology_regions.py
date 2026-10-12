import h5py
import numpy as np
import pytest
from import_climatology_regions import validate_arrays, cache_valid, atomic_json, save_region, extract, RangeFetcher, LAT, LON


def sample():
    return {'lat': LAT.copy(), 'lon': LON.copy(), 'T2M': np.ones((30, 32), dtype='float32')}


def test_rejects_invalid_coordinates_shapes_and_nonfinite():
    validate_arrays(sample(), ['T2M'], None)
    for key, value in [('lat', LAT + 1), ('T2M', np.ones((32, 30))),
                       ('T2M', np.full((30, 32), np.nan))]:
        arrays = sample()
        arrays[key] = value
        with pytest.raises(ValueError):
            validate_arrays(arrays, ['T2M'], None)


def test_vertical_preserves_original_level_order():
    levels = np.array([72, 34])
    arrays = {'lat': LAT, 'lon': LON, 'lev': levels, 'T': np.ones((2, 30, 32))}
    validate_arrays(arrays, ['T'], [34, 72])
    arrays['lev'] = np.array([34, 34])
    with pytest.raises(ValueError):
        validate_arrays(arrays, ['T'], [34, 72])


def test_atomic_cache_checks_bytes_provenance_and_decoded_values(tmp_path):
    path = tmp_path / 'sample.npz'
    entry = {'path': 'climatology/climate_surface_doy001_hour00.nc',
             'size': 123, 'lfs': {'oid': 'a' * 64}}
    record = save_region(path, sample(), entry, 'config-sha', 'remote_range', 500, 2)
    assert cache_valid(path, entry, 'config-sha', ['T2M'], None)
    assert not cache_valid(path, entry, 'changed', ['T2M'], None)
    record['source_sha256_metadata'] = 'b' * 64
    atomic_json(path.with_suffix('.json'), record)
    assert not cache_valid(path, entry, 'config-sha', ['T2M'], None)
    save_region(path, sample(), entry, 'config-sha', 'remote_range', 500, 2)
    path.write_bytes(b'bad')
    assert not cache_valid(path, entry, 'config-sha', ['T2M'], None)


@pytest.mark.parametrize('vertical', [False, True])
def test_range_extraction_exactly_matches_local_slices(tmp_path, vertical):
    path = tmp_path / 'grid.h5'
    levels = list(range(14)) if vertical else None
    shape = (14, 361, 576) if vertical else (361, 576)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with h5py.File(path, 'w') as handle:
        handle['lat'] = np.arange(-90, 90.5, .5)
        handle['lon'] = np.arange(-180, 180, .625)
        if vertical:
            handle['lev'] = levels[::-1]
        handle['T'] = values
    payload = path.read_bytes()
    def fetch(start, end):
        return payload[start:end + 1]
    with h5py.File(path, 'r') as handle:
        direct = extract(handle, ['T'], levels)
        ranged = extract(handle, ['T'], levels, fetch, len(payload))
    for key in direct:
        np.testing.assert_array_equal(direct[key], ranged[key])


class FakeResponse:
    def __init__(self, status, url, start=0, end=1):
        self.status_code, self.url = status, url
        self.headers = {'Content-Range': f'bytes {start}-{end}/10'}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_content(self, **kwargs):
        yield b'ab'


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.urls = []
    def get(self, url, **kwargs):
        self.urls.append(url)
        return next(self.responses)


def test_signed_url_reuse_and_expiry_return_to_official(monkeypatch):
    monkeypatch.setattr('import_climatology_regions.time.sleep', lambda _: None)
    signed = 'https://us.aws.cdn.hf.co/path?secret=do-not-persist'
    replacement = 'https://us.aws.cdn.hf.co/path?secret=new'
    session = FakeSession([FakeResponse(206, signed), FakeResponse(403, signed),
                           FakeResponse(206, replacement, 2, 3)])
    fetch = RangeFetcher(session, {'path': 'climatology/test.nc', 'size': 10})
    assert fetch(0, 1) == b'ab'
    assert fetch(2, 3) == b'ab'
    assert session.urls[0] == session.urls[2]
    assert session.urls[1] == signed
    assert fetch.requests == 3 and fetch.body_bytes == 4


def test_signed_url_rejects_untrusted_redirect_without_leaking(monkeypatch):
    monkeypatch.setattr('import_climatology_regions.time.sleep', lambda _: None)
    session = FakeSession([FakeResponse(206, 'https://evil.invalid/?secret=x')] * 3)
    fetch = RangeFetcher(session, {'path': 'climatology/test.nc', 'size': 10})
    with pytest.raises(RuntimeError, match='three attempts') as error:
        fetch(0, 1)
    assert 'secret' not in str(error.value)
    assert all(url.startswith('https://huggingface.co/') for url in session.urls)
