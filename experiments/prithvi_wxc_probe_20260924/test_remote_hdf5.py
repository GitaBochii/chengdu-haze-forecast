import io
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_hdf5 import RangeFile


def test_seek_read_cache_and_end():
    source = bytes(range(100))
    calls = []
    def fetch(start, end):
        calls.append((start, end))
        return source[start:end + 1]
    with RangeFile(100, fetch, block_size=16) as stream:
        stream.seek(14)
        assert stream.read(6) == source[14:20]
        stream.seek(14)
        assert stream.read(6) == source[14:20]
        assert calls == [(0, 15), (16, 31)]
        assert stream.seek(-3, 2) == 97
        assert stream.read(8) == source[97:]
        assert stream.read() == b''
        with pytest.raises(ValueError):
            stream.seek(-1)


def test_short_fetch_fails():
    with RangeFile(100, lambda a, b: b'x', block_size=16) as stream:
        with pytest.raises(ValueError):
            stream.read(2)


def test_hdf5_metadata_and_slice_without_full_file():
    buffer = io.BytesIO()
    values = np.arange(200000, dtype='float32').reshape(400, 500)
    with h5py.File(buffer, 'w') as handle:
        handle.create_dataset('field', data=values)
    source = buffer.getvalue()
    with RangeFile(len(source), lambda a, b: source[a:b + 1], block_size=4096) as stream:
        with h5py.File(stream, 'r') as handle:
            assert handle['field'].shape == (400, 500)
            assert np.array_equal(handle['field'][10:12, 20:24], values[10:12, 20:24])
        assert stream.bytes_fetched < len(source) / 10


def test_budget_limits_network_before_fetch():
    calls = []
    def fetch(a, b):
        calls.append((a, b))
        return bytes(b-a+1)
    with RangeFile(100, fetch, block_size=16, max_bytes=16) as stream:
        assert stream.read(16) == bytes(16)
        with pytest.raises(ValueError):
            stream.read(1)
        assert len(calls) == 1
