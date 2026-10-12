import zlib
import struct

import pytest
import torch


def fixture_storage(tmp_path):
    raw = bytes(range(8))
    (tmp_path / "1.bin").write_bytes(raw)
    record = {"key": "1", "bytes": 8, "crc": zlib.crc32(raw)}
    descriptor = {"storage": {"key": "1", "dtype": "bool", "elements": 8},
                  "offset": 1, "shape": [2, 2], "stride": [3, 1]}
    return record, descriptor


def test_maps_strided_tensor(tmp_path):
    from storage_loader import load_tensor
    record, descriptor = fixture_storage(tmp_path)
    value = load_tensor(tmp_path, descriptor, record)
    assert value.shape == (2, 2)
    assert value.dtype == torch.bool
    assert value.all()


def test_float_values_and_private_mapping(tmp_path):
    from storage_loader import load_tensor
    raw = struct.pack("<8f", *range(8))
    path = tmp_path / "1.bin"
    path.write_bytes(raw)
    record = {"key": "1", "bytes": 32, "crc": zlib.crc32(raw)}
    descriptor = {"storage": {"key": "1", "dtype": "float32", "elements": 8},
                  "offset": 1, "shape": [2, 2], "stride": [3, 1]}
    value = load_tensor(tmp_path, descriptor, record)
    torch.testing.assert_close(value, torch.tensor([[1., 2.], [4., 5.]]))
    value.add_(100)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("change", ["crc", "bounds", "path", "dtype"])
def test_rejects_invalid_storage(tmp_path, change):
    from storage_loader import load_tensor
    record, descriptor = fixture_storage(tmp_path)
    if change == "crc":
        record["crc"] += 1
    elif change == "bounds":
        descriptor["offset"] = 7
    elif change == "path":
        descriptor["storage"]["key"] = "../1"
    else:
        descriptor["storage"]["dtype"] = "unknown"
    with pytest.raises(ValueError):
        load_tensor(tmp_path, descriptor, record)
