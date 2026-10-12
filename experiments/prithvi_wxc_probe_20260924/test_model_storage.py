import importlib.util
from pathlib import Path
import struct
import zlib

import pytest

spec = importlib.util.spec_from_file_location("storage", Path(__file__).with_name("download_model_storage.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(payload=b"real-values", compression=0):
    name = b"step_400/data/12"
    header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 20, 0, compression, 0, 0,
                         zlib.crc32(payload), len(payload), len(payload), len(name), 0)
    return header+name+payload+b"trailing-descriptor"


def test_record_data_and_crc():
    item = {"filename": "step_400/data/12", "bytes": 11, "crc": zlib.crc32(b"real-values")}
    assert module.extract_record(record(), item) == b"real-values"
    with pytest.raises(ValueError):
        module.extract_record(record(b"fake-values"), item)
    with pytest.raises(ValueError):
        module.extract_record(record(compression=8), item)
    with pytest.raises(ValueError):
        module.extract_record(record()[:32], item)
