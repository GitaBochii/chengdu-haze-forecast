"""Map CRC-checked raw storages without deserializing arbitrary pickle objects."""
from pathlib import Path
import sys
import zlib

import torch


def load_tensor(folder, descriptor, record):
    storage = descriptor["storage"]
    key = storage["key"]
    dtypes = {"float32": (torch.float32, 4), "bool": (torch.bool, 1)}
    if not isinstance(key, str) or not key.isascii() or not key.isdigit():
        raise ValueError("Invalid storage key")
    if storage["dtype"] not in dtypes or key != record["key"]:
        raise ValueError("Unsupported or mismatched storage")
    if sys.byteorder != "little":
        raise ValueError("This checkpoint requires little-endian storage")
    dtype, width = dtypes[storage["dtype"]]
    count = storage["elements"]
    shape, stride, offset = descriptor["shape"], descriptor["stride"], descriptor["offset"]
    if len(shape) != len(stride) or any(type(n) is not int or n < 0 for n in [count, offset, *shape, *stride]):
        raise ValueError("Invalid tensor geometry")
    empty = any(n == 0 for n in shape)
    last = offset + sum((n-1)*s for n, s in zip(shape, stride))
    if (not empty and last >= count) or (empty and offset > count):
        raise ValueError("Tensor exceeds storage bounds")
    path = Path(folder) / f"{key}.bin"
    if record["bytes"] != count*width or path.stat().st_size != count*width:
        raise ValueError("Storage size mismatch")
    crc = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024):
            crc = zlib.crc32(chunk, crc)
    if crc != record["crc"]:
        raise ValueError("Storage CRC mismatch")
    # Private mapping keeps later in-place tensor operations off the original file.
    base = torch.from_file(str(path), shared=False, size=count, dtype=dtype)
    return torch.as_strided(base, size=shape, stride=stride, storage_offset=offset)
