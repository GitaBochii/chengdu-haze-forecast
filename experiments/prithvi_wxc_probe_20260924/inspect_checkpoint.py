"""Inspect official checkpoint metadata via bounded HTTP ranges, without tensor data."""
import io
import json
from pathlib import Path
import time

import requests
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT = Path(__file__).resolve().parent


class RangeReader(io.RawIOBase):
    def __init__(self, url, size, max_bytes=32 * 1024**2):
        self.url, self.size, self.pos = url, size, 0
        self.max_bytes, self.received = max_bytes, 0
        self.session = requests.Session()
        self.deadline = time.monotonic() + 150
        self.cache = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        target = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if target < 0:
            raise ValueError("Negative position")
        self.pos = target
        return target

    def read(self, size=-1):
        n = max(0, self.size-self.pos) if size < 0 else min(size, max(0, self.size-self.pos))
        if n == 0:
            return b""
        for start, block in self.cache:
            if start <= self.pos and self.pos+n <= start+len(block):
                value = block[self.pos-start:self.pos-start+n]
                self.pos += n
                return value
        if self.received+n > self.max_bytes or time.monotonic() > self.deadline:
            raise RuntimeError("Metadata read budget exceeded")
        fetch = min(max(n, 65536), self.size-self.pos, self.max_bytes-self.received)
        with self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos+fetch-1}"},
                              stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError("Server did not honor byte range")
            if not response.headers.get("Content-Range", "").startswith(f"bytes {self.pos}-"):
                raise RuntimeError("Wrong range returned")
            value = response.raw.read(fetch)
            if len(value) != fetch:
                raise RuntimeError("Truncated range")
        self.cache.append((self.pos, value))
        self.received += fetch
        self.pos += n
        return value[:n]

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)


def main():
    revision = json.loads((ROOT / "real_sample_manifest.json").read_text())["revision"]
    url = f"https://huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1/resolve/{revision}/prithvi.wxc.2300m.v1.pt"
    reader = RangeReader(url, 28447290466)
    report = {"revision": revision, "pretrained_loaded": False}
    try:
        with FakeTensorMode():
            checkpoint = torch.load(reader, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        report.update(status="metadata_read", top_level_keys=list(checkpoint.keys()),
                      tensor_shapes={k: list(v.shape) for k,v in state.items() if isinstance(v, torch.Tensor)})
    except Exception as exc:
        report.update(status="metadata_read_failed", error_type=type(exc).__name__)
        if type(exc) is RuntimeError:
            report["error_detail"] = str(exc)[:200]
    report["bytes_read"] = reader.received
    (ROOT / "checkpoint_metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "tensor_shapes"}, indent=2))


if __name__ == "__main__":
    main()
