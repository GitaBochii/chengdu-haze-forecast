import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("checkpoint", Path(__file__).with_name("inspect_checkpoint.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response:
    def __init__(self, status=206, start=2):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-4/10"}
        self.raw = io.BytesIO(b"abc")
    def raise_for_status(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def test_range_and_budget(monkeypatch):
    reader = module.RangeReader("https://example.invalid", 10, max_bytes=3)
    monkeypatch.setattr(reader.session, "get", lambda *a, **kw: Response())
    reader.seek(-8, 2)
    assert reader.read(3) == b"abc"
    assert reader.tell() == 5
    with pytest.raises(RuntimeError, match="budget"):
        reader.read(1)


@pytest.mark.parametrize("response", [Response(200), Response(206, 0)])
def test_rejects_full_or_wrong_range(monkeypatch, response):
    reader = module.RangeReader("https://example.invalid", 10)
    monkeypatch.setattr(reader.session, "get", lambda *a, **kw: response)
    reader.seek(2)
    with pytest.raises(RuntimeError):
        reader.read(3)


def test_prefetched_bytes_do_not_repeat_network_requests(monkeypatch):
    reader = module.RangeReader("https://example.invalid", 10)
    calls = []
    response = Response(206, 0)
    response.raw = io.BytesIO(b"0123456789")
    def get(*args, **kwargs):
        calls.append(kwargs["headers"]["Range"])
        return response
    monkeypatch.setattr(reader.session, "get", get)
    assert reader.read(2) == b"01"
    reader.seek(5)
    assert reader.read(3) == b"567"
    assert calls == ["bytes=0-9"]
