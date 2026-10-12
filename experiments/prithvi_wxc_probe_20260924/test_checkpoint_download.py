import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location("downloader", Path(__file__).with_name("download_checkpoint.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("bad_range,bad_hash", [(False, False), (True, False), (False, True)])
def test_resume_is_verified_before_final_name(tmp_path, monkeypatch, bad_range, bad_hash):
    payload = b"checkpoint-test-payload"
    prefix = payload[:5]
    folder = tmp_path / "weights"
    folder.mkdir()
    partial = folder / (module.NAME + ".partial")
    partial.write_bytes(prefix)
    digest = hashlib.sha256(payload).hexdigest() if not bad_hash else "0"*64

    class Response:
        status_code = 206
        headers = {"Content-Range": f"bytes {0 if bad_range else 5}-{len(payload)-1}/{len(payload)}"}
        def raise_for_status(self):
            pass
        def json(self):
            return [{"path": module.NAME, "size": len(payload), "lfs": {"oid": digest}}]
        def iter_content(self, size):
            yield payload[5:]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class Session:
        def get(self, url, **kwargs):
            if "resolve" in url:
                assert kwargs["headers"]["Range"] == f"bytes=5-{len(payload)-1}"
            return Response()

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.requests, "Session", Session)
    monkeypatch.setattr(sys, "argv", ["download_checkpoint.py", "--seconds", "5"])
    module.main()
    report = json.loads((tmp_path / "checkpoint_download.json").read_text())
    if bad_range or bad_hash:
        assert report["status"] == "failed"
        assert not (folder / module.NAME).exists()
    else:
        assert report["status"] == "verified"
        assert (folder / module.NAME).read_bytes() == payload
    if bad_range:
        assert partial.read_bytes() == prefix
