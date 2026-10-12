import hashlib
import time

import pytest

import prepare_real_sample as module


class Response:
    status_code = 200
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield b"data"


class Session:
    def get(self, *args, **kwargs):
        return Response()


def item(digest=None):
    return {"path": "merra-2/test.nc", "size": 4,
            "lfs": {"oid": digest or hashlib.sha256(b"data").hexdigest()}}


def test_bad_hash_never_promotes_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        module.download(Session(), "fixed", item("0"*64), time.monotonic()+10)
    assert not (tmp_path / "real_sample/merra-2/test.nc").exists()
    assert (tmp_path / "real_sample/merra-2/test.nc.partial").exists()


def test_complete_partial_verifies_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    partial = tmp_path / "real_sample/merra-2/test.nc.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"data")
    result = module.download(None, "fixed", item(), time.monotonic()+10)
    assert result["upstream_sha256_verified"]
    assert partial.with_suffix("").read_bytes() == b"data"


def test_valid_download_and_bad_existing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.download(Session(), "fixed", item(), time.monotonic()+10)["bytes"] == 4
    target = tmp_path / "real_sample/merra-2/test.nc"
    target.write_bytes(b"evil")
    with pytest.raises(ValueError, match="SHA256"):
        module.download(None, "fixed", item(), time.monotonic()+10)


@pytest.mark.parametrize("correct", [True, False])
def test_resume_range_is_exact(tmp_path, monkeypatch, correct):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    partial = tmp_path / "real_sample/merra-2/test.nc.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"da")

    class Resumed(Response):
        status_code = 206
        headers = {"Content-Range": "bytes 2-3/4" if correct else "bytes 2-3/99"}

        def iter_content(self, size):
            yield b"ta"

    class ResumeSession:
        def get(self, url, **kwargs):
            assert kwargs["headers"] == {"Range": "bytes=2-3"}
            return Resumed()

    if correct:
        assert module.download(ResumeSession(), "fixed", item(), time.monotonic()+10)["upstream_sha256_verified"]
    else:
        with pytest.raises(ValueError, match="range"):
            module.download(ResumeSession(), "fixed", item(), time.monotonic()+10)
        assert partial.read_bytes() == b"da"


def test_oversized_response_cannot_extend_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    value = item()
    value["size"] = 3
    with pytest.raises(ValueError, match="exceeds"):
        module.download(Session(), "fixed", value, time.monotonic()+10)
    assert not (tmp_path / "real_sample/merra-2/test.nc").exists()
