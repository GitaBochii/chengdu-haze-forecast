import hashlib
import os
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "satellite_download", Path(__file__).parents[1] / "scripts/download_satellite_sample.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rejects_login_html(tmp_path):
    path = tmp_path / "sample.hdf"
    path.write_bytes(b"<html>Login</html>")
    with pytest.raises(ValueError, match="HDF4"):
        module.inspect_download(path)


def test_hash_and_size(tmp_path):
    payload = b"\x0e\x03\x13\x01" + b"example"
    path = tmp_path / "sample.hdf"
    path.write_bytes(payload)
    result = module.inspect_download(path)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["bytes"] == len(payload)
    assert result["validation"] == "signature_only_not_science_quality"


def test_empty_download_fails():
    with pytest.raises(ValueError, match="No files"):
        module.inspect_downloads([])


def test_gui_input_and_visibility():
    checks = []

    def interact(window, username, password, visible, toggle, submit):
        username.insert(0, "test-user")
        password.insert(0, "test-password")
        checks.append(password.get() == "test-password")
        checks.append(password.cget("show") == "*")
        visible.set(True)
        toggle()
        checks.append(password.cget("show") == "")
        submit()

    assert module.credential_form(interact) == ("test-user", "test-password")
    assert checks == [True, True, True]


@pytest.mark.parametrize("fails", [False, True])
def test_form_auth_restores_environment(monkeypatch, fails):
    monkeypatch.setenv("EARTHDATA_TOKEN", "old-test-token")
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)

    class Client:
        def login(self, **kwargs):
            assert kwargs == {"strategy": "environment", "persist": False}
            assert "EARTHDATA_TOKEN" not in os.environ
            assert os.environ["EARTHDATA_USERNAME"] == "test-user"
            assert os.environ["EARTHDATA_PASSWORD"] == "test-password"
            if fails:
                raise RuntimeError("test failure")
            return "authenticated"

    if fails:
        with pytest.raises(RuntimeError):
            module.login_from_form(Client(), "test-user", "test-password")
    else:
        assert module.login_from_form(Client(), "test-user", "test-password") == "authenticated"
    assert os.environ["EARTHDATA_TOKEN"] == "old-test-token"
    assert "EARTHDATA_PASSWORD" not in os.environ
    assert "EARTHDATA_USERNAME" not in os.environ
