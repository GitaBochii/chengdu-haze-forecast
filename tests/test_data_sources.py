from pathlib import Path

import pytest
import requests

from haze_forecast import data_sources
from haze_forecast.data_sources import choose_archive_file, download_figshare_archive


def test_choose_archive_file_prefers_largest_supported_archive():
    files = [
        {"name": "readme.txt", "download_url": "https://example.com/readme", "size": 12},
        {"name": "small.zip", "download_url": "https://example.com/small", "size": 100},
        {"name": "air_data.rar", "download_url": "https://example.com/archive", "size": 200},
    ]

    assert choose_archive_file(files) == files[2]


def test_failed_download_does_not_write_a_source_manifest(monkeypatch, tmp_path: Path):
    def fail_request(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(data_sources.requests, "get", fail_request)

    with pytest.raises(requests.ConnectionError, match="offline"):
        download_figshare_archive(tmp_path)

    assert not (tmp_path / "source_manifest.json").exists()
    assert not list(tmp_path.glob("*.part"))
