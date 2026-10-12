"""下载和记录可追溯的原始数据。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

FIGSHARE_ARTICLE_URL = "https://api.figshare.com/v2/articles/24152163"
SUPPORTED_ARCHIVE_SUFFIXES = (".zip", ".7z", ".rar", ".tar.gz", ".tgz")


def choose_archive_file(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the largest archive entry exposed by the Figshare article."""
    archives = [
        item
        for item in files
        if str(item.get("name", "")).lower().endswith(SUPPORTED_ARCHIVE_SUFFIXES)
    ]
    if not archives:
        raise ValueError("Figshare metadata contains no supported archive file")
    return max(archives, key=lambda item: int(item.get("size", 0)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _license_name(article: dict[str, Any]) -> str | None:
    license_value = article.get("license")
    if isinstance(license_value, dict):
        return license_value.get("name")
    return str(license_value) if license_value else None


def download_figshare_archive(raw_dir: Path, timeout_s: int = 60) -> Path:
    """Download the selected archive atomically and persist its provenance."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(FIGSHARE_ARTICLE_URL, timeout=timeout_s)
    response.raise_for_status()
    article = response.json()
    selected = choose_archive_file(article["files"])

    destination = raw_dir / str(selected["name"])
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with requests.get(selected["download_url"], stream=True, timeout=timeout_s) as download:
            download.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "source": "Figshare article 24152163",
        "article_url": article.get("url_public_html"),
        "license": _license_name(article),
        "file_name": destination.name,
        "download_url": selected["download_url"],
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
