"""Bounded, resumable public sample acquisition; never download the full repository."""

import ctypes
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parent
REPO = "Prithvi-WxC/prithvi.wxc.2300m.v1"
FILES = ["merra-2/MERRA2_sfc_20200101.nc", "climatology/musigma_surface.nc",
         "climatology/musigma_vertical.nc", "climatology/anomaly_variance_surface.nc",
         "climatology/anomaly_variance_vertical.nc", "config.yaml",
         "merra-2/MERRA_pres_20200101.nc", "climatology/climate_surface_doy001_hour12.nc",
         "climatology/climate_vertical_doy001_hour12.nc"]


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]


def download(session, revision, item, budget_end):
    target = ROOT / "real_sample" / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item["size"]
    candidate = target
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError("Partial file exceeds expected size")
        if offset < expected:
            headers = {"Range": f"bytes={offset}-{expected-1}"} if offset else {}
            url = f"https://huggingface.co/{REPO}/resolve/{revision}/{item['path']}"
            with session.get(url, headers=headers, stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise ValueError("Server did not honor resume range")
                if offset and response.headers.get("Content-Range") != f"bytes {offset}-{expected-1}/{expected}":
                    raise ValueError("Unexpected range offset or total")
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if stream.tell()+len(chunk) > expected:
                            raise ValueError("Response exceeds expected size")
                        stream.write(chunk)
                        if time.monotonic() > budget_end:
                            raise TimeoutError("Download budget reached; partial file retained")
        if partial.stat().st_size != expected:
            raise ValueError("Downloaded size differs from repository manifest")
        candidate = partial
    if candidate.stat().st_size != expected:
        raise ValueError("Existing file size differs from repository manifest")
    with candidate.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected_hash = item.get("lfs", {}).get("oid")
    if expected_hash and digest != expected_hash:
        raise ValueError("LFS SHA256 mismatch")
    if candidate != target:
        candidate.replace(target)
    return {"path": item["path"], "bytes": expected, "sha256": digest,
            "upstream_sha256_verified": bool(expected_hash)}


def main():
    session = requests.Session()
    response = session.get(f"https://huggingface.co/api/models/{REPO}", timeout=30)
    response.raise_for_status()
    revision = response.json()["sha"]
    metadata = []
    for directory in ("", "/merra-2", "/climatology"):
        url = f"https://huggingface.co/api/models/{REPO}/tree/{revision}{directory}?limit=1000"
        while url:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            metadata.extend(response.json())
            url = response.links.get("next", {}).get("url")
    lookup = {item["path"]: item for item in metadata}
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("Could not read physical memory")
    report = {"repo": REPO, "revision": revision,
              "ram_total_gib": memory.total / 2**30, "ram_available_gib": memory.available / 2**30,
              "pretrained_loaded": False, "downloads": [],
              "selected_inventory": [lookup[name] for name in FILES],
              "vertical_sample": [x for x in metadata if x["path"] == "merra-2/MERRA_pres_20200101.nc"]}
    deadline = time.monotonic() + 150
    try:
        for name in FILES:
            report["downloads"].append(download(session, revision, lookup[name], deadline))
            print(f"Verified {name}", flush=True)
        report["status"] = "sample_files_downloaded_input_assembly_pending"
    except Exception as exc:
        report["status"] = "incomplete"
        report["error"] = type(exc).__name__
    (ROOT / "real_sample_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("selected_inventory", "vertical_sample")}, indent=2))


if __name__ == "__main__":
    main()
