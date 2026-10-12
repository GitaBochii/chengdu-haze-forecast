"""Download the pinned public checkpoint with validated resumable ranges."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import time

import requests

ROOT = Path(__file__).resolve().parent
REPO = "Prithvi-WxC/prithvi.wxc.2300m.v1"
REVISION = "e2f62b3e19890f1a5e225bb123acb4b33df85ef8"
NAME = "prithvi.wxc.2300m.v1.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    if args.seconds < 1:
        raise ValueError("Time budget must be positive")
    session = requests.Session()
    r = session.get(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}", timeout=30)
    r.raise_for_status()
    item = next(v for v in r.json() if v["path"] == NAME)
    size, digest = item["size"], item["lfs"]["oid"]
    folder = ROOT / "weights"
    folder.mkdir(exist_ok=True)
    target = folder / NAME
    partial = folder / (NAME + ".partial")
    report = {"revision": REVISION, "expected_bytes": size, "expected_sha256": digest,
              "pretrained_inference_done": False}
    start = time.monotonic()
    deadline = start + args.seconds
    try:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError("Partial file exceeds expected size")
        if shutil.disk_usage(folder).free < size-offset+1024**3:
            raise RuntimeError("Insufficient disk headroom")
        attempts = 0
        while not target.exists() and offset < size and time.monotonic() < deadline:
            attempts += 1
            if attempts > 4:
                break
            try:
                # Explicit range and no cache avoid stale cached partial responses.
                url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}?download=true&offset={offset}"
                with session.get(url, headers={"Range": f"bytes={offset}-{size-1}"}, stream=True,
                                 timeout=(15, 30)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{size-1}/{size}":
                        raise ValueError("Incorrect resume response")
                    next_log = offset + 512*1024**2
                    with partial.open("ab" if offset else "wb") as stream:
                        for chunk in response.iter_content(256*1024):
                            if offset+len(chunk) > size:
                                raise ValueError("Too many response bytes")
                            stream.write(chunk)
                            offset += len(chunk)
                            if offset >= next_log:
                                print(f"Downloaded {offset/1024**3:.2f}/{size/1024**3:.2f} GiB", flush=True)
                                next_log = offset + 512*1024**2
                            if time.monotonic() >= deadline:
                                break
            except requests.RequestException as exc:
                report["last_network_error"] = type(exc).__name__
                print(f"Network retry {attempts}: {type(exc).__name__}", flush=True)
                time.sleep(2)
                offset = partial.stat().st_size if partial.exists() else 0
        candidate = target if target.exists() else partial
        actual = candidate.stat().st_size if candidate.exists() else 0
        report["downloaded_bytes"] = actual
        if actual == size:
            with candidate.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != digest:
                raise ValueError("SHA256 mismatch; file not accepted")
            if candidate == partial:
                partial.replace(target)
            report.update(status="verified", sha256=actual_hash)
        else:
            report["status"] = "partial_resume_required"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
    report["elapsed_seconds"] = time.monotonic()-start
    (ROOT / "checkpoint_download.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
