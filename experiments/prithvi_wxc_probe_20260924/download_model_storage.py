"""Retrieve only model storage records from a pinned official PyTorch ZIP."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib

import requests

ROOT = Path(__file__).resolve().parent
REVISION = "e2f62b3e19890f1a5e225bb123acb4b33df85ef8"
TOTAL = 28447290466
URL = f"https://huggingface.co/Prithvi-WxC/prithvi.wxc.2300m.v1/resolve/{REVISION}/prithvi.wxc.2300m.v1.pt"


def extract_record(blob, item):
    if len(blob) < 30 or blob[:4] != b"PK\x03\x04":
        raise ValueError("Bad ZIP local header")
    compression = struct.unpack_from("<H", blob, 8)[0]
    n, extra = struct.unpack_from("<HH", blob, 26)
    if compression != 0 or blob[30:30+n].decode("ascii") != item["filename"]:
        raise ValueError("Unsupported or mismatched record")
    start = 30+n+extra
    payload = blob[start:start+item["bytes"]]
    if len(payload) != item["bytes"] or zlib.crc32(payload) != item["crc"]:
        raise ValueError("Record length/CRC mismatch")
    return payload


def inventory():
    path = ROOT / "model_storage_inventory.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data["revision"] != REVISION:
            raise ValueError("Revision mismatch")
        return data["records"]
    from inspect_checkpoint import RangeReader
    reader = RangeReader(URL, TOTAL)
    with zipfile.ZipFile(reader) as archive:
        all_files = sorted(archive.infolist(), key=lambda x: x.header_offset)
    state = json.loads((ROOT / "tensor_descriptors.json").read_text())["model_state"]
    wanted = {v["storage"]["key"]: v["storage"] for v in state.values()}
    records = []
    for i, entry in enumerate(all_files[:-1]):
        key = entry.filename.rsplit("/", 1)[-1]
        if "/data/" not in entry.filename or key not in wanted:
            continue
        size = wanted[key]["elements"] * (4 if wanted[key]["dtype"] == "float32" else 1)
        if entry.compress_type != 0 or entry.file_size != size or not key.isdigit():
            raise ValueError("Unsupported storage record")
        records.append({"key": key, "filename": entry.filename, "bytes": size, "crc": entry.CRC,
                        "start": entry.header_offset, "end": all_files[i+1].header_offset-1})
    if len(records) != len(wanted):
        raise ValueError("Missing model storage records")
    path.write_text(json.dumps({"revision": REVISION, "records": records}, indent=2), encoding="utf-8")
    return records


def fetch(item, deadline):
    folder = ROOT / "weights/model_storage"
    folder.mkdir(exist_ok=True, parents=True)
    target = folder / f"{item['key']}.bin"
    if target.exists():
        payload = target.read_bytes()
        if len(payload) != item["bytes"] or zlib.crc32(payload) != item["crc"]:
            raise ValueError("Cached storage CRC mismatch")
    else:
        partial = ROOT / "weights/prithvi.wxc.2300m.v1.pt.partial"
        if partial.exists() and partial.stat().st_size > item["end"]:
            with partial.open("rb") as stream:
                stream.seek(item["start"])
                blob = stream.read(item["end"]-item["start"]+1)
        else:
            if time.monotonic() > deadline:
                return None
            expected = f"bytes {item['start']}-{item['end']}/{TOTAL}"
            blob = bytearray()
            with requests.get(URL+f"?record={item['key']}", headers={"Range": f"bytes={item['start']}-{item['end']}"},
                              stream=True, timeout=(15, 25)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Range mismatch")
                for chunk in response.iter_content(256*1024):
                    blob.extend(chunk)
                    if len(blob) > item["end"]-item["start"]+1:
                        raise ValueError("Excess response bytes")
                    if time.monotonic() > deadline:
                        return None
        payload = extract_record(blob, item)
        temp = target.with_suffix(".tmp")
        temp.write_bytes(payload)
        temp.replace(target)
    return {"key": item["key"], "bytes": len(payload), "crc32_verified": True,
            "sha256_local": hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.seconds < 1 or not 1 <= args.workers <= 4:
        raise ValueError("Invalid bounds")
    records = inventory()
    deadline = time.monotonic()+args.seconds
    completed, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch, record, deadline): record for record in records}
        for job in as_completed(jobs):
            try:
                result = job.result()
                if result:
                    completed.append(result)
                    if len(completed) % 25 == 0:
                        print(f"CRC verified {len(completed)}/{len(records)} storages", flush=True)
            except Exception as exc:
                errors.append({"key": jobs[job]["key"], "error_type": type(exc).__name__})
    report = {"revision": REVISION, "verified_records": completed, "errors": errors,
              "expected_records": len(records), "verified_bytes": sum(x["bytes"] for x in completed),
              "status": "complete" if len(completed) == len(records) else "partial",
              "whole_checkpoint_sha256_verified": False, "pretrained_inference_done": False}
    (ROOT / "model_storage_download.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "verified_records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
