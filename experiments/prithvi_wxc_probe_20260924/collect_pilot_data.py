"""Bounded collection of frozen pilot inputs; no substitute dates or training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import requests

from prepare_real_sample import download

ROOT = Path(__file__).resolve().parent
REVISION = "e2f62b3e19890f1a5e225bb123acb4b33df85ef8"


def validate_inventory(data, candidate_hash):
    if (data.get("revision") != REVISION
            or data.get("candidate_manifest_sha256") != candidate_hash or data.get("missing")):
        raise ValueError("Inventory does not match frozen complete candidate manifest")
    files = data["files"]
    paths = [item["path"] for item in files]
    if not files or len(files) > 100 or len(set(paths)) != len(paths):
        raise ValueError("Invalid or duplicate pilot file list")
    pattern = r"(?:merra-2/(?:MERRA2_sfc|MERRA_pres)_2020\d{4}|climatology/climate_(?:surface|vertical)_doy\d{3}_hour12)\.nc"
    for item in files:
        if (not re.fullmatch(pattern, item["path"])
                or type(item["size"]) is not int or not 0 < item["size"] < 2_000_000_000
                or not re.fullmatch(r"[0-9a-f]{64}", item.get("lfs", {}).get("oid", ""))):
            raise ValueError("Invalid pilot path, size or upstream hash")
    if sum(item["size"] for item in files) > 25_000_000_000:
        raise ValueError("Pilot transfer exceeds approved bounded scope")
    return files


def fetch(item, deadline):
    if time.monotonic() >= deadline:
        return None
    with requests.Session() as session:
        return download(session, REVISION, item, deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800 or not 1 <= args.workers <= 4:
        raise ValueError("Invalid collection bounds")
    candidate_bytes = (ROOT / "pilot_candidate_audit.json").read_bytes()
    candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
    inventory = json.loads((ROOT / "pilot_file_inventory.json").read_text())
    files = validate_inventory(inventory, candidate_hash)
    from prepare_pilot_inventory import selected_paths
    requested = list(dict.fromkeys(path for row in json.loads(candidate_bytes)["rows"]
                                  for path in selected_paths(row["timestamp"])))
    if {item["path"] for item in files} != set(requested):
        raise ValueError("File inventory does not exactly cover frozen origin dates")
    # Complete each frozen date's bundle before moving on; never change the population.
    lookup = {item["path"]: item for item in files}
    files = [lookup[path] for path in requested]
    deadline = time.monotonic()+args.seconds
    verified, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch, item, deadline): item for item in files}
        for job in as_completed(jobs):
            try:
                result = job.result()
                if result:
                    verified.append(result)
                    print(f"SHA256 verified {len(verified)}/{len(files)}: {result['path']}", flush=True)
            except Exception as exc:
                errors.append({"path": jobs[job]["path"], "error_type": type(exc).__name__})
    report = {"revision": REVISION, "candidate_manifest_sha256": candidate_hash,
              "inventory_sha256": hashlib.sha256((ROOT / "pilot_file_inventory.json").read_bytes()).hexdigest(),
              "status": "files_hash_verified" if len(verified) == len(files) else "partial",
              "expected_files": len(files), "verified_files": verified, "errors": errors,
              "verified_bytes": sum(item["bytes"] for item in verified),
              "physical_schema_verified": False, "multisample_training_done": False}
    (ROOT / "pilot_data_download.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "verified_files"}, indent=2))


if __name__ == "__main__":
    main()
