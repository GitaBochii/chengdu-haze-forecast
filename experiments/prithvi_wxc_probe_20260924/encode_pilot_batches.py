"""Cache frozen Prithvi features while retaining every frozen pilot origin."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

from encoder_loading import build_encoder, load_encoder_state
from spatial_encoder import encode_spatial

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
REVISION = "e2f62b3e19890f1a5e225bb123acb4b33df85ef8"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_identity(root):
    paths = ["official_batch.pt", "real_sample/config.yaml", "encoder_loading.py",
             "storage_loader.py", "spatial_encoder.py", "tensor_descriptors.json",
             "vendor/PrithviWxC/model.py", "encode_pilot_batches.py"]
    return {"dependency_sha256": {name: sha(root / name) for name in paths}}


def validate_report_rows(rows, frozen):
    by_time = {r["timestamp"]: r for r in rows}
    if len(rows) != len(frozen) or len(by_time) != len(rows) or set(by_time) != {r["timestamp"] for r in frozen}:
        raise ValueError("Batch report omitted or duplicated frozen origins")
    return by_time


def validate_batch_identity(stored, origin, candidate_hash):
    from prepare_pilot_batches import validate_batch
    instant = datetime.fromisoformat(origin["timestamp"]).astimezone(timezone.utc)
    expected = {"timestamp": origin["timestamp"], "partition": origin["partition"],
                "origin_utc": instant.isoformat(), "revision": REVISION,
                "candidate_manifest_sha256": candidate_hash, "future_actual_weather_read": False,
                "weather_times_utc": [(instant-timedelta(hours=6)).isoformat(), instant.isoformat()]}
    if any(stored.get(key) != value for key, value in expected.items()):
        raise ValueError("Unmatched batch identity or time provenance")
    validate_batch(stored["batch"])


def valid_cache(value, identity):
    feature = value.get("embedding")
    return (all(value.get(k) == v for k, v in identity.items())
            and isinstance(feature, torch.Tensor) and feature.shape == (1, 2560)
            and feature.is_floating_point() and bool(torch.isfinite(feature).all())
            and bool(feature.std() > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-dates", type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.max_dates <= 25:
        raise ValueError("Invalid date budget")
    candidate_path = ROOT / "pilot_candidate_audit.json"
    candidate_hash = sha(candidate_path)
    frozen = json.loads(candidate_path.read_text())["rows"]
    batch_report = json.loads((ROOT / "pilot_batch_report.json").read_text())
    if (batch_report["candidate_manifest_sha256"] != candidate_hash
            or batch_report["revision"] != REVISION):
        raise ValueError("Unmatched batch provenance")
    by_time = validate_report_rows(batch_report["rows"], frozen)
    weights_path = ROOT / "model_storage_inventory.json"
    weights = json.loads(weights_path.read_text())
    if weights["revision"] != REVISION:
        raise ValueError("Unmatched weight revision")
    identity_base = {"candidate_manifest_sha256": candidate_hash,
                     "weights_inventory_sha256": sha(weights_path), "revision": REVISION,
                     "torch_version": str(torch.__version__), "pooling": "mean_global_and_local_tokens",
                     **dependency_identity(ROOT)}
    folder = ROOT / "pilot_embeddings"
    folder.mkdir(exist_ok=True)
    model, load_report, produced, rows = None, None, 0, []
    for origin in frozen:
        source = by_time[origin["timestamp"]]
        row = {"timestamp": origin["timestamp"], "date": origin["date"], "partition": origin["partition"]}
        if source["partition"] != origin["partition"]:
            raise ValueError("Partition was changed")
        if source["status"] != "ready":
            rows.append({**row, "status": "awaiting_batch", "batch_status": source["status"]})
            continue
        batch_path = (ROOT / source["batch_path"]).resolve()
        if not batch_path.is_relative_to(ROOT.resolve()) or sha(batch_path) != source["batch_sha256"]:
            raise ValueError("Batch path or checksum mismatch")
        identity = {**identity_base, **row, "batch_sha256": source["batch_sha256"]}
        output = folder / f"{origin['date']}.pt"
        if output.exists():
            cached = torch.load(output, weights_only=True, map_location="cpu")
            if valid_cache(cached, identity):
                rows.append({**row, "status": "ready", "embedding_path": str(output.relative_to(ROOT)),
                             "embedding_sha256": sha(output), "batch_sha256": source["batch_sha256"]})
                continue
        if produced >= args.max_dates:
            rows.append({**row, "status": "deferred_by_budget"})
            continue
        stored = torch.load(batch_path, weights_only=True, map_location="cpu")
        validate_batch_identity(stored, origin, candidate_hash)
        if stored["source_sha256"] != source["source_sha256"]:
            raise ValueError("Batch source checksum provenance mismatch")
        batch = stored["batch"]
        if model is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for frozen encoder extraction")
            values = torch.load(ROOT / "official_batch.pt", weights_only=True, map_location="cpu")
            cfg = yaml.safe_load((ROOT / "real_sample/config.yaml").read_text())["params"]
            model = build_encoder(cfg, values, (30, 32))
            free, _ = torch.cuda.mem_get_info()
            if free < sum(p.numel()*2 for p in model.parameters()) + 1024**3:
                raise RuntimeError("Insufficient free GPU memory")
            descriptions = json.loads((ROOT / "tensor_descriptors.json").read_text())["model_state"]
            records = {r["key"]: r for r in weights["records"]}
            load_report = load_encoder_state(model, descriptions, records, ROOT / "weights/model_storage", "cuda", torch.bfloat16)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        torch.manual_seed(24)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = encode_spatial(model, batch)
        value = {**identity, "embedding": tokens.float().mean(dim=(1, 2)).cpu(),
                 "origin_utc": stored["origin_utc"], "source": "frozen_pretrained_regional_encoder"}
        if not valid_cache(value, identity):
            raise ValueError("Invalid pretrained feature values")
        temp = output.with_suffix(".tmp")
        torch.save(value, temp)
        temp.replace(output)
        rows.append({**row, "status": "ready", "embedding_path": str(output.relative_to(ROOT)),
                     "embedding_sha256": sha(output), "batch_sha256": source["batch_sha256"]})
        produced += 1
        print(f"Encoded real weather {origin['date']}", flush=True)
    count = sum(r["status"] == "ready" for r in rows)
    report = {**identity_base, "status": "complete" if count == len(frozen) else "partial",
              "expected_origins": len(frozen), "ready_origins": count, "newly_encoded": produced,
              "load_this_run": load_report, "rows": rows, "accuracy_evaluated": False}
    (ROOT / "pilot_embedding_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
