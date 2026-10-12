"""Extract real regional Prithvi features only after every encoder storage exists."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
REVISION = "e2f62b3e19890f1a5e225bb123acb4b33df85ef8"


def storage_preflight(descriptions, records, folder):
    required = {d["storage"]["key"] for name, d in descriptions.items()
                if not name.startswith(("decoder.", "decoder_shifter.", "unembed."))}
    missing, invalid = [], []
    for key in sorted(required, key=int):
        if not key.isascii() or not key.isdigit():
            raise ValueError("Invalid storage key")
        path = Path(folder) / f"{key}.bin"
        if not path.exists():
            missing.append(key)
        elif path.stat().st_size != records[key]["bytes"]:
            invalid.append(key)
    return {"required_records": len(required), "missing_records": missing,
            "invalid_size_records": invalid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    inventory = json.loads((ROOT / "model_storage_inventory.json").read_text())
    if inventory["revision"] != REVISION:
        raise ValueError("Unapproved checkpoint revision")
    descriptions = json.loads((ROOT / "tensor_descriptors.json").read_text())["model_state"]
    records = {r["key"]: r for r in inventory["records"]}
    folder = ROOT / "weights/model_storage"
    check = storage_preflight(descriptions, records, folder)
    report = {"revision": REVISION, "preflight": check, "pretrained_inference_done": False,
              "whole_checkpoint_sha256_verified": False, "accuracy_evaluated": False}
    path = ROOT / "pretrained_encoder_report.json"
    if check["missing_records"] or check["invalid_size_records"] or args.preflight_only:
        report["status"] = "awaiting_encoder_storage" if check["missing_records"] or check["invalid_size_records"] else "preflight_only"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    import torch
    import yaml
    from encoder_loading import build_encoder, load_encoder_state
    from spatial_encoder import encode_spatial
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this bounded regional extraction")
    values = torch.load(ROOT / "official_batch.pt", weights_only=True, map_location="cpu")
    config = yaml.safe_load((ROOT / "real_sample/config.yaml").read_text())["params"]
    model = build_encoder(config, values, (30, 32))
    free, _ = torch.cuda.mem_get_info()
    required_bytes = sum(p.numel()*2 for p in model.parameters()) + 1024**3
    if free < required_bytes:
        raise RuntimeError("Insufficient free GPU memory; other user processes left untouched")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    loaded = load_encoder_state(model, descriptions, records, folder, "cuda", torch.bfloat16)
    batch = {key: values["batch"][key].to("cuda")
             for key in ("x", "static", "climate", "input_time", "lead_time")}
    torch.manual_seed(24)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = encode_spatial(model, batch)
    if not torch.isfinite(encoded).all():
        raise ValueError("Nonfinite pretrained encoder output")
    embedding = encoded.float().mean(dim=(1, 2)).cpu()
    if embedding.shape != (1, 2560) or embedding.std() <= 0:
        raise ValueError("Unexpected or constant spatial embedding")
    paired = json.loads((ROOT / "paired_sample_report.json").read_text())
    output = ROOT / "pretrained_spatial_embedding.pt"
    torch.save({"embedding": embedding, "origin_utc": paired["origin_utc"],
                "weather_times_utc": paired["weather_times_utc"], "revision": REVISION,
                "scope": "offline regional reanalysis; not an accuracy evaluation"}, output)
    report.update(status="real_pretrained_encoder_passed", pretrained_inference_done=True,
                  load=loaded, embedding_shape=list(embedding.shape), token_shape=list(encoded.shape),
                  embedding_file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  input_file_sha256=hashlib.sha256((ROOT / "official_batch.pt").read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-start,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  future_truth_used=False, regional_geometry_regenerated=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
