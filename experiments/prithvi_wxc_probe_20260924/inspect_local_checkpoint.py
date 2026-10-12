"""Memory-map a verified checkpoint; compare all keys/shapes without copying weights."""
import json
import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from PrithviWxC.model import PrithviWxC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    path = ROOT / "weights/prithvi.wxc.2300m.v1.pt"
    if not args.metadata_only:
        verified = json.loads((ROOT / "checkpoint_download.json").read_text())
        if verified["status"] != "verified":
            raise RuntimeError("Checkpoint download not verified")
        if path.stat().st_size != verified["expected_bytes"]:
            raise ValueError("Checkpoint size changed")
    config = yaml.safe_load((ROOT / "real_sample/config.yaml").read_text())["params"]
    values = torch.load(ROOT / "official_batch.pt", weights_only=True, map_location="cpu")
    with torch.device("meta"):
        model = PrithviWxC(
            **config, input_scalers_mu=values["mu"], input_scalers_sigma=values["sigma"],
            static_input_scalers_mu=values["static_mu"], static_input_scalers_sigma=values["static_sigma"],
            output_scalers=values["output_sigma"], mask_ratio_inputs=0., mask_ratio_targets=0.,
            residual="climate", masking_mode="global", positional_encoding="fourier",
            encoder_shifting=True, decoder_shifting=True,
        )
    # mmap avoids materialising model and optimizer storage together in physical RAM.
    if args.metadata_only:
        checkpoint = json.loads((ROOT / "tensor_descriptors.json").read_text())
        state = checkpoint["model_state"]
        shapes = {k: v["shape"] for k,v in state.items()}
    else:
        checkpoint = torch.load(path, mmap=True, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        shapes = {k: list(v.shape) for k,v in state.items()}
    expected = model.state_dict()
    differences = {k: {"expected": list(expected[k].shape), "actual": shapes[k]}
                   for k in expected.keys() & state.keys() if list(expected[k].shape) != shapes[k]}
    missing = sorted(expected.keys() - state.keys())
    extra = sorted(state.keys() - expected.keys())
    report = {"status": "compatible" if not (differences or missing or extra) else "incompatible",
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "missing_keys": missing, "extra_keys": extra, "shape_differences": differences,
              "metadata_only": args.metadata_only,
              "checkpoint_top_level_keys": checkpoint["top_level_keys"] if args.metadata_only else list(checkpoint.keys()),
              "model_tensors": len(state), "inference_done": False}
    filename = "checkpoint_shape_report.json" if args.metadata_only else "local_checkpoint_report.json"
    (ROOT / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
