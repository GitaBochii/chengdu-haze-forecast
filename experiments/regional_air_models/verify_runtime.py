"""Record reproducible vendor provenance and a real CUDA backward smoke test."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The independent experiment requires a working CUDA runtime")
    x = torch.randn(16, 32, device="cuda", requires_grad=True)
    loss = x.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    vendors = []
    for name in ("PM25-GNN", "AirFormer"):
        folder = ROOT / "vendor" / name
        tracked = subprocess.check_output(
            ["git", "-C", str(folder), "ls-files", "-z"]
        ).decode().split("\0")
        hashes = {
            path: hashlib.sha256((folder / path).read_bytes()).hexdigest()
            for path in tracked if path and (folder / path).is_file()
        }
        vendors.append({
            "name": name,
            "commit": subprocess.check_output(
                ["git", "-C", str(folder), "rev-parse", "HEAD"]
            ).decode().strip(),
            "origin": subprocess.check_output(
                ["git", "-C", str(folder), "remote", "get-url", "origin"]
            ).decode().strip(),
            "license": "MIT" if name == "PM25-GNN" else "No LICENSE file in upstream checkout; no commercial permission assumed",
            "sha256": hashes,
        })
    installed_packages = {}
    for dist in importlib.metadata.distributions():
        installed_packages.setdefault(dist.metadata["Name"], dist.version)
    report = {
        "python": sys.executable,
        "python_version": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_backward_finite": True,
        "dependency_reuse": "Read-only .pth bridge to existing .venv/Lib/site-packages; independent environment packages take priority",
        "installed_packages": installed_packages,
        "vendors": vendors,
    }
    path = ROOT / "runtime_manifest.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("vendors", "installed_packages")}, ensure_ascii=True, indent=2))
    print(f"Manifest: {path}")


if __name__ == "__main__":
    main()
