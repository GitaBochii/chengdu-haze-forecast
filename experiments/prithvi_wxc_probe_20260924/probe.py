"""Official-code feasibility probe, NOT pretrained inference or accuracy evaluation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parent
source = ROOT / "vendor/model.py"
spec = importlib.util.spec_from_file_location("official_prithvi_model", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_model(full):
    return module.PrithviWxC(
        in_channels=160, input_size_time=2, in_channels_static=4,
        input_scalers_mu=torch.zeros(160), input_scalers_sigma=torch.ones(160),
        input_scalers_epsilon=1e-6, static_input_scalers_mu=torch.zeros(7),
        static_input_scalers_sigma=torch.ones(7), static_input_scalers_epsilon=1e-6,
        output_scalers=torch.ones(160),
        n_lats_px=360 if full else 16, n_lons_px=576 if full else 16,
        patch_size_px=(2, 2), mask_unit_size_px=(30, 32) if full else (8, 8),
        mask_ratio_inputs=0., mask_ratio_targets=0.,
        embed_dim=2560 if full else 128,
        n_blocks_encoder=12 if full else 1, n_blocks_decoder=2 if full else 1,
        mlp_multiplier=4, n_heads=16 if full else 4, dropout=0., drop_path=0.,
        parameter_dropout=0., residual="climate", masking_mode="both",
        positional_encoding="fourier", encoder_shifting=True, decoder_shifting=True,
    )


def main():
    torch.manual_seed(24)
    with torch.device("meta"):
        full = make_model(True)
    count = sum(p.numel() for p in full.parameters())
    del full
    report = {
        "source_commit": "79dabfcd17abe77e2d5c696707c0164a04f2ec01",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "torch": torch.__version__, "full_config_parameters": count,
        "full_weights_fp32_gib": count * 4 / 2**30,
        "full_weights_fp16_gib": count * 2 / 2**30,
        "pretrained_weights_loaded": False, "real_weather_inputs_used": False,
        "accuracy_evaluated": False,
        "scope": "full model meta allocation and reduced random-initialized architecture smoke test",
    }
    if not torch.cuda.is_available():
        report["smoke_status"] = "blocked_no_cuda"
    else:
        report["gpu"] = torch.cuda.get_device_name()
        free, total = torch.cuda.mem_get_info()
        report.update(gpu_free_gib=free/2**30, gpu_total_gib=total/2**30)
        torch.cuda.reset_peak_memory_stats()
        model = make_model(False).cuda().eval()
        report["smoke_parameters"] = sum(p.numel() for p in model.parameters())
        batch = {
            "x": torch.randn(1, 2, 160, 16, 16, device="cuda"),
            "y": torch.zeros(1, 160, 16, 16, device="cuda"),
            "static": torch.zeros(1, 6, 16, 16, device="cuda"),
            "climate": torch.zeros(1, 160, 16, 16, device="cuda"),
            "input_time": torch.tensor([-6.], device="cuda"),
            "lead_time": torch.tensor([6.], device="cuda"),
        }
        batch["static"][:, 0] = 30.5
        batch["static"][:, 1] = 104.
        start = time.perf_counter()
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
            torch.cuda.synchronize()
            assert output.shape == (1, 160, 16, 16)
            assert torch.isfinite(output).all()
            report.update(smoke_status="passed", output_shape=list(output.shape),
                          elapsed_seconds=time.perf_counter()-start,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        except Exception as exc:
            report.update(smoke_status="failed", error=f"{type(exc).__name__}: {exc}")
    (ROOT / "probe_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
