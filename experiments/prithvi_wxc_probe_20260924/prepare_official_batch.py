"""Use pinned official preprocessing, then crop a regional inference batch."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from PrithviWxC.dataloaders.merra2 import (
    Merra2Dataset, SampleSpec, preproc, input_scalers, static_input_scalers, output_scalers,
)


def main():
    data = ROOT / "real_sample"
    cfg = yaml.safe_load((ROOT / "vendor/config.yaml").read_text())["data"]
    surface, vertical, levels = cfg["surface_vars"], cfg["vertical_vars"], cfg["levels"]
    dataset = Merra2Dataset(
        time_range=("2020-01-01T00:00:00", "2020-01-01T23:59:59"),
        lead_times=[6], input_times=[-6], data_path_surface=data / "merra-2",
        data_path_vertical=data / "merra-2", climatology_path_surface=data / "climatology",
        climatology_path_vertical=data / "climatology", surface_vars=surface,
        static_surface_vars=cfg["static_surface_vars"], vertical_vars=vertical,
        levels=levels, positional_encoding="fourier",
    )
    spec = SampleSpec.get(pd.Timestamp("2020-01-01T06:00:00"), 6, 6)
    sample = dataset.get_data_from_sample_spec(spec)
    raw = np.load(ROOT / "regional_raw_sample.npz")
    iy = int(np.where(dataset.lats == raw["lat"][0])[0][0])
    ix = int(np.where(dataset.lons == raw["lon"][0])[0][0])
    sample = {k: v[..., iy:iy+30, ix:ix+32].clone() if isinstance(v, torch.Tensor) else v
              for k, v in sample.items()}
    batch = preproc([sample], {"level": (0, 0), "lat": (0, 0), "lon": (0, 0)})
    np.testing.assert_allclose(batch["x"][0].numpy(), raw["x"], rtol=0, atol=0)
    assert batch["static"].shape == (1, 10, 30, 32)
    assert batch["input_time"].item() == 6
    for value in batch.values():
        assert torch.isfinite(value).all()
    c = data / "climatology"
    mu, sigma = input_scalers(surface, vertical, levels, c / "musigma_surface.nc", c / "musigma_vertical.nc")
    smu, ssigma = static_input_scalers(c / "musigma_surface.nc", cfg["static_surface_vars"])
    out_sigma = output_scalers(surface, vertical, levels, c / "anomaly_variance_surface.nc", c / "anomaly_variance_vertical.nc").sqrt()
    assert smu.shape == (11,) and smu[3:].shape == (8,)
    torch.save({"batch": batch, "mu": mu, "sigma": sigma, "static_mu": smu,
                "static_sigma": ssigma, "output_sigma": out_sigma}, ROOT / "official_batch.pt")
    report = {"shapes": {k: list(v.shape) for k,v in batch.items()},
              "raw_cube_matches_official_exactly": True,
              "static_channels_explanation": "2 lat/lon radians + 4 calendar encodings + 4 physical fields; model Fourier consumes 2 separately, leaving 8",
              "input_time_hours": 6, "lead_time_hours": 6,
              "inference_done": False}
    (ROOT / "official_batch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
