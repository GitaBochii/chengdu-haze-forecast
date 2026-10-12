"""Build a real regional channel cube; not yet a model-ready normalized batch."""

import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "real_sample"


def main():
    # Read the pinned source configuration rather than independently reorder channels.
    config = yaml.safe_load((ROOT / "vendor/config.yaml").read_text())["data"]
    surface = config["surface_vars"]
    vertical = config["vertical_vars"]
    levels = config["levels"]
    with xr.open_dataset(DATA / "merra-2/MERRA2_sfc_20200101.nc", engine="h5netcdf") as s, \
         xr.open_dataset(DATA / "merra-2/MERRA_pres_20200101.nc", engine="h5netcdf") as v:
        cy = int(np.abs(s.lat.values - 30.67).argmin())
        cx = int(np.abs(s.lon.values - 104.06).argmin())
        region = dict(lat=slice(cy-15, cy+15), lon=slice(cx-16, cx+16))
        s = s.isel(**region).load()
        v = v.isel(**region).load()
        if not np.array_equal(s.lat, v.lat) or not np.array_equal(s.lon, v.lon):
            raise ValueError("Surface and vertical coordinates differ")
        selected_times = ["2020-01-01T00:00:00", "2020-01-01T06:00:00"]
        arrays = [s[name].sel(time=selected_times).transpose("time", "lat", "lon").values for name in surface]
        labels = list(surface)
        for name in vertical:
            for level in levels:
                arrays.append(v[name].sel(time=selected_times, lev=level).transpose("time", "lat", "lon").values)
                labels.append(f"{name}_level_{level}")
        cube = np.stack(arrays, axis=1).astype(np.float32)
        if cube.shape != (2, 160, 30, 32) or not np.isfinite(cube).all():
            raise ValueError("Unexpected shape or missing regional values")
        static = np.stack([s[name].values for name in config["static_surface_vars"]])
        if static.shape != (4, 30, 32) or not np.isfinite(static).all():
            raise ValueError("Invalid static fields")
        np.savez_compressed(ROOT / "regional_raw_sample.npz", x=cube, static=static,
                            lat=s.lat.values, lon=s.lon.values, channel_names=labels,
                            times=selected_times)
        report = {
            "status": "real_regional_raw_cube_verified", "shape": list(cube.shape),
            "times_from_file": selected_times, "channels": labels,
            "lat_range": [float(s.lat.min()), float(s.lat.max())],
            "lon_range": [float(s.lon.min()), float(s.lon.max())],
            "all_finite": True, "static_shape": list(static.shape),
            "pretrained_inference_performed": False, "normalized_model_ready": False,
            "notes": ["Broad region centred near Chengdu; minimum official mask-unit dimensions.",
                      "Raw cube only: official preprocessing, climate alignment and scalers still require validation.",
                      "HF config static channels=8 differs from pinned source validation config=4; do not bypass strict weight loading."]}
        (ROOT / "regional_sample_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k != "channels"}, indent=2))


if __name__ == "__main__":
    main()
