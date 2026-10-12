"""Audit downloaded MAIAC tiles; tile-level counts are NOT Chengdu coverage."""

import hashlib
import json
from contextlib import chdir
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD19A2_GRANULES"
BBOX = (103.5, 30.2, 104.7, 31.1)


def grid_coordinates(grid):
    from pyproj import CRS, Transformer
    if grid["Projection"] != "GCTP_SNSOID" or grid["GridOrigin"] != "HDFE_GD_UL":
        raise ValueError("Unsupported grid projection/origin")
    nx, ny = int(grid["XDim"]), int(grid["YDim"])
    left, top = grid["UpperLeftPointMtrs"]
    right, bottom = grid["LowerRightMtrs"]
    x = left + (np.arange(nx) + 0.5) * (right - left) / nx
    y = top + (np.arange(ny) + 0.5) * (bottom - top) / ny
    source = CRS.from_proj4(f"+proj=sinu +R={grid['ProjParams'][0]} +units=m")
    return Transformer.from_crs(source, 4326, always_xy=True).transform(*np.meshgrid(x, y))


def region_mask(lon, lat, bbox):
    west, south, east, north = bbox
    return (lon >= west) & (lon < east) & (lat >= south) & (lat < north)


def orbit_labels(value, count):
    tokens = value.strip().split()
    if len(tokens) != count:
        raise ValueError("Orbit count/time mismatch")
    labels = []
    for token in tokens:
        if len(token) != 12 or token[-1] not in "TA":
            raise ValueError("Invalid orbit label")
        time = datetime.strptime(token[:-1], "%Y%j%H%M")
        if time.strftime("%Y%j%H%M") != token[:-1]:
            raise ValueError("Invalid orbit date")
        labels.append({"raw": token, "nominal_time": time.isoformat(),
                       "platform": "Terra" if token[-1] == "T" else "Aqua",
                       "usable_for_hourly_alignment": False})
    return labels


def strict_mask(raw, qa, fill, valid_range):
    # Clear land, no cloud adjacency, best AOD quality. Keep valid negative retrievals.
    return (
        (raw != fill) & (raw >= valid_range[0]) & (raw <= valid_range[1])
        & ((qa & 7) == 1) & (((qa >> 3) & 3) == 0)
        & (((qa >> 5) & 7) == 0) & (((qa >> 8) & 15) == 0)
    )


def main():
    from pyhdf.SD import SD, SDC
    import pvl

    folder = ROOT / "data/raw/maiac_pilot_20260924"
    manifest = json.loads((folder / "download_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for expected in manifest["files"]:
        path = folder / expected["name"]
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Downloaded file no longer matches manifest")
        # HDF4's Windows API cannot open Unicode absolute paths; filenames are ASCII.
        with chdir(path.parent):
            handle = SD(path.name, SDC.READ)
        try:
            field = handle.select("Optical_Depth_055")
            attrs = field.attributes()
            raw = field.get()
            qa = handle.select("AOD_QA").get()
            if raw.ndim != 3 or qa.shape != raw.shape:
                raise ValueError("Unexpected orbit grid dimensions")
            valid = strict_mask(raw, qa, attrs["_FillValue"], attrs["valid_range"])
            counts = valid.sum(axis=(1, 2)).tolist()
            global_attrs = handle.attributes()
            grid = pvl.loads(global_attrs["StructMetadata.0"].rstrip("\x00"))["GridStructure"]["GRID_1"]
            lon, lat = grid_coordinates(grid)
            if lon.shape != raw.shape[1:]:
                raise ValueError("Grid metadata and data dimensions disagree")
            area = region_mask(lon, lat, BBOX)
            labels = orbit_labels(global_attrs["Orbit_time_stamp"], raw.shape[0])
            if int(global_attrs["Orbit_amount"]) != raw.shape[0]:
                raise ValueError("Orbit amount disagrees with array")
            crop_rows = []
            for index, label in enumerate(labels):
                selected = valid[index] & area
                values = raw[index][selected] * attrs["scale_factor"]
                if attrs["add_offset"] != 0:
                    raise ValueError("Nonzero offset needs explicit convention review")
                crop_rows.append({**label, "region_pixels": int(area.sum()),
                    "nonfill_pixels": int(((raw[index] != attrs["_FillValue"]) & area).sum()),
                    "cloud_mask_counts": {str(int(k)): int(v) for k, v in zip(
                        *np.unique((qa[index][area] & 7), return_counts=True))},
                    "aod_quality_counts": {str(int(k)): int(v) for k, v in zip(
                        *np.unique(((qa[index][area] >> 8) & 15), return_counts=True))},
                    "strict_valid_pixels": int(selected.sum()),
                    "valid_fraction": float(selected.sum()/area.sum()) if area.any() else None,
                    "aod055_median": float(np.median(values)) if values.size else None,
                    "aod055_p25": float(np.quantile(values, .25)) if values.size else None,
                    "aod055_p75": float(np.quantile(values, .75)) if values.size else None})
            rows.append({
                "file": path.name, "sha256_matches": True,
                "shape_orbit_y_x": list(raw.shape),
                "scale_factor": attrs["scale_factor"], "add_offset": attrs["add_offset"],
                "fill_value": attrs["_FillValue"], "valid_range": attrs["valid_range"],
                "chengdu_bbox_orbits": crop_rows,
                "strict_valid_pixels_per_orbit_entire_tile": counts,
                "strict_valid_fraction_per_orbit_entire_tile": [
                    c / (raw.shape[1] * raw.shape[2]) for c in counts
                ],
            })
        finally:
            handle.end()
    report = {
        "status": "readable_hdf4_tiles_with_quality_masks",
        "scope": "Chengdu rectangular study region, not administrative boundary; nominal orbit labels only",
        "bbox_west_south_east_north": BBOX,
        "hourly_alignment_verified": False,
        "qa_reference": SOURCE,
        "quality_rule": "clear land, normal adjacency, best AOD; file valid_range and fill",
        "model_trained": False, "operational_availability_proven": False,
        "files": rows,
    }
    target = ROOT / "artifacts/maiac_pilot_20260924/quality_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
