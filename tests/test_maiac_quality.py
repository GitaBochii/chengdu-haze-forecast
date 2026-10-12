import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "maiac_quality", Path(__file__).parents[1] / "scripts/audit_maiac_sample.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_strict_quality_excludes_cloud_water_adjacency_and_fill():
    raw = np.array([100, 100, 100, 100, 100, -28672, -50, 6001])
    qa = np.array([1, 3, 1 | (1 << 3), 1 | (1 << 5), 1 | (5 << 8), 1, 1, 1])
    mask = module.strict_mask(raw, qa, -28672, [-100, 6000])
    assert mask.tolist() == [True, False, False, False, False, False, True, False]


def test_grid_centres_and_bbox():
    from pyproj import CRS, Transformer
    radius = 6371007.181
    forward = Transformer.from_crs(4326, CRS.from_proj4(f"+proj=sinu +R={radius} +units=m"), always_xy=True)
    x, y = forward.transform(104, 30.5)
    grid = {"Projection": "GCTP_SNSOID", "GridOrigin": "HDFE_GD_UL",
            "XDim": 1, "YDim": 1, "ProjParams": [radius],
            "UpperLeftPointMtrs": [x-500, y+500], "LowerRightMtrs": [x+500, y-500]}
    lon, lat = module.grid_coordinates(grid)
    assert lon[0, 0] == pytest.approx(104)
    assert lat[0, 0] == pytest.approx(30.5)
    assert module.region_mask(lon, lat, (103.5, 30.2, 104.7, 31.1)).all()
    assert not module.region_mask(lon, lat, (105, 30.2, 106, 31.1)).any()


def test_orbit_labels_fail_on_mismatch():
    assert module.orbit_labels("20240010200T 20240010505A", 2)[1]["platform"] == "Aqua"
    with pytest.raises(ValueError):
        module.orbit_labels("20240010200T", 2)
    with pytest.raises(ValueError):
        module.orbit_labels("20243670200T", 1)
