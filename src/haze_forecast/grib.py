"""Small, auditable GRIB point extraction via NOAA's wgrib2 utility."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

POINT_VALUE_PATTERN = re.compile(
    r"lon=(?P<longitude>-?\d+(?:\.\d+)?),lat=(?P<latitude>-?\d+(?:\.\d+)?),"
    r"val=(?P<value>-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)


@dataclass(frozen=True)
class GRIBPointValue:
    """Value at wgrib2's nearest grid point, with its actual coordinates."""

    longitude: float
    latitude: float
    value: float


def parse_wgrib2_point_output(output: str) -> GRIBPointValue:
    """Parse a single ``wgrib2 -lon`` inventory line and reject missing data."""
    matches = POINT_VALUE_PATTERN.findall(output)
    if len(matches) != 1:
        raise ValueError(f"expected one numeric wgrib2 point value, found {len(matches)}")
    longitude, latitude, value = matches[0]
    return GRIBPointValue(longitude=float(longitude), latitude=float(latitude), value=float(value))


def query_grib_point(
    grib_path: Path,
    longitude: float,
    latitude: float,
    wgrib2_executable: Path,
    timeout_s: int = 30,
) -> GRIBPointValue:
    """Read one nearest-grid value from a GRIB message using an explicit binary."""
    if not grib_path.is_file():
        raise FileNotFoundError(grib_path)
    if not wgrib2_executable.is_file():
        raise FileNotFoundError(wgrib2_executable)
    if not 0 <= longitude <= 360:
        raise ValueError("wgrib2 longitude must be in [0, 360]")
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be in [-90, 90]")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    result = subprocess.run(
        [
            str(wgrib2_executable),
            str(grib_path),
            "-s",
            "-lon",
            str(longitude),
            str(latitude),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return parse_wgrib2_point_output(result.stdout)
