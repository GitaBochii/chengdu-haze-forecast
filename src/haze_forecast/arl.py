"""Read a small, local subset exported by NOAA ARL's READY archive service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ARLGrid:
    """Regular latitude-longitude grid encoded in an ARL packed file."""

    nx: int
    ny: int
    latitude0: float
    longitude0: float
    latitude_step: float
    longitude_step: float


@dataclass(frozen=True)
class ARLSurfaceForecast:
    """A point forecast decoded from one HYSPLIT-compatible ARL subset."""

    valid_at: datetime
    forecast_hours: int
    grid_latitude: float
    grid_longitude: float
    temperature_c: float
    specific_humidity_2m_kg_kg: float
    relative_humidity_pct: float
    wind_u_ms: float
    wind_v_ms: float
    boundary_layer_height_m: float


def parse_arl_grid(index_record: bytes) -> ARLGrid:
    """Parse the 50-byte label plus 108-byte extended ARL index header."""
    if len(index_record) < 158:
        raise ValueError("ARL index record must contain a 50-byte label and 108-byte header")
    if index_record[14:18].decode("ascii") != "INDX":
        raise ValueError("the first ARL record must be an INDX header")
    header = index_record[50:158].decode("ascii")
    nx = int(header[93:96])
    ny = int(header[96:99])
    if nx <= 0 or ny <= 0:
        raise ValueError("ARL grid dimensions must be positive")
    latitude_step = float(header[23:30])
    longitude_step = float(header[30:37])
    latitude0 = float(header[72:79])
    longitude0 = float(header[79:86])
    if latitude_step <= 0 or longitude_step <= 0:
        raise ValueError("ARL grid increments must be positive")
    return ARLGrid(
        nx=nx,
        ny=ny,
        latitude0=latitude0,
        longitude0=longitude0,
        latitude_step=latitude_step,
        longitude_step=longitude_step,
    )


def _parse_label(label: bytes) -> tuple[datetime, int, int, str, int, float]:
    if len(label) != 50:
        raise ValueError("ARL record label must be exactly 50 bytes")
    text = label.decode("ascii")
    year = 2000 + int(text[0:2])
    valid_at = datetime(
        year, int(text[2:4]), int(text[4:6]), int(text[6:8]), tzinfo=UTC
    )
    return (
        valid_at,
        int(text[8:10]),
        int(text[10:12]),
        text[14:18],
        int(text[18:22]),
        float(text[36:50]),
    )


def _nearest_grid_index(value: float, origin: float, step: float, count: int) -> tuple[int, float]:
    index = round((value - origin) / step)
    if not 0 <= index < count:
        raise ValueError("requested point lies outside the ARL extraction grid")
    return index, origin + index * step


def _decode_point(packed: bytes, grid: ARLGrid, column: int, row: int, exponent: int, first: float) -> float:
    """Unpack one point using ARL's row-wise delta encoding."""
    if len(packed) != grid.nx * grid.ny:
        raise ValueError("ARL packed field length does not match the grid dimensions")
    scale = 2.0 ** (7 - exponent)
    previous = first
    target: float | None = None
    offset = 0
    for current_row in range(grid.ny):
        row_first = previous
        for current_column in range(grid.nx):
            previous += (packed[offset] - 127.0) / scale
            offset += 1
            if current_column == 0:
                row_first = previous
            if current_row == row and current_column == column:
                target = previous
        previous = row_first
    if target is None:
        raise RuntimeError("ARL point decoding did not reach the requested grid point")
    return target


def _relative_humidity_pct(temperature_c: float, specific_humidity: float, pressure_hpa: float) -> float:
    vapor_pressure_hpa = specific_humidity * pressure_hpa / (0.622 + 0.378 * specific_humidity)
    saturation_hpa = 6.112 * pow(2.718281828459045, 17.67 * temperature_c / (temperature_c + 243.5))
    return 100.0 * vapor_pressure_hpa / saturation_hpa


def read_arl_surface_forecast(
    path: Path,
    *,
    longitude: float,
    latitude: float,
    valid_at: datetime,
    forecast_hours: int = 3,
) -> ARLSurfaceForecast:
    """Decode the surface fields for a labelled, no-leak ARL forecast time."""
    if valid_at.tzinfo is None:
        raise ValueError("valid_at must include timezone information")
    valid_at = valid_at.astimezone(UTC)
    with path.open("rb") as handle:
        grid = parse_arl_grid(handle.read(158))
        record_length = 50 + grid.nx * grid.ny
        file_size = path.stat().st_size
        if file_size % record_length:
            raise ValueError("ARL file size is not a whole number of records")
        column, grid_longitude = _nearest_grid_index(
            longitude, grid.longitude0, grid.longitude_step, grid.nx
        )
        row, grid_latitude = _nearest_grid_index(latitude, grid.latitude0, grid.latitude_step, grid.ny)
        wanted = {"PRSS", "SPH2", "U10M", "V10M", "T02M", "PBLH"}
        values: dict[str, float] = {}
        for record_number in range(1, file_size // record_length):
            handle.seek(record_number * record_length)
            label = handle.read(50)
            record_valid_at, record_forecast_hours, level, variable, exponent, first = _parse_label(label)
            if (
                record_valid_at != valid_at
                or record_forecast_hours != forecast_hours
                or level != 0
                or variable not in wanted
            ):
                continue
            values[variable] = _decode_point(
                handle.read(grid.nx * grid.ny), grid, column, row, exponent, first
            )
    missing = wanted.difference(values)
    if missing:
        raise ValueError(f"ARL file does not contain requested surface fields: {sorted(missing)}")
    temperature_c = values["T02M"] - 273.15
    return ARLSurfaceForecast(
        valid_at=valid_at,
        forecast_hours=forecast_hours,
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
        temperature_c=temperature_c,
        specific_humidity_2m_kg_kg=values["SPH2"],
        relative_humidity_pct=_relative_humidity_pct(
            temperature_c, values["SPH2"], values["PRSS"]
        ),
        wind_u_ms=values["U10M"],
        wind_v_ms=values["V10M"],
        boundary_layer_height_m=values["PBLH"],
    )
