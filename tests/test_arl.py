from datetime import UTC, datetime

import pytest

from haze_forecast.arl import parse_arl_grid, read_arl_surface_forecast


def _label(variable: str, *, forecast_hours: int, exponent: int, first: float) -> bytes:
    return (
        f"{21:02d}{3:2d}{23:02d}{21:02d}{forecast_hours:2d}{0:2d}{99:2d}"
        f"{variable:<4}{exponent:4d}"
        f"{1.0:14.7E}{first:14.7E}"
    ).encode("ascii")


def _grid_header() -> bytes:
    return (
        f"{'GFSQ':<4}{0:3d}{0:2d}"
        f"{40.0:7.4f}{112.0:7.3f}{1.0:7.5f}{1.0:7.5f}"
        f"{0.0:7.4f}{0.0:7.4f}{0.0:7.4f}{1.0:7.4f}{1.0:7.4f}"
        f"{30.0:7.4f}{100.0:7.3f}{0.0:7.4f}{10:3d}{12:3d}{1:3d}{4:2d}{108:4d}"
    ).encode("ascii")


def _field(first: float, nx: int, ny: int) -> bytes:
    return bytes([127]) * (nx * ny)


def test_parse_arl_grid_and_decode_surface_forecast(tmp_path):
    nx, ny = 10, 12
    index = _label("INDX", forecast_hours=0, exponent=0, first=0.0)
    payload = _grid_header() + bytes(nx * ny - 108)
    fields = {
        "PRSS": 1000.0,
        "SPH2": 0.006,
        "U10M": -1.5,
        "V10M": 2.0,
        "T02M": 283.15,
        "PBLH": 120.0,
    }
    path = tmp_path / "subset.bin"
    path.write_bytes(index + payload + b"".join(
        _label(name, forecast_hours=3, exponent=7, first=value) + _field(value, nx, ny)
        for name, value in fields.items()
    ))

    grid = parse_arl_grid(path.read_bytes()[:158])
    result = read_arl_surface_forecast(
        path,
        longitude=104.0,
        latitude=30.0,
        valid_at=datetime(2021, 3, 23, 21, tzinfo=UTC),
    )

    assert (grid.nx, grid.ny) == (10, 12)
    assert (result.grid_longitude, result.grid_latitude) == (104.0, 30.0)
    assert result.temperature_c == pytest.approx(10.0)
    assert result.relative_humidity_pct == pytest.approx(78.32, abs=0.01)
    assert result.wind_u_ms == pytest.approx(-1.5)
    assert result.boundary_layer_height_m == pytest.approx(120.0)


def test_arl_reader_rejects_a_missing_requested_field(tmp_path):
    nx, ny = 10, 12
    path = tmp_path / "incomplete.bin"
    path.write_bytes(
        _label("INDX", forecast_hours=0, exponent=0, first=0.0)
        + _grid_header()
        + bytes(nx * ny - 108)
        + _label("T02M", forecast_hours=3, exponent=7, first=283.15)
        + _field(283.15, nx, ny)
    )

    with pytest.raises(ValueError, match="does not contain"):
        read_arl_surface_forecast(
            path,
            longitude=104.0,
            latitude=30.0,
            valid_at=datetime(2021, 3, 23, 21, tzinfo=UTC),
        )
