from datetime import UTC, datetime

from haze_forecast.forecast_extraction import extract_gfs_surface_forecast_at_point
from haze_forecast.forecast_sources import GFSFieldMessage
from haze_forecast.grib import GRIBPointValue


def test_surface_forecast_extraction_converts_temperature_and_keeps_grid_location(tmp_path):
    messages = {
        name: GFSFieldMessage(name, "https://example.test/file", (0, 3), b"GRIB")
        for name in (
            "temperature_2m",
            "relative_humidity_2m",
            "u_wind_10m",
            "v_wind_10m",
            "pblh",
        )
    }
    values = {
        "temperature_2m": 300.15,
        "relative_humidity_2m": 60.0,
        "u_wind_10m": -1.5,
        "v_wind_10m": 2.5,
        "pblh": 800.0,
    }

    def downloader(*_args, **_kwargs):
        return messages

    def point_query(path, *_args, **_kwargs):
        assert path.read_bytes() == b"GRIB"
        return GRIBPointValue(longitude=104.0, latitude=30.75, value=values[path.stem])

    forecast = extract_gfs_surface_forecast_at_point(
        datetime(2021, 3, 23, 18, tzinfo=UTC),
        lead_hours=6,
        longitude=104.06,
        latitude=30.67,
        wgrib2_executable=tmp_path / "wgrib2.exe",
        message_downloader=downloader,
        point_query=point_query,
    )

    assert forecast.valid_at == datetime(2021, 3, 24, 0, tzinfo=UTC)
    assert forecast.grid_longitude == 104.0
    assert forecast.grid_latitude == 30.75
    assert forecast.temperature_c == 27.0
    assert forecast.relative_humidity_pct == 60.0
    assert forecast.wind_u_ms == -1.5
    assert forecast.wind_v_ms == 2.5
    assert forecast.boundary_layer_height_m == 800.0
