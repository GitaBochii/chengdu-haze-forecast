from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from haze_forecast.grib import GRIBPointValue
from haze_forecast.operational_gefs import GEFSFieldMessage
from haze_forecast.operational_gefs_dataset import enrich_event_rows_with_operational_gefs


def _fake_messages(
    run_time: datetime, *, product: str, lead_hours: int, **_: object
) -> dict[str, GEFSFieldMessage]:
    return {
        field: GEFSFieldMessage(
            product=product,
            feature_name=field,
            run_time=run_time,
            lead_hours=lead_hours,
            valid_time=run_time + timedelta(hours=lead_hours),
            source_url="https://example.invalid/data",
            index_url="https://example.invalid/data.idx",
            byte_range=(0, 1),
            content=field.encode(),
            sha256="test",
        )
        for field in ("temperature_2m", "relative_humidity_2m", "u_wind_10m", "v_wind_10m", "pblh")
    }


def test_enrichment_adds_mean_spread_features_and_keeps_timing_evidence():
    events = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 14:00"], utc=True)})

    result = enrich_event_rows_with_operational_gefs(
        events,
        longitude=104.0665,
        latitude=30.5723,
        wgrib2_executable=Path("unused.exe"),
        downloader=_fake_messages,
        point_query=lambda *args, **kwargs: GRIBPointValue(104.0, 30.5, 12.0),
    )

    assert result.loc[0, "gefs_mean_temperature_c_f024"] == 12.0 - 273.15
    assert result.loc[0, "gefs_control_boundary_layer_height_m_f024"] == 12.0
    assert result.loc[0, "gefs_run_utc"] == pd.Timestamp("2024-01-01 12:00", tz="UTC")
    assert result.loc[0, "gefs_forecast_valid_utc_f024"] == pd.Timestamp(
        "2024-01-02 12:00", tz="UTC"
    )


def test_enrichment_keeps_rows_with_missing_features_when_one_gefs_run_fails():
    events = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 14:00"], utc=True)})

    result = enrich_event_rows_with_operational_gefs(
        events,
        longitude=104.0665,
        latitude=30.5723,
        wgrib2_executable=Path("unused.exe"),
        downloader=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("archive unavailable")),
    )

    assert result["gefs_mean_temperature_c_f006"].isna().all()
    assert result.loc[0, "gefs_extraction_error"] == "archive unavailable"
