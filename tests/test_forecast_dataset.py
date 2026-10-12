from datetime import UTC, datetime, timedelta

import pandas as pd

from haze_forecast.forecast_dataset import enrich_event_rows_with_gfs_forecast
from haze_forecast.forecast_extraction import GFSPointForecast


def test_event_enrichment_uses_latest_issued_gfs_cycle_without_future_runs(tmp_path):
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-03-13 01:00:00+08:00", "2021-03-13 05:00:00+08:00"]
            ),
            "remaining_hours": [10.0, 6.0],
        }
    )
    calls = []

    def extractor(run_time, lead_hours, *_args, **_kwargs):
        calls.append((run_time, lead_hours))
        return GFSPointForecast(
            run_time=run_time,
            valid_at=run_time + timedelta(hours=lead_hours),
            lead_hours=lead_hours,
            grid_longitude=104.0,
            grid_latitude=30.75,
            temperature_c=12.5,
            relative_humidity_pct=55.0,
            wind_u_ms=-1.0,
            wind_v_ms=2.0,
            boundary_layer_height_m=600.0,
        )

    enriched = enrich_event_rows_with_gfs_forecast(
        events,
        longitude=104.06,
        latitude=30.67,
        wgrib2_executable=tmp_path / "wgrib2.exe",
        lead_hours=6,
        extractor=extractor,
        max_workers=2,
    )

    assert set(calls) == {
        (datetime(2021, 3, 12, 12, tzinfo=UTC), 6),
        (datetime(2021, 3, 12, 18, tzinfo=UTC), 6),
    }
    assert enriched["gfs_forecast_run_utc"].tolist() == [
        pd.Timestamp("2021-03-12 12:00:00+00:00"),
        pd.Timestamp("2021-03-12 18:00:00+00:00"),
    ]
    assert enriched["gfs_forecast_run_utc"].le(
        pd.to_datetime(enriched["timestamp"], utc=True)
    ).all()
    assert enriched["gfs_temperature_c_f006"].tolist() == [12.5, 12.5]


def test_event_enrichment_names_features_for_the_requested_forecast_horizon(tmp_path):
    events = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2021-03-13 01:00:00+08:00"]), "remaining_hours": [10.0]}
    )

    def extractor(run_time, lead_hours, *_args, **_kwargs):
        return GFSPointForecast(
            run_time=run_time,
            valid_at=run_time + timedelta(hours=lead_hours),
            lead_hours=lead_hours,
            grid_longitude=104.0,
            grid_latitude=30.75,
            temperature_c=12.5,
            relative_humidity_pct=55.0,
            wind_u_ms=-1.0,
            wind_v_ms=2.0,
            boundary_layer_height_m=600.0,
        )

    enriched = enrich_event_rows_with_gfs_forecast(
        events,
        longitude=104.06,
        latitude=30.67,
        wgrib2_executable=tmp_path / "wgrib2.exe",
        lead_hours=24,
        extractor=extractor,
    )

    assert "gfs_temperature_c_f024" in enriched.columns
    assert "gfs_temperature_c_f006" not in enriched.columns
