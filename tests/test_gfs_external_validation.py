from __future__ import annotations

import pandas as pd
import pytest

from haze_forecast.gfs_external_validation import validate_gfs_forecast_timing


def test_validate_gfs_forecast_timing_accepts_future_valid_time() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-01T06:00:00Z")],
            "gfs_forecast_run_utc": [pd.Timestamp("2023-01-01T00:00:00Z")],
            "gfs_forecast_valid_utc": [pd.Timestamp("2023-01-01T12:00:00Z")],
        }
    )

    validate_gfs_forecast_timing(frame)


def test_validate_gfs_forecast_timing_rejects_a_future_issued_run() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-01T06:00:00Z")],
            "gfs_forecast_run_utc": [pd.Timestamp("2023-01-01T12:00:00Z")],
            "gfs_forecast_valid_utc": [pd.Timestamp("2023-01-01T18:00:00Z")],
        }
    )

    with pytest.raises(ValueError, match="issued after"):
        validate_gfs_forecast_timing(frame)
