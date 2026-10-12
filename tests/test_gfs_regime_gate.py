import pandas as pd
import pytest

from haze_forecast.gfs_regime_gate import gate_mask


def _forecast_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gfs_boundary_layer_height_m_f006": [300.0, 300.0, None],
            "gfs_boundary_layer_height_m_f024": [450.0, 340.0, 600.0],
            "gfs_wind_u_ms_f006": [1.0, 1.0, 1.0],
            "gfs_wind_v_ms_f006": [0.0, 0.0, 0.0],
            "gfs_wind_u_ms_f024": [3.0, 1.3, 4.0],
            "gfs_wind_v_ms_f024": [0.0, 0.0, 0.0],
            "gfs_relative_humidity_pct_f006": [85.0, 85.0, 85.0],
            "gfs_relative_humidity_pct_f024": [70.0, 82.0, 60.0],
        }
    )


def test_gate_mask_uses_enhanced_model_only_for_complete_high_dispersion_forecasts():
    assert gate_mask(_forecast_rows(), "pbl_or_wind").tolist() == [True, False, False]


def test_gate_mask_rejects_unknown_gate_name():
    with pytest.raises(ValueError, match="unknown gate"):
        gate_mask(_forecast_rows(), "unknown")
