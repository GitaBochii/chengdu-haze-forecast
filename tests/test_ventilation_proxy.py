import numpy as np
import pandas as pd
import pytest

from haze_forecast.ventilation_proxy import ventilation_proxy


def test_formula_and_calm_conditions():
    frame = pd.DataFrame({'gefs_mean_wind_u_ms_f024': [3., 0.],
                          'gefs_mean_wind_v_ms_f024': [4., 0.],
                          'gefs_control_boundary_layer_height_m_f024': [100., 500.]})
    original = frame.copy()
    result = ventilation_proxy(frame)
    np.testing.assert_array_equal(result.to_numpy(), [[5., 500.], [0., 0.]])
    pd.testing.assert_frame_equal(original, frame)


def test_missing_components_propagate():
    frame = pd.DataFrame({'gefs_mean_wind_u_ms_f024': [np.nan, 3.],
                          'gefs_mean_wind_v_ms_f024': [4., 4.],
                          'gefs_control_boundary_layer_height_m_f024': [100., np.nan]})
    result = ventilation_proxy(frame)
    assert result.iloc[0].isna().all()
    assert result.iloc[1, 0] == 5. and np.isnan(result.iloc[1, 1])


@pytest.mark.parametrize('height', [-1., np.inf])
def test_invalid_physical_input_rejected(height):
    frame = pd.DataFrame({'gefs_mean_wind_u_ms_f024': [3.],
                          'gefs_mean_wind_v_ms_f024': [4.],
                          'gefs_control_boundary_layer_height_m_f024': [height]})
    with pytest.raises(ValueError):
        ventilation_proxy(frame)
