"""Surface mean-vector wind and a control-PBL transport proxy, not layer ventilation."""

import numpy as np
import pandas as pd


def ventilation_proxy(frame):
    cols = ['gefs_mean_wind_u_ms_f024', 'gefs_mean_wind_v_ms_f024',
            'gefs_control_boundary_layer_height_m_f024']
    values = frame[cols].to_numpy(dtype=float)
    if np.isinf(values).any() or (values[:, 2] < 0).any():
        raise ValueError('Require finite-or-missing wind and nonnegative-or-missing height')
    speed = np.hypot(values[:, 0], values[:, 1])
    proxy = speed * values[:, 2]
    if np.isinf(proxy).any() or np.isinf(speed).any():
        raise ValueError('Derived feature overflow')
    return pd.DataFrame({'gefs_mean_vector_speed_f024': speed,
                         'gefs_surface_transport_proxy_f024': proxy}, index=frame.index)
