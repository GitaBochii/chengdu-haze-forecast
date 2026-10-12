import numpy as np
import pandas as pd

from scripts.rebuild_task02 import delayed_grid


def test_proxy_uses_exact_72h_shift_without_future_fill():
    times = pd.date_range('2023-01-01', periods=100, freq='h', tz='UTC')
    raw = pd.DataFrame({'timestamp': times, 'temperature_c': np.arange(100.),
                        'relative_humidity_pct': 50., 'wind_u_ms': 1., 'wind_v_ms': 2.})
    raw.loc[0, 'temperature_c'] = np.nan
    result = delayed_grid(raw, times)
    assert result.temperature_c.iloc[:73].isna().all()
    assert result.temperature_c.iloc[73] == 1.
    changed = raw.copy()
    changed.loc[28:, 'temperature_c'] = 999.
    pd.testing.assert_frame_equal(result, delayed_grid(changed, times))
