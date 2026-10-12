import numpy as np
import pandas as pd
from isd_wind import corrected_grid


def test_scalar_speed_preserved_without_inventing_direction():
    f=pd.DataFrame(dict(temperature_c=[20]*4,relative_humidity_pct=[50]*4,
                       wind_u_ms=[np.nan,np.nan,2.,np.nan],wind_v_ms=[np.nan,np.nan,0.,np.nan],
                       noaa_wind_speed_ms=[0.,1.,2.,np.nan]))
    g=corrected_grid(f)
    np.testing.assert_array_equal(g[0,2:],[0,0,0])
    assert np.isnan(g[1,2:4]).all() and g[1,4]==1
    np.testing.assert_array_equal(g[2,2:],[2,0,2])
    assert np.isnan(g[3,2:]).all()
