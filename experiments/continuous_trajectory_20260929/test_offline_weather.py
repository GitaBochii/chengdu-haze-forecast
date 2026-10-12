import numpy as np
import pandas as pd
from offline_weather import weather_features


def test_weather_future_not_used_and_no_pbl():
    frame=pd.DataFrame({n:np.arange(30,dtype=float) for n in ['temperature_c','relative_humidity_pct','wind_u_ms','wind_v_ms']})
    a=weather_features(frame,np.array([23]))
    frame.loc[24:,:]=1e8
    frame['boundary_layer_height_m']=1e20
    np.testing.assert_array_equal(a,weather_features(frame,np.array([23])))
    assert a.shape==(1,225)
