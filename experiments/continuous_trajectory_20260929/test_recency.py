import pandas as pd
import numpy as np
from recency import recency_weights


def test_weights_only_depend_on_past_age():
    end=pd.Timestamp('2020-01-01',tz='Asia/Shanghai')
    ts=pd.DatetimeIndex([end-pd.Timedelta(days=730.5),end])
    np.testing.assert_allclose(recency_weights(ts,end,2),[.5,1.])
