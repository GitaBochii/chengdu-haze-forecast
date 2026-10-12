import numpy as np
from season_decision import sufficient


def test_minimum_class_counts():
    assert sufficient(np.array([1]*20+[0]*50,dtype=bool))
    assert not sufficient(np.array([1]*19+[0]*50,dtype=bool))
    assert not sufficient(np.array([1]*20+[0]*49,dtype=bool))
