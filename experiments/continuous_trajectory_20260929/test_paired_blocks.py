import numpy as np
from paired_blocks import counts,differences


def test_counts_and_paired_direction():
    y=np.array([1,1,0,0],bool)
    a=np.array([1,0,1,0],bool)
    b=y.copy()
    np.testing.assert_array_equal(counts(y,a).sum(axis=0),[1,1,1,1])
    np.testing.assert_allclose(differences(np.r_[counts(y,a).sum(axis=0),counts(y,b).sum(axis=0)]),[.5,-.5])
