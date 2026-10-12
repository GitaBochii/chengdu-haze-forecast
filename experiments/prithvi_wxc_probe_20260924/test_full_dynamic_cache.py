import numpy as np
import pytest

from prepare_full_dynamic_cache import boundary_flag, validate_cube, cache_name


def test_boundary_flag_is_narrow():
    assert boundary_flag('2020-01-01T00:00:00+00:00')
    assert boundary_flag('2020-01-01T08:00:00+08:00')
    assert not boundary_flag('2020-01-01T06:00:00+00:00')
    assert not boundary_flag('2020-02-01T00:00:00+00:00')
    with pytest.raises(ValueError):
        boundary_flag('2020-01-01T00:00:00')


def test_cache_name_is_utc():
    assert cache_name('2020-01-01T08:00:00+08:00') == '20200101T0000.npz'


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'nan'])
def test_invalid_cube_rejected(bad):
    x = np.zeros((160, 30, 32), dtype='float32')
    if bad == 'shape': x = x[:159]
    if bad == 'dtype': x = x.astype('float64')
    if bad == 'nan': x[0, 0, 0] = np.nan
    with pytest.raises(ValueError): validate_cube(x)
