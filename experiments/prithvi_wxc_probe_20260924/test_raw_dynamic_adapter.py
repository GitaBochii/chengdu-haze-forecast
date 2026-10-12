import numpy as np
import pytest
import xarray as xr
import yaml

from raw_dynamic_adapter import assemble, HERE, LAT, LON, PRODUCTS, EXPECTED_UNITS


def sample():
    config = yaml.safe_load((HERE / 'vendor/config.yaml').read_text())['data']
    stamp = np.datetime64('2020-01-02T00:00', 'm')
    products = {}
    for product, names in PRODUCTS.items():
        averaged = product.startswith('M2T')
        vertical = product == 'M2I3NVASM'
        times = stamp + np.array([-30, 30] if averaged else [0], dtype='timedelta64[m]')
        coords = dict(time=times, lat=LAT, lon=LON)
        dims = ('time', 'lat', 'lon')
        shape = (len(times), len(LAT), len(LON))
        if vertical:
            coords['lev'] = np.arange(34, 73)
            dims = ('time', 'lev', 'lat', 'lon')
            shape = (1, 39, len(LAT), len(LON))
        data = {}
        for name in names:
            values = np.ones(shape, dtype='float32')
            if averaged:
                values[0] = 2
                values[1] = 6
            if vertical:
                values[:] = coords['lev'][None, :, None, None]
            data[name] = (dims, values, {'units': EXPECTED_UNITS[name]})
        products[product] = xr.Dataset(data, coords=coords)
    return config, stamp, products


def test_channel_and_level_order_and_midnight_mean():
    config, stamp, products = sample()
    result = assemble(products, stamp, config)
    assert result.shape == (160, 30, 32)
    assert result.dtype == np.float32
    assert np.all(result[0] == 4)
    np.testing.assert_array_equal(result[20:34, 0, 0], config['levels'])
    reversed_products = dict(products)
    reversed_products['M2I3NVASM'] = products['M2I3NVASM'].isel(lev=slice(None, None, -1))
    np.testing.assert_array_equal(assemble(reversed_products, stamp, config), result)


@pytest.mark.parametrize('damage', ['missing_time', 'duplicate_time', 'units', 'nan', 'fill', 'grid', 'level'])
def test_rejects_invalid_inputs(damage):
    config, stamp, products = sample()
    ds = products['M2T1NXLND']
    if damage == 'missing_time':
        products['M2T1NXLND'] = ds.isel(time=[1])
    elif damage == 'duplicate_time':
        products['M2T1NXLND'] = xr.concat([ds, ds.isel(time=[0])], dim='time')
    elif damage == 'units':
        ds['LAI'].attrs['units'] = 'wrong'
    elif damage == 'nan':
        ds['LAI'].values[0, 0, 0] = np.nan
    elif damage == 'fill':
        ds['LAI'].values[0, 0, 0] = 1e15
    elif damage == 'grid':
        products['M2T1NXLND'] = ds.assign_coords(lat=LAT + .1)
    elif damage == 'level':
        products['M2I3NVASM'] = products['M2I3NVASM'].sel(lev=slice(35, 72))
    with pytest.raises(ValueError):
        assemble(products, stamp, config)
