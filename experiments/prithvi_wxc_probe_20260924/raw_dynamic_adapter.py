"""Raw physical dynamic channels; no scaling, padding, targets or boundary clamp."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.import_merra_history import PRODUCTS, EXPECTED_UNITS
from scripts.download_merra_regional_sample import LAT, LON


def assemble(products, time, config):
    stamp = np.datetime64(time, 'ns')
    if np.isnat(stamp) or stamp != stamp.astype('datetime64[h]'):
        raise ValueError('Exact integer-hour UTC analysis time required')
    if set(products) != set(PRODUCTS):
        raise ValueError('Exactly five raw products required')
    if (len(set(config['surface_vars'])) != 20 or len(set(config['vertical_vars'])) != 10
            or len(set(config['levels'])) != 14):
        raise ValueError('Expected frozen 160-channel contract')
    fields = {}
    for product, names in PRODUCTS.items():
        ds = products[product]
        for key, expected in [('lat', LAT), ('lon', LON)]:
            if key not in ds.coords or not np.array_equal(ds[key].values, expected):
                raise ValueError('Unexpected coordinates: ' + key)
        if 'time' not in ds.coords or not ds.indexes['time'].is_unique:
            raise ValueError('Missing or duplicate time coordinates')
        vertical = product == 'M2I3NVASM'
        averaged = product.startswith('M2T')
        times = [stamp - np.timedelta64(30, 'm'), stamp + np.timedelta64(30, 'm')] if averaged else [stamp]
        try:
            selected = ds.sel(time=times)
            if vertical:
                if not ds.indexes['lev'].is_unique:
                    raise ValueError('Duplicate levels')
                # Frozen prepared files descend; the official reader flips them.
                # Select physical identifiers, never reuse raw storage indices.
                selected = selected.sel(lev=config['levels'])
        except KeyError as exc:
            raise ValueError('Required exact source time or level absent') from exc
        for name in names:
            if name not in selected:
                raise ValueError('Missing variable: ' + name)
            unit = selected[name].attrs.get('units')
            if isinstance(unit, bytes):
                unit = unit.decode()
            if unit != EXPECTED_UNITS[name]:
                raise ValueError('Unexpected units: ' + name)
            dims = ('time', 'lev', 'lat', 'lon') if vertical else ('time', 'lat', 'lon')
            values = selected[name].transpose(*dims).values.astype('float32')
            if not np.isfinite(values).all() or np.any(np.abs(values) >= 1e14):
                raise ValueError('Invalid values: ' + name)
            fields[name] = values.mean(axis=0, dtype='float32') if averaged else values[0]
    surface = np.stack([fields[name] for name in config['surface_vars']])
    vertical = np.stack([fields[name] for name in config['vertical_vars']]).reshape(140, 30, 32)
    return np.ascontiguousarray(np.concatenate([surface, vertical]))
