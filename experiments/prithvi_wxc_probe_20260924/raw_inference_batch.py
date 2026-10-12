"""Unnormalized NumPy inference contract matching the frozen regional loader."""
from datetime import timedelta, timezone

import numpy as np

from raw_dynamic_adapter import LAT, LON
from import_climatology_regions import validate_arrays


def utc(time):
    if time.tzinfo is None or time.utcoffset() is None:
        raise ValueError('Timezone-aware weather time required')
    time = time.astimezone(timezone.utc)
    if time.minute or time.second or time.microsecond or time.hour % 6:
        raise ValueError('Exact six-hour weather time required')
    return time


def climate_key(latest):
    target = utc(latest) + timedelta(hours=6)
    return min(target.timetuple().tm_yday, 365), target.hour


def checked(array, shape):
    value = np.asarray(array, dtype='float32')
    if value.shape != shape or not np.isfinite(value).all() or np.any(np.abs(value) >= 1e14):
        raise ValueError('Invalid input array')
    return value


def auxiliary(latest, constants, surface, vertical, config):
    latest = utc(latest)
    validate_arrays(surface, config['surface_vars'], None)
    validate_arrays(vertical, config['vertical_vars'], config['levels'])
    if set(constants) != set(config['static_surface_vars']):
        raise ValueError('Exactly four configured static fields required')
    # Match official float32 coordinate arithmetic before trigonometric encoding.
    lat, lon = np.meshgrid(LAT.astype('float32'), LON.astype('float32'), indexing='ij')
    positions = np.stack([lat / 360. * 2.0 * np.pi, lon / 360. * 2.0 * np.pi])
    result = np.empty((10, 30, 32), dtype='float32')
    result[:2] = positions
    day, hour = latest.timetuple().tm_yday, latest.hour
    for i, value in enumerate([np.cos(2*np.pi*day/366), np.sin(2*np.pi*day/366),
                               np.cos(2*np.pi*hour/24), np.sin(2*np.pi*hour/24)], 2):
        result[i] = value
    for i, name in enumerate(config['static_surface_vars'], 6):
        result[i] = checked(constants[name], (30, 32))
    indices = [np.flatnonzero(vertical['lev'] == level).item() for level in config['levels']]
    climate = np.concatenate([
        np.stack([surface[name] for name in config['surface_vars']]),
        np.stack([vertical[name][indices] for name in config['vertical_vars']]).reshape(140, 30, 32)
    ])
    return checked(result, (10, 30, 32)), checked(climate, (160, 30, 32))


def pack(dynamic, static, climate):
    return {'x': checked(dynamic, (2, 160, 30, 32))[None],
            'static': checked(static, (10, 30, 32))[None],
            'climate': checked(climate, (160, 30, 32))[None],
            'input_time': np.array([6], dtype='float32'),
            'lead_time': np.array([6], dtype='float32')}


def gate_status(rows):
    return bool(rows) and all(row['status'] == 'exact' for row in rows)
