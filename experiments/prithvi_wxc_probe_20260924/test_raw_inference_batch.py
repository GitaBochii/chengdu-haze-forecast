from datetime import datetime, timezone

import numpy as np
import pytest
import yaml

from raw_inference_batch import auxiliary, pack, climate_key, gate_status
from raw_dynamic_adapter import HERE, LAT, LON


def inputs():
    config = yaml.safe_load((HERE / 'vendor/config.yaml').read_text())['data']
    static = {name: np.ones((30, 32), dtype='float32') for name in config['static_surface_vars']}
    surface = dict(lat=LAT, lon=LON)
    surface.update({name: np.ones((30, 32), dtype='float32') for name in config['surface_vars']})
    levels = np.array(config['levels'][::-1])
    vertical = dict(lat=LAT, lon=LON, lev=levels)
    vertical.update({name: np.broadcast_to(levels[:, None, None], (14, 30, 32)).copy()
                     for name in config['vertical_vars']})
    return config, static, surface, vertical


def test_position_calendar_and_climate_order():
    config, static, surface, vertical = inputs()
    stamp = datetime(2020, 1, 2, 6, tzinfo=timezone.utc)
    s, c = auxiliary(stamp, static, surface, vertical, config)
    assert s.shape == (10, 30, 32)
    assert c.shape == (160, 30, 32)
    np.testing.assert_array_equal(c[20:34, 0, 0], config['levels'])
    np.testing.assert_allclose(s[0, :, 0], LAT / 180 * np.pi, rtol=1e-6)
    assert s[2, 0, 0] == np.float32(np.cos(2 * np.pi * 2 / 366))
    assert s[5, 0, 0] == 1
    batch = pack(np.zeros((2, 160, 30, 32), dtype='float32'), s, c)
    assert set(batch) == {'x', 'static', 'climate', 'input_time', 'lead_time'}
    assert batch['x'].shape == (1, 2, 160, 30, 32)


def test_calendar_preserves_official_leap_cap():
    assert climate_key(datetime(2020, 12, 31, 12, tzinfo=timezone.utc)) == (365, 18)
    assert climate_key(datetime(2020, 12, 31, 18, tzinfo=timezone.utc)) == (1, 0)
    with pytest.raises(ValueError):
        climate_key(datetime(2020, 1, 1))


@pytest.mark.parametrize('statuses, expected', [
    ([], False), (['missing_overlap'], False), (['exact', 'different'], False),
    (['exact', 'missing_overlap'], False), (['exact', 'exact'], True),
    (['assembled'], False)])
def test_gate_requires_all_rows_compared_and_exact(statuses, expected):
    assert gate_status([{'status': s} for s in statuses]) == expected


@pytest.mark.parametrize('field', ['static', 'grid', 'level'])
def test_invalid_auxiliary_rejected(field):
    config, static, surface, vertical = inputs()
    if field == 'static':
        static['PHIS'][0, 0] = np.nan
    elif field == 'grid':
        surface['lat'] = LAT[::-1]
    else:
        vertical['lev'][0] = vertical['lev'][1]
    with pytest.raises(ValueError):
        auxiliary(datetime(2020, 1, 2, tzinfo=timezone.utc), static, surface, vertical, config)
