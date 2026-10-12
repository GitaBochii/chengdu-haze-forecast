from datetime import UTC, datetime

import pytest

from scripts.verify_gefs_received_cycle import validate_grid_units

TEXT = '''1:0:vt=2026091100:2 m above ground:24 hour fcst:TMP Temperature [K]:ens mean
grid_template=0:winds(N/S):
lat-lon grid:(720 x 361)
lat 90.000000 to -90.000000 by 0.500000
lon 0.000000 to 359.500000 by 0.500000 #points=259920'''


def test_accepts_expected_full_grid_and_units():
    validate_grid_units(TEXT, 'temperature_2m', datetime(2026, 9, 11, tzinfo=UTC))


@pytest.mark.parametrize('old,new', [('720 x 361', '360 x 181'), ('[K]', '[C]'),
                                    ('2026091100', '2026091200'), ('winds(N/S)', 'winds(grid)')])
def test_rejects_wrong_grid_units_time_or_wind_reference(old, new):
    with pytest.raises(ValueError):
        validate_grid_units(TEXT.replace(old, new), 'temperature_2m', datetime(2026, 9, 11, tzinfo=UTC))
