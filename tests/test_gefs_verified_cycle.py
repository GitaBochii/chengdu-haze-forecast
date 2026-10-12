from datetime import UTC, datetime

import pytest

from scripts.collect_gefs_verified_cycle import validate_inventory, validate_value

RUN = datetime(2026, 9, 10, tzinfo=UTC)


def test_rejects_wrong_member_and_wrong_run():
    text = '1:0:d=2026091000:TMP:2 m above ground:24 hour fcst:ens mean'
    validate_inventory(text, RUN, 'geavg', 'temperature_2m')
    with pytest.raises(ValueError):
        validate_inventory(text, RUN, 'gespr', 'temperature_2m')
    with pytest.raises(ValueError):
        validate_inventory(text.replace('2026091000', '2026091006'), RUN, 'geavg', 'temperature_2m')


@pytest.mark.parametrize('product,field,value', [
    ('gespr', 'temperature_2m', -1), ('geavg', 'relative_humidity_2m', 110),
    ('gec00', 'pblh', 99999), ('geavg', 'temperature_2m', float('nan')),
])
def test_rejects_invalid_value(product, field, value):
    with pytest.raises(ValueError):
        validate_value(product, field, value)
