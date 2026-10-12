import numpy as np
import pytest

from experiments.prithvi_wxc_probe_20260924.export_input_maps import extract_fields, validate_times, checked_file, export


def test_physical_units_and_lat_major_order():
    x = np.zeros((1, 2, 4, 30, 32), dtype=np.float32)
    x[:, :, 0] = 273.15
    x[:, :, 1] = .005
    x[:, :, 2] = 3
    x[:, :, 3] = 4
    x[0, 1, 0, 1, 0] = 283.15
    fields = extract_fields(x, ['T2M', 'QV2M', 'U10M', 'V10M'])
    assert len(fields['t2m']) == 2
    assert len(fields['t2m'][0]) == 960
    assert fields['t2m'][1][32] == pytest.approx(10, abs=.0001)
    assert fields['q2m'][0][0] == pytest.approx(5, abs=.0001)
    assert fields['wind10m'][0][0] == 5


def test_nonfinite_input_rejected():
    x = np.zeros((1, 2, 4, 30, 32))
    x[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        extract_fields(x, ['T2M', 'QV2M', 'U10M', 'V10M'])


def test_future_or_reversed_weather_rejected():
    origin = '2020-01-01T14:00:00+08:00'
    validate_times(origin, ['2020-01-01T00:00:00+00:00', '2020-01-01T06:00:00+00:00'])
    for times in [['2020-01-01T06:00:00Z', '2020-01-01T12:00:00Z'],
                  ['2020-01-01T06:00:00Z', '2020-01-01T00:00:00Z']]:
        with pytest.raises(ValueError):
            validate_times(origin, times)


def test_hash_mismatch_rejected(tmp_path):
    path = tmp_path / 'batch.pt'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        checked_file(path, '0' * 64)


def test_all_real_batches_match_encoder_lineage_and_physical_sources():
    result = export()
    assert len(result['samples']) == 25
    assert result['provenance']['physical_source_cells_verified'] == 192000
    assert result['latitudes'] == np.arange(23, 38, .5).tolist()
    assert result['longitudes'] == np.arange(93.75, 113.75, .625).tolist()
    for sample in result['samples']:
        assert set(sample['fields']) == {'t2m', 'q2m', 'wind10m'}
        for pair in sample['fields'].values():
            assert np.asarray(pair).shape == (2, 960)
            assert np.isfinite(pair).all()
