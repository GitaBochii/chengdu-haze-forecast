import pytest

from full_history_batch_loader import describe_origin, FullHistoryBatchLoader, validate_origin_mapping


def test_exact_origin_respects_support_end():
    item = describe_origin('2020-01-02T06:00:00+00:00')
    assert item['latest'] == '2020-01-02T00:00:00+00:00'
    assert item['earlier'] == '2020-01-01T18:00:00+00:00'
    assert item['support_end'] == '2020-01-02T01:00:00+00:00'
    assert not item['year_start_boundary_uncertain']


def test_next_hour_accepts_new_pair():
    a = describe_origin('2020-01-02T15:00:00+08:00')
    b = describe_origin('2020-01-02T07:00:00+00:00')
    assert a == b
    assert a['latest'] == '2020-01-02T06:00:00+00:00'
    assert a['support_end'] == b['origin']
    assert a['climate_key'] == [2, 12]


def test_boundary_is_not_discarded():
    row = describe_origin('2020-01-01T07:00:00+00:00')
    assert row['year_start_boundary_uncertain']
    assert row['earlier'] == '2020-01-01T00:00:00+00:00'


@pytest.mark.parametrize('stamp', ['2020-01-01T00:00:00', '2020-01-01T00:01:00+00:00'])
def test_invalid_origin_rejected(stamp):
    with pytest.raises(ValueError):
        describe_origin(stamp)


def test_unknown_origin_is_not_silently_added():
    loader = FullHistoryBatchLoader.__new__(FullHistoryBatchLoader)
    loader.origins = {}
    with pytest.raises(ValueError, match='outside frozen'):
        loader.load('2020-01-02T07:00:00+00:00')


def test_wrong_dependency_rejected_before_reading():
    loader = FullHistoryBatchLoader.__new__(FullHistoryBatchLoader)
    loader.origins = {'2020-01-02T07:00:00+00:00':
                     {'files': ['wrong.npz'], 'year_start_boundary_uncertain': False}}
    with pytest.raises(ValueError, match='dependency mapping'):
        loader.load('2020-01-02T07:00:00+00:00')


def test_tampered_file_rejected(tmp_path):
    loader = FullHistoryBatchLoader.__new__(FullHistoryBatchLoader)
    loader.hashes = {}
    path = tmp_path / 'input.bin'
    path.write_bytes(b'original')
    digest = loader.verify(path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        loader.verify(path, digest)
    with pytest.raises(ValueError, match='changed during batch'):
        loader.verify_unchanged()


@pytest.mark.parametrize('damage', ['none', 'missing', 'partition', 'duplicate_source'])
def test_loader_origin_mapping_is_bound_to_source(damage):
    stamp = '2020-01-02T07:00:00+00:00'
    source = [{'timestamp': stamp, 'partition': 'train'}]
    metadata = describe_origin(stamp)
    origins = {stamp: {'origin': stamp, 'partition': 'train',
                      'files': metadata['dynamic_files'], 'year_start_boundary_uncertain': False}}
    if damage == 'missing': origins = {}
    if damage == 'partition': origins[stamp]['partition'] = 'test'
    if damage == 'duplicate_source': source *= 2
    if damage == 'none':
        validate_origin_mapping(origins, source)
    else:
        with pytest.raises(ValueError):
            validate_origin_mapping(origins, source)
