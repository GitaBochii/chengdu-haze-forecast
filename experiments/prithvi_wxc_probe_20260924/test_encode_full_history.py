import torch
import pytest

from encode_full_history import valid_feature


def test_valid_feature_requires_exact_identity():
    identity = {'pair_id': 'a', 'batch_sha256': 'b'}
    value = {**identity, 'embedding': torch.arange(2560, dtype=torch.float32).reshape(1, -1)}
    assert valid_feature(value, identity)
    assert not valid_feature(value, {**identity, 'pair_id': 'other'})


@pytest.mark.parametrize('kind', ['zero', 'nan', 'shape'])
def test_invalid_feature_rejected(kind):
    x = torch.ones(1, 2560)
    if kind == 'nan': x[0, 0] = float('nan')
    if kind == 'shape': x = x[:, :20]
    assert not valid_feature({'embedding': x}, {})
