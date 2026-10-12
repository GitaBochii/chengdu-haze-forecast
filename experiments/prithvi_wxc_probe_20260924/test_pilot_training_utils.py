import numpy as np
import pytest
import torch

from experiments.prithvi_wxc_probe_20260924.pilot_training_utils import (
    fit_spatial_pca, transform_spatial, block_weights, error_summary)


def test_pca_fits_training_only_and_roundtrips():
    torch.manual_seed(17)
    x = torch.randn(25, 2560)
    train = torch.arange(25) < 14
    state = fit_spatial_pca(x, train)
    changed = x.clone()
    changed[~train] = 100000
    again = fit_spatial_pca(changed, train)
    for key in state:
        torch.testing.assert_close(state[key], again[key], rtol=0, atol=0)
    z = transform_spatial(x, state)
    assert z.shape == (25, 4)
    torch.testing.assert_close(z[train].mean(0), torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(z[train].std(0, correction=0), torch.ones(4))
    assert torch.isfinite(z).all()


def test_pca_constant_features_and_invalid_training():
    x = torch.ones(8, 10)
    train = torch.arange(8) < 5
    state = fit_spatial_pca(x, train)
    assert torch.equal(transform_spatial(x, state), torch.zeros(8, 4))
    with pytest.raises(ValueError):
        fit_spatial_pca(x, torch.arange(8) < 3)
    with pytest.raises(ValueError):
        fit_spatial_pca(x * float('nan'), train)


def test_equal_block_weight_not_equal_rows():
    w = block_weights([1, 1, 1, 2])
    torch.testing.assert_close(w, torch.tensor([1/6, 1/6, 1/6, 1/2]))


def test_interval_errors_tail_and_macro_counts():
    rows = [dict(lower_bound_hours=3, upper_bound_hours=5, reason='interval', continuity_block=1),
            dict(lower_bound_hours=10, upper_bound_hours=10, reason='exact', continuity_block=1),
            dict(lower_bound_hours=84, upper_bound_hours=84, reason='exact', continuity_block=2)]
    out = error_summary(rows, [4, 8, 60])
    assert out['row_count'] == 3 and out['block_count'] == 2
    assert out['original_exact_count'] == 2
    assert out['exact_row_mae_capped_hours'] == 5
    assert np.isclose(out['row_error_lower_hours'], 10/3)
    assert np.isclose(out['row_error_upper_hours'], 11/3)
    assert out['block_error_lower_hours'] == 4.5
    assert out['block_error_upper_hours'] == 4.75
    with pytest.raises(ValueError):
        error_summary(rows, [4, 8])
