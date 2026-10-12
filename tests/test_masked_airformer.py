import numpy as np
import torch

from haze_forecast.regional_airformer import create_airformer
from scripts.intervene_inactive_spatial import spatial_support
from scripts.masked_airformer import MaskedAirFormer


def test_all_active_and_mixed_masks_match_existing_intervention():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = create_airformer(17, np.array([[104., 30.], [104.1, 30.1], [104.2, 30.2]])).eval()
    x = torch.randn(2, 24, 3, 17)
    with torch.no_grad():
        original = base(x)
        with spatial_support(base, [True, False, True]):
            altered = base(x)
    n = sum(p.numel() for p in base.parameters())
    wrapped = MaskedAirFormer(base).eval()
    flags = torch.ones(2, 24, 3, 1)
    with torch.no_grad():
        torch.testing.assert_close(wrapped(torch.cat([x, flags], dim=-1)), original, rtol=0, atol=0)
        flags[1, :, 1] = 0
        result = wrapped(torch.cat([x, flags], dim=-1))
    torch.testing.assert_close(result[0], original[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result[1], altered[1], rtol=1e-5, atol=1e-6)
    assert sum(p.numel() for p in wrapped.parameters()) == n


def test_masked_training_has_finite_gradients():
    torch.set_num_threads(1)
    base = create_airformer(17, np.array([[104., 30.], [104.1, 30.1]])).train()
    model = MaskedAirFormer(base)
    x = torch.randn(2, 24, 2, 18)
    x[..., -1] = 1
    x[1, :, 1, -1] = 0
    model(x).abs().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_all_active_preserves_initialization_and_training_gradients():
    torch.set_num_threads(1)
    coords = np.array([[104., 30.], [104.1, 30.1]])
    torch.manual_seed(42)
    control = create_airformer(17, coords).train()
    torch.manual_seed(42)
    candidate = MaskedAirFormer(create_airformer(17, coords)).train()
    for left, right in zip(control.parameters(), candidate.parameters(), strict=True):
        assert torch.equal(left, right)
    x = torch.randn(2, 24, 2, 17)
    torch.manual_seed(7)
    left = control(x)
    torch.manual_seed(7)
    right = candidate(torch.cat([x, torch.ones(2, 24, 2, 1)], dim=-1))
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.abs().mean().backward()
    right.abs().mean().backward()
    for a, b in zip(control.parameters(), candidate.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
