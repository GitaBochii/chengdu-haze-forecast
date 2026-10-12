import numpy as np
import pytest

torch = pytest.importorskip("torch")

from haze_forecast.regional_airformer import create_airformer


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("nodes", [1, 3])
def test_airformer_has_official_attention_and_finite_backward(nodes):
    coords = np.array([[104.0 + i * 0.2, 30.6] for i in range(nodes)])
    model = create_airformer(5, coords, history=8, horizon=6, hidden=8)
    inputs = torch.randn(2, 8, nodes, 5, requires_grad=True)
    output = model(inputs)
    assert output.shape == (2, 6, nodes, 2)
    assert torch.isfinite(output).all()
    assert any(type(module).__name__ == "DS_MSA" for module in model.modules())
    assert any(type(module).__name__ == "CT_MSA" for module in model.modules())
    assert model.stochastic_flag is False
    output.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_airformer_eval_is_deterministic_and_handles_default_horizon():
    model = create_airformer(4, np.array([[104.0, 30.6]]), hidden=8).eval()
    inputs = torch.randn(1, 24, 1, 4)
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs)
    assert first.shape == (1, 72, 1, 2)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_airformer_partition_is_normalized_and_never_masks_all_sectors():
    coords = np.array([[104.0, 30.6], [104.1, 30.6], [105.0, 30.6]])
    model = create_airformer(3, coords, history=7, horizon=2, hidden=8)
    assignment = model.assignment
    assert assignment.shape == (3, 3, 17)
    torch.testing.assert_close(assignment.sum(dim=1), (~model.mask).float())
    assert not model.mask[:, 0].any()
    torch.testing.assert_close(assignment[:, :, 0], torch.eye(3))
    assert model(torch.randn(2, 7, 3, 3)).shape == (2, 2, 3, 2)


def test_airformer_geometry_and_causal_masks_are_registered_buffers():
    model = create_airformer(3, np.array([[104.0, 30.6]]), hidden=8)
    for module in model.modules():
        for name in ("assignment", "mask"):
            if isinstance(getattr(module, name, None), torch.Tensor):
                assert name in module._buffers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_airformer_moves_all_attention_masks_to_cuda():
    model = create_airformer(3, np.array([[104.0, 30.6]]), hidden=8).cuda()
    output = model(torch.randn(2, 24, 1, 3, device="cuda"))
    output.square().mean().backward()
    assert torch.isfinite(output).all()


def test_airformer_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match="coordinates"):
        create_airformer(3, np.array([[104.0, np.nan]]))
