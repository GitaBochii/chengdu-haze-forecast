import numpy as np
import pytest

torch = pytest.importorskip("torch")

from haze_forecast.regional_air_networks import (
    PM25GNNAdapter,
    PM25GNNHistoryAdapter,
    SequenceGRU,
    directional_edges,
    transport_weight,
)


def test_westerly_wind_flows_east_not_west():
    edges, attrs = directional_edges(np.array([[104.0, 30.0], [104.1, 30.0]]))
    east = np.flatnonzero((edges[0] == 0) & (edges[1] == 1))[0]
    west = np.flatnonzero((edges[0] == 1) & (edges[1] == 0))[0]
    # Meteorological directions specify where the wind comes FROM.
    weights = transport_weight(torch.tensor([10., 10.]), torch.tensor([270., 270.]),
                               torch.tensor(attrs[:, 0]), torch.tensor(attrs[:, 1]))
    assert weights[east] > 0
    assert weights[west] == 0


def test_pm25_official_adapter_shape_backward_and_variable_batch():
    model = PM25GNNAdapter(
        input_dim=5, node_coords=np.array([[104., 30.], [104.1, 30.], [104., 30.1]]),
        wind_mean=np.array([10., 270.]), wind_scale=np.array([1., 1.]),
        history=4, horizon=6, target_indices=(0, 2),
    )
    for batch in (2, 1):
        x = torch.zeros(batch, 4, 3, 5)
        out = model(x)
        assert out.shape == (batch, 6, 3, 2)
        assert torch.isfinite(out).all()
        out.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_gru_baseline_same_output_contract():
    model = SequenceGRU(input_dim=5, horizon=6, output_dim=2)
    assert model(torch.zeros(2, 4, 3, 5)).shape == (2, 6, 3, 2)


def test_hourly_transport_and_checkpoint_graph_metadata():
    weight = transport_weight(torch.tensor([10.]), torch.tensor([270.]),
                              torch.tensor([10.]), torch.tensor([270.]))
    torch.testing.assert_close(weight, torch.ones(1))
    model = PM25GNNAdapter(5, np.array([[104., 30.], [104.1, 30.]]),
                           np.array([10., 270.]), np.ones(2), history=4, horizon=6)
    state = model.state_dict()
    assert 'heads.0.graph_gnn.edge_attr' in state
    assert 'heads.0.graph_gnn.wind_mean' in state


def paired_models(history=4):
    args = (5, np.array([[104., 30.], [104.1, 30.], [104., 30.1]]),
            np.array([10., 270.]), np.ones(2))
    torch.manual_seed(42)
    baseline = PM25GNNAdapter(*args, history=history, horizon=6)
    torch.manual_seed(42)
    warmed = PM25GNNHistoryAdapter(*args, history=history, horizon=6)
    return baseline, warmed


def test_history_warmup_uses_earlier_frames_without_adding_parameters():
    baseline, warmed = paired_models()
    assert baseline.state_dict().keys() == warmed.state_dict().keys()
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in warmed.parameters())
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, warmed.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(2, 4, 3, 5)
    altered = x.clone()
    altered[:, :-1] += 3
    torch.testing.assert_close(baseline(x), baseline(altered), rtol=0, atol=0)
    assert not torch.allclose(warmed(x), warmed(altered))


def test_warmed_history_has_finite_early_gradients_and_variable_batches():
    _, model = paired_models()
    for batch in (2, 1):
        model.zero_grad(set_to_none=True)
        x = torch.randn(batch, 4, 3, 5, requires_grad=True)
        output = model(x)
        assert output.shape == (batch, 6, 3, 2)
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        assert x.grad[:, :-1].abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_one_frame_warmup_is_exactly_original_decoder():
    baseline, warmed = paired_models(history=1)
    x = torch.randn(1, 1, 3, 5)
    torch.testing.assert_close(baseline(x), warmed(x), rtol=0, atol=0)
