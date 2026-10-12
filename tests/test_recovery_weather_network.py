import numpy as np
import pytest
import torch

from haze_forecast.recovery_weather_network import (
    WeatherControlRecoveryPM25GNN,
    WeatherRecoveryPM25GNN,
)
from haze_forecast.regional_recovery_network import RecoveryPM25GNN


def make(cls):
    torch.manual_seed(42)
    return cls(5, np.array([[104., 30.], [104.1, 30.]]), np.zeros(2), np.ones(2), history=4)


def inputs():
    return (torch.randn(2, 4, 2, 5), torch.ones(2, 2, 2, dtype=torch.bool),
            torch.randn(2, 24, 4), torch.ones(2, 24, 4, dtype=torch.bool))


def test_matched_capacity_and_zero_initialized_base_equivalence():
    base, weather, control = [make(cls) for cls in
                              (RecoveryPM25GNN, WeatherRecoveryPM25GNN, WeatherControlRecoveryPM25GNN)]
    for key, value in weather.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, weather.state_dict()[key], rtol=0, atol=0)
    x, active, forecast, mask = inputs()
    expected = base(x, active)
    for model in (weather, control):
        for a, b in zip(model(x, active, forecast, mask), expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_weather_changes_hazard_not_curves_control_ignores_values():
    x, active, forecast, mask = inputs()
    for cls in (WeatherRecoveryPM25GNN, WeatherControlRecoveryPM25GNN):
        model = make(cls)
        with torch.no_grad():
            model.weather_head[2].weight.fill_(.1)
        a = model(x, active, forecast, mask)
        b = model(x, active, forecast + 5, mask)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        assert torch.equal(a[1], b[1]) == (cls is WeatherControlRecoveryPM25GNN)


def test_all_missing_falls_back_even_after_branch_update():
    model = make(WeatherRecoveryPM25GNN)
    x, active, forecast, mask = inputs()
    with torch.no_grad():
        model.weather_head[2].weight.fill_(.2)
        model.weather_head[2].bias.fill_(3.)
    mask.zero_()
    forecast.fill_(float('nan'))
    expected = RecoveryPM25GNN.forward(model, x, active)
    actual = model(x, active, forecast, mask)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_two_backward_steps_reach_weather_weights():
    model = make(WeatherRecoveryPM25GNN)
    batch = inputs()
    model(*batch)[1].sum().backward()
    with torch.no_grad():
        model.weather_head[2].weight.add_(model.weather_head[2].weight.grad, alpha=-.01)
    model.zero_grad(set_to_none=True)
    model(*batch)[1].sum().backward()
    gradient = model.weather_head[0].weight.grad[:, 128:224]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_reject_observed_nan_and_wrong_shape():
    model = make(WeatherRecoveryPM25GNN)
    x, active, forecast, mask = inputs()
    with pytest.raises(ValueError):
        model(x, active, forecast[:, :-1], mask)
    forecast[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        model(x, active, forecast, mask)
