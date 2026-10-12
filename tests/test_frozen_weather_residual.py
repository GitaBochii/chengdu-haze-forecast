import pytest
import torch

from haze_forecast.frozen_weather_residual import FrozenWeatherResidual


def inputs(batch=5, state_dim=8):
    return (torch.randn(batch, state_dim), torch.randn(batch, 24, 4),
            torch.ones(batch, 24, 4, dtype=torch.bool))


def activate(model):
    with torch.no_grad():
        model.head[-1].weight.fill_(.1)
        model.head[-1].bias.fill_(.2)


@pytest.mark.parametrize('centered', [False, True])
def test_zero_initial_residual(centered):
    model = FrozenWeatherResidual(8, centered=centered)
    data = inputs()
    result = model(*data, reference=data if centered else None)
    assert result.shape == (5, 67)
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)


def test_masked_control_is_invariant_to_observed_numeric_values():
    model = FrozenWeatherResidual(8, use_weather=False)
    activate(model)
    state, weather, mask = inputs()
    torch.testing.assert_close(model(state, weather, mask), model(state, weather * 100 + 99, mask),
                               rtol=0, atol=0)


@pytest.mark.parametrize('centered', [False, True])
def test_missing_numeric_nan_is_safe_and_no_weather_bypasses(centered):
    model = FrozenWeatherResidual(8, centered=centered)
    activate(model)
    reference = inputs() if centered else None
    state, weather, mask = inputs()
    mask[0] = False
    mask[1, :3] = False
    clean = weather.masked_fill(~mask, 0)
    weather = weather.masked_fill(~mask, float('nan'))
    expected = model(state, clean, mask, reference=reference)
    actual = model(state, weather, mask, reference=reference)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], torch.zeros(67), rtol=0, atol=0)


def test_centered_reference_mean_and_reference_gradient():
    model = FrozenWeatherResidual(8, centered=True)
    activate(model)
    reference = inputs(batch=7)
    reference = tuple(value.requires_grad_() if value.is_floating_point() else value for value in reference)
    centered = model(*reference, reference=reference)
    torch.testing.assert_close(centered.mean(0), torch.zeros(67), rtol=0, atol=2e-7)
    query = inputs(batch=3)
    model(*query, reference=reference).square().sum().backward()
    assert reference[0].grad is not None and reference[0].grad.abs().sum() > 0
    assert reference[1].grad is not None and reference[1].grad.abs().sum() > 0
    assert model.head[-1].weight.grad is not None and model.head[-1].weight.grad.abs().sum() > 0


def test_zero_initial_head_can_start_learning():
    model = FrozenWeatherResidual(8, centered=True)
    result = model(*inputs(batch=3), reference=inputs(batch=7))
    (result - 1).square().sum().backward()
    assert model.head[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('bad', ['missing', 'empty', 'tuple', 'missing_mask', 'reference_nan'])
def test_invalid_centering_reference_rejected(bad):
    model = FrozenWeatherResidual(8, centered=True)
    reference = inputs()
    if bad == 'missing':
        reference = None
    elif bad == 'empty':
        reference = inputs(batch=0)
    elif bad == 'tuple':
        reference = reference[:2]
    elif bad == 'missing_mask':
        reference[2][0, 0, 0] = False
    else:
        reference[1][0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        model(*inputs(), reference=reference)


@pytest.mark.parametrize('bad', ['state_shape', 'weather_shape', 'mask_shape', 'mask_dtype',
                                'weather_dtype', 'state_dtype', 'state_nan', 'weather_nan',
                                'parameter_dtype'])
def test_invalid_query_rejected(bad):
    model = FrozenWeatherResidual(8)
    state, weather, mask = inputs()
    if bad == 'state_shape':
        state = state[:, :-1]
    elif bad == 'weather_shape':
        weather = weather[:, :-1]
    elif bad == 'mask_shape':
        mask = mask[:, :-1]
    elif bad == 'mask_dtype':
        mask = mask.float()
    elif bad == 'weather_dtype':
        weather = weather.double()
    elif bad == 'state_dtype':
        state = state.long()
    elif bad == 'state_nan':
        state[0, 0] = float('nan')
    elif bad == 'weather_nan':
        weather[0, 0, 0] = float('nan')
    else:
        model.double()
    with pytest.raises(ValueError):
        model(state, weather, mask)


def test_plain_head_ignores_optional_reference():
    model = FrozenWeatherResidual(8)
    activate(model)
    query = inputs()
    torch.testing.assert_close(model(*query), model(*query, reference='unused'), rtol=0, atol=0)


@pytest.mark.parametrize('dimension', [0, -1, True, 2.5])
def test_invalid_dimension(dimension):
    with pytest.raises(ValueError):
        FrozenWeatherResidual(dimension)


@pytest.mark.parametrize('target', ['weather', 'mask', 'model'])
def test_device_mismatch_rejected_before_computation(target):
    model = FrozenWeatherResidual(8)
    state, weather, mask = inputs()
    if target == 'weather':
        weather = weather.to('meta')
    elif target == 'mask':
        mask = mask.to('meta')
    else:
        model.to('meta')
    with pytest.raises(ValueError, match='device'):
        model(state, weather, mask)


@pytest.mark.parametrize('name', ['use_weather', 'centered'])
def test_flags_must_be_boolean(name):
    with pytest.raises(TypeError):
        FrozenWeatherResidual(8, **{name: 1})
