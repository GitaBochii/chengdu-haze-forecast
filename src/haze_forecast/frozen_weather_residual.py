"""Small weather hazard residual for cached, frozen backbone states."""

import torch
from torch import nn


class FrozenWeatherResidual(nn.Module):
    def __init__(self, state_dim, use_weather=True, centered=False):
        super().__init__()
        if not isinstance(state_dim, int) or isinstance(state_dim, bool) or state_dim < 1:
            raise ValueError('state_dim must be a positive integer')
        if not isinstance(use_weather, bool) or not isinstance(centered, bool):
            raise TypeError('use_weather and centered must be boolean')
        self.state_dim = state_dim
        self.use_weather = use_weather
        self.centered = centered
        self.head = nn.Sequential(nn.Linear(state_dim + 192, 16), nn.Tanh(), nn.Linear(16, 67))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _raw(self, state, weather, mask):
        if not all(isinstance(value, torch.Tensor) for value in (state, weather, mask)):
            raise ValueError('state, weather and mask must be tensors')
        if state.ndim != 2 or state.shape[0] < 1 or state.shape[1] != self.state_dim:
            raise ValueError('state must be nonempty [batch, state_dim]')
        if weather.shape != (state.shape[0], 24, 4) or mask.shape != weather.shape:
            raise ValueError('weather and mask must be [batch, 24, 4]')
        if (not state.is_floating_point() or weather.dtype != state.dtype
                or mask.dtype != torch.bool or self.head[0].weight.dtype != state.dtype):
            raise ValueError('floating state/weather must match model dtype and mask must be boolean')
        if (weather.device != state.device or mask.device != state.device
                or self.head[0].weight.device != state.device):
            raise ValueError('state, weather, mask and model must share a device')
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(weather[mask]).all()):
            raise ValueError('state and observed weather must be finite')
        numeric = torch.where(mask, weather, 0.) if self.use_weather else torch.zeros_like(weather)
        features = torch.cat((state, numeric.flatten(1), mask.to(state.dtype).flatten(1)), dim=1)
        return self.head(features)

    def forward(self, state, weather, mask, reference=None):
        raw = self._raw(state, weather, mask)
        if self.centered:
            if not isinstance(reference, tuple) or len(reference) != 3:
                raise ValueError('centered residual requires a (state, weather, mask) reference tuple')
            reference_raw = self._raw(*reference)
            if not bool(reference[2].all()):
                raise ValueError('centering reference requires fully observed weather masks')
            # Keep the reference branch in the graph so centering constrains training gradients.
            raw = raw - reference_raw.mean(dim=0, keepdim=True)
        return torch.where(mask.flatten(1).any(dim=1, keepdim=True), raw, torch.zeros_like(raw))
