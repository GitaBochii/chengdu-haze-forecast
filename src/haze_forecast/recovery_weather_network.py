"""Matched-capacity weather and masked-value control recovery residuals."""

import torch
from torch import nn

from haze_forecast.regional_recovery_network import RecoveryPM25GNN


class WeatherRecoveryPM25GNN(RecoveryPM25GNN):
    """Condition hazards on 24 ordered three-hour, four-feature forecasts.

    Weather is city-grid exogenous information, not independent station data.
    The inherited concentration decoder is unchanged. All-missing origins
    bypass the residual, including after its parameters have been trained.
    """

    use_weather = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.horizon != 72:
            raise ValueError('weather branch requires the fixed 72-hour protocol')
        self.weather_head = nn.Sequential(
            nn.Linear(self.risk_head.in_features + 24 * 4 * 2, 16),
            nn.Tanh(), nn.Linear(16, self.horizon - 5))
        nn.init.zeros_(self.weather_head[2].weight)
        nn.init.zeros_(self.weather_head[2].bias)

    def forward(self, x, active, weather, weather_mask):
        if (weather.shape != (x.shape[0], 24, 4) or weather_mask.shape != weather.shape
                or weather_mask.dtype != torch.bool or weather.device != x.device
                or weather_mask.device != x.device or weather.dtype != x.dtype):
            raise ValueError('weather requires matching [batch, 24, 4] values and boolean mask')
        if not torch.isfinite(weather[weather_mask]).all():
            raise ValueError('observed weather values must be finite')
        safe = torch.where(weather_mask, weather, 0.)
        if not self.use_weather:
            safe = torch.zeros_like(safe)
        curves, state = self.forward_with_state(x, active)
        features = torch.cat([state, safe.flatten(1), weather_mask.to(x.dtype).flatten(1)], dim=1)
        residual = self.weather_head(features)
        available = weather_mask.flatten(1).any(dim=1, keepdim=True)
        residual = torch.where(available, residual, 0.)
        return curves, self.risk_head(state) + residual


class WeatherControlRecoveryPM25GNN(WeatherRecoveryPM25GNN):
    """Same parameters, state and missingness, but no weather numeric values."""

    use_weather = False
