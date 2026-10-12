"""Matched-capacity state-only and state-age residual recovery heads."""

from __future__ import annotations

import torch
from torch import nn

from haze_forecast.regional_recovery_network import RecoveryPM25GNN


class ConditionalRecoveryPM25GNN(RecoveryPM25GNN):
    """Add a state-conditioned residual without changing the base decoder.

    input_dim counts only the original features. The final added input channel
    is the dataset-scaled causal age, shared by all nodes. Both ablation arms
    retain identical parameter shapes and initialization; only age use differs.
    """

    use_age = True

    def __init__(self, input_dim, node_coords, wind_mean, wind_scale,
                 history=24, horizon=72, target_indices=(0, 2), step_hours=1.0):
        super().__init__(input_dim, node_coords, wind_mean, wind_scale,
                         history, horizon, target_indices, step_hours)
        state_dim = self.risk_head.in_features
        self.conditional_head = nn.Sequential(
            nn.Linear(state_dim + 1, 16), nn.Tanh(), nn.Linear(16, horizon - 5))
        nn.init.zeros_(self.conditional_head[2].weight)
        nn.init.zeros_(self.conditional_head[2].bias)

    def forward(self, x, active):
        if (x.ndim != 4 or x.shape[1:] != (
                self.history, self.node_count, self.input_dim + 1) or x.shape[0] < 1):
            raise ValueError('expected base input features plus one final age channel')
        age_channel = x[..., -1:]
        if not torch.isfinite(age_channel).all():
            raise ValueError('age values must be finite')
        if not torch.equal(age_channel, age_channel[:, :, :1].expand_as(age_channel)):
            raise ValueError('age channel must agree across nodes')
        age = age_channel[:, -1, 0]
        if not self.use_age:
            age = torch.zeros_like(age)
        curves, state = self.forward_with_state(x[..., :-1], active)
        residual = self.conditional_head(torch.cat([state, age], dim=-1))
        return curves, self.risk_head(state) + residual


class StateOnlyRecoveryPM25GNN(ConditionalRecoveryPM25GNN):
    """Capacity-matched control that always replaces age with zero."""

    use_age = False


class StateAgeRecoveryPM25GNN(ConditionalRecoveryPM25GNN):
    """State-conditioned residual that consumes the current causal age."""

    use_age = True
