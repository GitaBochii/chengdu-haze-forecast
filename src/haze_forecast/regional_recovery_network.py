"""History-aware concentration decoder with an origin-state recovery hazard head.

Activity masks affect pooling only, not the inherited graph's internal messages.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from haze_forecast.regional_air_networks import PM25GNNHistoryAdapter


class RecoveryPM25GNN(PM25GNNHistoryAdapter):
    """Predict concentration and hazards for six-hour-confirmable recovery starts."""

    def __init__(self, input_dim, node_coords, wind_mean, wind_scale,
                 history=24, horizon=72, target_indices=(0, 2), step_hours=1.0):
        if history < 1 or horizon < 6:
            raise ValueError("history must be positive and horizon must be at least six")
        if len(target_indices) != 2 or any(i < 0 or i >= input_dim for i in target_indices):
            raise ValueError("two valid concentration target indices are required")
        super().__init__(input_dim, node_coords, wind_mean, wind_scale,
                         history, horizon, target_indices, step_hours)
        self.input_dim = input_dim
        self.node_count = len(node_coords)
        self.risk_head = nn.Linear(sum(head.hid_dim for head in self.heads), horizon - 5)
        nn.init.constant_(self.risk_head.bias, math.log(1 / 23))

    def forward(self, x, active):
        curves, state = self.forward_with_state(x, active)
        return curves, self.risk_head(state)

    def forward_with_state(self, x, active):
        """Return unchanged concentration curves and the pooled origin state."""
        if (x.ndim != 4 or x.shape[1:] != (
                self.history, self.node_count, self.input_dim) or x.shape[0] < 1):
            raise ValueError("expected nonempty [batch, history, node, feature] inputs")
        batch, _, nodes, _ = x.shape
        if (active.shape != (batch, nodes, 2) or active.dtype != torch.bool
                or active.device != x.device):
            raise ValueError("active must be a boolean [batch, node, 2] tensor on input device")
        if not active.any(dim=1).all():
            raise ValueError("each batch and target must have at least one active node")
        predictions, pooled = [], []
        for target, (index, model) in enumerate(zip(self.target_indices, self.heads, strict=True)):
            model.graph_gnn.device = x.device
            hidden = x.new_zeros(batch * nodes, model.hid_dim)

            def step(values, state, model=model):
                spatial = model.graph_gnn(values.contiguous())
                return model.gru_cell(torch.cat([spatial, values], dim=-1), state)

            for hour in range(self.history - 1):
                observed = torch.cat([x[:, hour, :, index:index + 1], x[:, hour]], dim=-1)
                hidden = step(observed, hidden)
            current = x[:, -1, :, index:index + 1]
            decoded = []
            for lead in range(self.horizon):
                hidden = step(torch.cat([current, x[:, -1]], dim=-1), hidden)
                encoded = hidden.view(batch, nodes, model.hid_dim)
                if lead == 0:
                    # This state has consumed every observation but no forecast value.
                    mask = active[:, :, target].unsqueeze(-1)
                    pooled.append((encoded * mask).sum(dim=1) / mask.sum(dim=1))
                current = model.fc_out(encoded)
                decoded.append(current)
            predictions.append(torch.stack(decoded, dim=1))
        return torch.cat(predictions, dim=-1), torch.cat(pooled, dim=-1)
