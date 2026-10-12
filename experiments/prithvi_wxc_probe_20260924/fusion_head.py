"""Offline fusion of cached spatial embeddings and pollutant history states.

This module does not load or certify Prithvi weights. Callers must validate
embedding provenance, observation times and splits before training/evaluation.
"""
import torch
from torch import nn
from haze_forecast.regional_recovery_network import RecoveryPM25GNN


class SpatialRecoveryFusion(nn.Module):
    """Zero-initialized hazard residual with exact all-missing fallback."""

    def __init__(self, baseline, embedding_dim, hidden_dim=16, freeze_baseline=True):
        super().__init__()
        if not isinstance(baseline, RecoveryPM25GNN) or baseline.horizon != 72:
            raise ValueError('requires RecoveryPM25GNN with 72-hour horizon')
        if (type(embedding_dim) is not int or embedding_dim < 1
                or type(hidden_dim) is not int or hidden_dim < 1):
            raise ValueError('embedding and hidden dimensions must be positive integers')
        self.baseline = baseline
        self.embedding_dim = embedding_dim
        self.freeze_baseline = freeze_baseline
        self.baseline.requires_grad_(not freeze_baseline)
        self.residual = nn.Sequential(
            nn.Linear(baseline.risk_head.in_features + embedding_dim, hidden_dim),
            nn.Tanh(), nn.Linear(hidden_dim, 67))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_baseline:
            self.baseline.eval()
        return self

    def forward(self, x, active, embedding, present):
        if (embedding.shape != (x.shape[0], self.embedding_dim)
                or present.shape != (x.shape[0],) or present.dtype != torch.bool
                or embedding.device != x.device or present.device != x.device
                or embedding.dtype != x.dtype):
            raise ValueError('expected matching [batch, embedding_dim] and boolean [batch] presence')
        if not torch.isfinite(embedding[present]).all():
            raise ValueError('observed embeddings must be finite')
        # Missing rows may contain NaN; sanitize before the MLP, not afterwards.
        safe = torch.where(present[:, None], embedding, torch.zeros_like(embedding))
        if self.freeze_baseline:
            with torch.no_grad():
                curves, state = self.baseline.forward_with_state(x, active)
                logits = self.baseline.risk_head(state)
        else:
            curves, state = self.baseline.forward_with_state(x, active)
            logits = self.baseline.risk_head(state)
        residual = self.residual(torch.cat((state, safe), dim=1))
        return curves, logits + torch.where(present[:, None], residual, torch.zeros_like(residual))
