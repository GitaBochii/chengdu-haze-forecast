"""Small direct-duration encoder; no integration with deployed models."""

import math

import numpy as np
import torch
from torch import nn


def fit_scale(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not len(x) or np.isinf(x).any():
        raise ValueError('Expected nonempty matrix with finite values or NaN')
    count = np.isfinite(x).sum(0)
    mean = np.nansum(x, axis=0) / np.maximum(count, 1)
    variance = np.nansum((x - mean)**2, axis=0) / np.maximum(count, 1)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.)
    return mean, scale


def transform(values, mean, scale, supported=None):
    x = np.asarray(values, dtype=float)
    missing = np.isnan(x)
    result = np.where(missing, 0., (x - mean) / scale)
    if supported is not None:
        result = np.where(supported, result, 0.)
    return np.concatenate([result, missing.astype(float)], axis=-1).astype(np.float32)


class DirectHistoryAFT(nn.Module):
    def __init__(self, history_dim, snapshot_dim):
        super().__init__()
        self.history = nn.GRU(history_dim, 16, batch_first=True)
        self.snapshot = nn.Sequential(nn.Linear(snapshot_dim, 32), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.constant_(self.output[-1].bias, math.log(24))

    def forward(self, history, snapshot):
        _, state = self.history(history)
        combined = torch.cat([state[-1], self.snapshot(snapshot)], dim=1)
        return self.output(combined).squeeze(1)
