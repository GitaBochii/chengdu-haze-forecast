"""Small direct-duration LSTM encoder; no integration with deployed models."""

import math

import torch
from torch import nn


class DirectLSTMAFT(nn.Module):
    def __init__(self, history_dim, snapshot_dim):
        super().__init__()
        self.history = nn.LSTM(history_dim, 16, batch_first=True)
        self.snapshot = nn.Sequential(nn.Linear(snapshot_dim, 32), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.constant_(self.output[-1].bias, math.log(24))

    def forward(self, history, snapshot):
        _, (hidden, _) = self.history(history)
        combined = torch.cat([hidden[-1], self.snapshot(snapshot)], dim=1)
        return self.output(combined).squeeze(1)
