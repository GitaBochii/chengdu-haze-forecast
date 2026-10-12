"""Zero-initialized correction using only the frozen historical state."""

import torch
from torch import nn


class HistoryStateResidual(nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.head = (nn.Sequential(nn.Linear(128, 16), nn.Tanh(), nn.Linear(16, 67))
                     if nonlinear else nn.Linear(128, 67))
        last = self.head[-1] if nonlinear else self.head
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, state):
        parameter = next(self.parameters())
        if (state.ndim != 2 or state.shape[1] != 128 or not state.is_floating_point()
                or state.dtype != parameter.dtype or state.device != parameter.device
                or not torch.isfinite(state).all()):
            raise ValueError('expected finite floating [batch,128] state on model dtype/device')
        return self.head(state)
