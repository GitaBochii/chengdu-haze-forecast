"""Causal rule-memory age and a zero-initialized hazard-only age residual."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from haze_forecast.regional_recovery_network import RecoveryPM25GNN


def causal_recovery_age(aqi, particle_primary):
    """Return rule-memory hours, not a guarantee of continuous observations.

    Missing AQI interrupts confirmation/clean runs but does not clear an active
    process. Missing particle flags never count as particle pollution. Age is
    zero until three consecutive polluted hours confirm a process, then starts
    at three and increases until six consecutive clean hours clear the state.
    """
    values = np.asarray(aqi, dtype=float)
    flags = np.asarray(particle_primary, dtype=object)
    if values.ndim != 1 or flags.shape != values.shape:
        raise ValueError('expected aligned one-dimensional observations')
    particle = np.where(pd.isna(flags), False, flags).astype(bool)
    ages = np.zeros(len(values), dtype=np.int64)
    active, polluted_run, clean_run, age = False, 0, 0, 0
    for index, value in enumerate(values):
        polluted = np.isfinite(value) and value > 100 and particle[index]
        clean = np.isfinite(value) and value <= 100
        polluted_run = polluted_run + 1 if polluted else 0
        clean_run = clean_run + 1 if clean else 0
        if active:
            age += 1
        elif polluted_run >= 3:
            active, age = True, 3
        if clean_run >= 6:
            active, age = False, 0
        ages[index] = age
    return ages


class AgeRecoveryPM25GNN(RecoveryPM25GNN):
    """Keep base inputs/weights intact and add a scalar age residual to hazards.

    ``input_dim`` counts original features, while forward expects one additional
    final channel. The dataset supplies log1p(age)/log1p(168), without clipping;
    the fixed 168-hour denominator sets scale, not a maximum process length.
    """

    def __init__(self, input_dim, node_coords, wind_mean, wind_scale,
                 history=24, horizon=72, target_indices=(0, 2), step_hours=1.0):
        super().__init__(input_dim, node_coords, wind_mean, wind_scale,
                         history, horizon, target_indices, step_hours)
        self.age_projection = nn.Linear(1, horizon - 5, bias=False)
        nn.init.zeros_(self.age_projection.weight)

    def forward(self, x, active):
        if (x.ndim != 4 or x.shape[1:] != (
                self.history, self.node_count, self.input_dim + 1) or x.shape[0] < 1):
            raise ValueError('expected base input features plus one final age channel')
        age = x[..., -1:]
        if not torch.isfinite(age).all():
            raise ValueError('age values must be finite')
        if not torch.equal(age, age[:, :, :1].expand_as(age)):
            raise ValueError('age channel must agree across nodes')
        curves, logits = super().forward(x[..., :-1], active)
        return curves, logits + self.age_projection(age[:, -1, 0])
