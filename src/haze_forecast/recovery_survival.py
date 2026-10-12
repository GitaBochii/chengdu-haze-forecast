"""Observed-prefix survival labels for the first six-hour AQI recovery window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def validate_hourly_grid(timestamps) -> None:
    times = pd.DatetimeIndex(timestamps)
    if (len(times) < 2 or times.tz is None or times.hasnans or times.has_duplicates
            or not np.all(np.diff(times.asi8) == pd.Timedelta(hours=1).value)):
        raise ValueError('expected a unique, timezone-aware, continuous hourly grid')


def recovery_hazard_labels(future_aqi, stable_hours=6, threshold=100):
    future = np.asarray(future_aqi, dtype=float)
    if future.ndim != 2 or stable_hours < 1 or future.shape[1] < stable_hours:
        raise ValueError('expected [batch,horizon] supporting the stable window')
    bins = future.shape[1] - stable_hours + 1
    target = np.zeros((len(future), bins), dtype=np.float32)
    mask = np.zeros_like(target, dtype=bool)
    for row, values in enumerate(future):
        missing = np.flatnonzero(~np.isfinite(values))
        prefix = int(missing[0]) if len(missing) else len(values)
        for start in range(max(0, prefix - stable_hours + 1)):
            mask[row, start] = True
            if (values[start:start + stable_hours] <= threshold).all():
                target[row, start] = 1
                break
    return target, mask


def causal_recovery_eligibility(aqi, particle_primary):
    values = np.asarray(aqi, dtype=float)
    particle = np.asarray(particle_primary, dtype=bool)
    if values.ndim != 1 or particle.shape != values.shape:
        raise ValueError('expected aligned one-dimensional observations')
    eligible = np.zeros(len(values), dtype=bool)
    active, polluted_run, clean_run = False, 0, 0
    for index, value in enumerate(values):
        polluted = np.isfinite(value) and value > 100 and particle[index]
        clean = np.isfinite(value) and value <= 100
        polluted_run = polluted_run + 1 if polluted else 0
        clean_run = clean_run + 1 if clean else 0
        if polluted_run >= 3:
            active = True
        if clean_run >= 6:
            active = False
        eligible[index] = active or polluted
    return eligible


def masked_survival_loss(logits, target, mask, eligible):
    if logits.shape != target.shape or mask.shape != target.shape or eligible.shape != logits.shape[:1]:
        raise ValueError('survival arrays must align by origin and hazard bin')
    rows = eligible.bool() & mask.bool().any(dim=1)
    if not rows.any():
        return None
    terms = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (terms * mask)[rows].sum(dim=1).mean()


def hazard_distribution(logits):
    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError('expected nonempty [batch,hazard_bin] logits')
    log_survival = F.logsigmoid(-logits).cumsum(dim=1)
    prior = torch.cat([torch.zeros_like(log_survival[:, :1]), log_survival[:, :-1]], dim=1)
    events = (prior + F.logsigmoid(logits)).exp()
    return torch.cat([events, log_survival[:, -1:].exp()], dim=1)


def hazard_median(logits):
    probability = hazard_distribution(logits)
    return (probability.cumsum(dim=1) < 0.5).sum(dim=1).clamp_max(logits.shape[1]) + 1
