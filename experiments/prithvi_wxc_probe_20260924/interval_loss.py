"""Likelihood of closed interval bounds for capped discrete recovery time.

The source interval_recovery_bounds returns the earliest possible clean-window
start L and earliest confirmed start U. Both ends are inclusive. U=+inf means
T >= L, not T > L. Project the complete bounds onto C=min(T,68): bins 1..67
are events and bin 68 is all remaining survival mass, not exact recovery.
"""
import torch
from torch.nn import functional as F


def interval_survival_nll(logits, lower, upper, reduction='mean'):
    """Return -log P(min(L,68) <= C <= min(U,68)) per row or reduced.

    logits: finite floating [batch,67] conditional hazard logits.
    lower/upper: [batch] hourly integer-valued bounds. Lower must be finite;
    upper may be +inf. Bounds are validated BEFORE capping. No midpoint label
    is substituted. Fully uncertain capped intervals have exactly zero loss.
    Float16/BFloat16 logits accumulate probabilities in float32.
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.shape[1] != 67 or logits.shape[0] < 1
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError('expected finite floating [batch,67] logits')
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError('reduction must be none, mean or sum')
    raw_low = torch.as_tensor(lower, device=logits.device)
    raw_high = torch.as_tensor(upper, device=logits.device)
    if (raw_low.dtype == torch.bool or raw_high.dtype == torch.bool
            or raw_low.is_complex() or raw_high.is_complex()):
        raise ValueError('boolean or complex bounds are not valid hours')
    low = torch.as_tensor(lower, device=logits.device, dtype=torch.float64)
    high = torch.as_tensor(upper, device=logits.device, dtype=torch.float64)
    if low.shape != logits.shape[:1] or high.shape != low.shape:
        raise ValueError('bounds must be one-dimensional and match batch size')
    finite_high = torch.isfinite(high)
    if (not torch.isfinite(low).all() or not (low >= 1).all()
            or not (low == low.floor()).all() or torch.isnan(high).any()
            or not (high >= low).all()
            or not (high[finite_high] == high[finite_high].floor()).all()):
        raise ValueError('ordered positive integer bounds required; only upper may be +inf')
    values = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
    log_survival = F.logsigmoid(-values).cumsum(dim=1)
    log_prior = torch.cat((torch.zeros_like(log_survival[:, :1]), log_survival[:, :-1]), dim=1)
    log_event = log_prior + F.logsigmoid(values)
    log_mass = torch.cat((log_event, log_survival[:, -1:]), dim=1)
    low, high = low.clamp_max(68), high.clamp_max(68)
    bins = torch.arange(1, 69, device=logits.device)
    allowed = (bins[None] >= low[:, None]) & (bins[None] <= high[:, None])
    loss = -torch.logsumexp(log_mass.masked_fill(~allowed, -torch.inf), dim=1)
    # Avoid roundoff residual and spurious gradients for completely unknown C.
    loss = torch.where((low == 1) & (high == 68), torch.zeros_like(loss), loss)
    if reduction == 'none':
        return loss
    return loss.mean() if reduction == 'mean' else loss.sum()
