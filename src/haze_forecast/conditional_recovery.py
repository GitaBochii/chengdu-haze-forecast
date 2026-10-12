"""Conditional recovery-time correction with the frozen base tail mass."""

import torch
from torch.nn import functional as F


def conditional_log_probability(base_logits, correction):
    if (base_logits.ndim != 2 or base_logits.shape[1] != 67
            or correction.shape != base_logits.shape or not base_logits.is_floating_point()
            or correction.dtype != base_logits.dtype or correction.device != base_logits.device
            or not torch.isfinite(base_logits).all() or not torch.isfinite(correction).all()):
        raise ValueError('expected aligned finite floating [batch,67] tensors')
    survival = F.logsigmoid(-base_logits).cumsum(1)
    prior = torch.cat((torch.zeros_like(survival[:, :1]), survival[:, :-1]), 1)
    event = prior + F.logsigmoid(base_logits)
    # Sum the event masses in log space, avoiding cancellation when q is almost one.
    event_total = torch.logsumexp(event, 1, keepdim=True)
    adjusted = event_total + F.log_softmax(event - event_total + correction, dim=1)
    return torch.cat((adjusted, survival[:, -1:]), 1)


def observed_prefix_nll(log_probability, target, risk, eligible):
    n = len(log_probability)
    if (log_probability.shape != (n, 68) or target.shape != (n, 67)
            or risk.shape != target.shape or eligible.shape != (n,)
            or risk.dtype != torch.bool or eligible.dtype != torch.bool
            or not log_probability.is_floating_point()
            or any(x.device != log_probability.device for x in (target, risk, eligible))
            or not torch.isfinite(log_probability).all()
            or not ((target == 0) | (target == 1)).all() or (target.sum(1) > 1).any()
            or (target.bool() & ~risk).any() or (risk[:, 1:] & ~risk[:, :-1]).any()):
        raise ValueError('invalid aligned log probabilities and observed-prefix labels')
    events = target.sum(1) == 1
    if (events & (target.argmax(1) != risk.sum(1) - 1)).any():
        raise ValueError('event must end observed prefix')
    rows = eligible & risk.any(1)
    if not rows.any():
        return None
    event_loss = -log_probability[:, :67].gather(1, target.argmax(1)[:, None]).squeeze(1)
    remaining = torch.arange(68, device=risk.device)[None, :] >= risk.sum(1)[:, None]
    censor_loss = -torch.logsumexp(log_probability.masked_fill(~remaining, -torch.inf), dim=1)
    return torch.where(events, event_loss, censor_loss)[rows].mean()
