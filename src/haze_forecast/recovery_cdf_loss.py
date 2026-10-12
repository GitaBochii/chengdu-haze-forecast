"""Row-equal CDF squared loss over known recovery-status observation times."""

import torch

from haze_forecast.recovery_survival import hazard_distribution


def observed_cdf_loss(logits, target, risk_mask, eligible):
    """Score observed CDF cells, not an IPCW or unbiased survival Brier score.

    An exact event identifies status at all 67 times. Without an event, only
    the observed risk prefix is known; missing later times are not tail labels.
    Each eligible row contributes its mean squared error over known cells, then
    rows are averaged equally. Return None when no eligible row has known cells.
    """
    values = (logits, target, risk_mask, eligible)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise ValueError('CDF loss inputs must be tensors')
    if (logits.ndim != 2 or logits.shape[1] != 67 or target.shape != logits.shape
            or risk_mask.shape != logits.shape or eligible.shape != logits.shape[:1]):
        raise ValueError('CDF loss requires aligned [batch, 67] arrays and [batch] eligibility')
    if not all(value.device == logits.device for value in values):
        raise ValueError('CDF loss inputs must share a device')
    if (not logits.is_floating_point() or target.is_complex()
            or risk_mask.dtype != torch.bool or eligible.dtype != torch.bool):
        raise ValueError('CDF loss requires floating logits and boolean risk/eligibility')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('CDF logits must all be finite')
    if not bool(((target == 0) | (target == 1)).all()) or bool((target.sum(dim=1) > 1).any()):
        raise ValueError('CDF target must contain at most one binary event per row')
    event_cells = target == 1
    if bool((event_cells & ~risk_mask).any()):
        raise ValueError('CDF events must occur inside the observed risk prefix')
    if bool((risk_mask[:, 1:] & ~risk_mask[:, :-1]).any()):
        raise ValueError('CDF risk masks must be contiguous prefixes')
    exact = event_cells.any(dim=1)
    event_position = event_cells.to(torch.int64).argmax(dim=1) + 1
    if bool((exact & (event_position != risk_mask.sum(dim=1))).any()):
        raise ValueError('an exact event must be the last observed risk cell')
    known = risk_mask | exact[:, None]
    rows = eligible & known.any(dim=1)
    if not bool(rows.any()):
        return None
    cdf = hazard_distribution(logits)[:, :67].cumsum(dim=1)
    label = target.to(logits.dtype).cumsum(dim=1)
    squared = (cdf - label).square()
    known_rows = known[rows]
    return (squared[rows].masked_fill(~known_rows, 0).sum(dim=1)
            / known_rows.sum(dim=1)).mean()
