"""Research probability decoder for six-hour clean-run first passage."""

import numpy as np


def recovery_start_distribution(clean_probability):
    """Map conditional clean probabilities [batch, future_hour, run=0..5].

    Origins start polluted (run zero). Each probability is conditional on the
    current run state and no prior confirmed recovery. Columns 0..H-6 of the
    result represent clean-window starts at future hours 1..H-5; the final
    column is no confirmed recovery within H hours, not permanent pollution.
    Unknown probabilities are rejected; this decoder does not impute data.
    """
    q = np.asarray(clean_probability, dtype=np.float64)
    if (q.ndim != 3 or q.shape[0] < 1 or q.shape[1] < 6 or q.shape[2] != 6
            or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()):
        raise ValueError('Require finite probabilities in [0,1] with shape [batch,H>=6,6]')
    batch, horizon, _ = q.shape
    state = np.zeros((batch, 6))
    state[:, 0] = 1
    result = np.zeros((batch, horizon - 4))
    for hour in range(horizon):
        clean_mass = state * q[:, hour]
        next_state = np.zeros_like(state)
        next_state[:, 0] = (state * (1 - q[:, hour])).sum(axis=1)
        next_state[:, 1:] = clean_mass[:, :-1]
        if hour >= 5:
            # Confirmation at hour+1 means window start five hours earlier.
            result[:, hour - 5] = clean_mass[:, 5]
        state = next_state
    result[:, -1] = state.sum(axis=1)
    return result
