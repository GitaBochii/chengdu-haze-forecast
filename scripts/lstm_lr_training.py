"""Learning-rate-only fit loop for the frozen direct-history LSTM."""

import math
import random

import numpy as np
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.torch_interval_aft import interval_aft_nll
from scripts.benchmark_lstm_direct_crossyear import choose_epoch, make_model, predict


def _validate_inputs(history, snapshot, lower, upper, epochs, learning_rate):
    try:
        rate = float(learning_rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("learning_rate must be finite and positive") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")

    history, snapshot = np.asarray(history), np.asarray(snapshot)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (
        history.ndim != 3
        or snapshot.ndim != 2
        or len(history) == 0
        or len(snapshot) != len(history)
        or history.shape[1] == 0
        or history.shape[2] == 0
        or snapshot.shape[1] == 0
    ):
        raise ValueError("Training inputs must have aligned nonempty history and snapshot shapes")
    if lower.shape != (len(history),) or upper.shape != lower.shape:
        raise ValueError("Training labels must be aligned one-dimensional arrays")
    if (
        not np.isfinite(lower).all()
        or (lower <= 0).any()
        or not ((np.isfinite(upper) | np.isposinf(upper)).all())
        or (upper < lower).any()
    ):
        raise ValueError("Training labels must have valid positive interval bounds")
    return history, snapshot, lower, upper, int(epochs), rate


def fit_lstm(seed, history, snapshot, lower, upper, epochs, learning_rate, validation=None):
    history, snapshot, lower, upper, epochs, learning_rate = _validate_inputs(
        history, snapshot, lower, upper, epochs, learning_rate
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model("lstm_history", history.shape[-1], snapshot.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    tensors = [torch.as_tensor(a, device="cuda") for a in (history, snapshot, lower, upper)]
    generator = torch.Generator().manual_seed(seed)
    trace, best, best_state, best_epoch = [], math.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(history), generator=generator).cuda()
        total = 0.0
        for ids in permutation.split(128):
            optimizer.zero_grad(set_to_none=True)
            mu = model(tensors[0][ids], tensors[1][ids])
            loss = interval_aft_nll(mu, tensors[2][ids], tensors[3][ids], scale=1.2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        entry = {"epoch": epoch, "training_nll": total / len(history)}
        if validation is not None:
            hv, xv, frame = validation
            score = score_predictions(frame, predict(model, hv, xv))["exact_block_mae"]
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite development score")
            entry["selection_block_mae"] = score
            if score < best:
                best, best_epoch = score, epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
        trace.append(entry)
        if epoch == 1 or epoch % 10 == 0:
            print({"arm": "lstm_history", "seed": seed, **entry}, flush=True)
        if validation is not None and epoch - best_epoch >= 8:
            break
    if validation is not None:
        assert choose_epoch(trace) == best_epoch and best_state is not None
        model.load_state_dict(best_state)
    return model, trace, best_epoch
