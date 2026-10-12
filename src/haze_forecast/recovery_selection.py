"""Predeclared recovery checkpoint criteria and forward-development gate."""

from __future__ import annotations

import numpy as np


def selection_metrics(records):
    selected = records.loc[records.label_verified_in_scoring_window & records.actual_hours.notna()].copy()
    if selected.empty:
        raise ValueError('no verifiable recovery observations')
    predicted = selected.predicted_hours.to_numpy(dtype=float)
    if np.isinf(predicted).any() or (predicted[np.isfinite(predicted)] < 1).any():
        raise ValueError('predictions must be positive hours or unknown tail')
    selected['error'] = np.abs(np.minimum(selected.actual_hours, 68)
                               - np.minimum(np.nan_to_num(predicted, nan=68), 68))
    short = selected.loc[selected.actual_hours.lt(68)]
    tail = selected.loc[selected.actual_hours.ge(68)]
    short_mean = float(short.groupby('event_id').error.mean().mean()) if len(short) else None
    tail_mean = float(tail.groupby('event_id').error.mean().mean()) if len(tail) else None
    return {'rows': len(selected), 'events': int(selected.event_id.nunique()),
            'row_mae': float(selected.error.mean()),
            'event_equal_mae': float(selected.groupby('event_id').error.mean().mean()),
            'short_event_equal_mae': short_mean, 'tail_event_equal_mae': tail_mean,
            'balanced_mae': (short_mean + tail_mean) / 2 if short_mean is not None and tail_mean is not None else None,
            'short_rows': len(short), 'tail_rows': len(tail),
            'short_events': [int(value) for value in sorted(short.event_id.unique())],
            'tail_events': [int(value) for value in sorted(tail.event_id.unique())],
            'group_overlap_events': len(set(short.event_id) & set(tail.event_id))}


def selection_keys(metrics):
    if metrics['balanced_mae'] is None:
        raise ValueError('balanced selection requires both short and tail support')
    return {'legacy': (metrics['event_equal_mae'], metrics['row_mae']),
            'balanced': (metrics['balanced_mae'], metrics['event_equal_mae'], metrics['row_mae'])}


def development_gate(folds):
    if not folds:
        raise ValueError('development folds are required')
    tail_ids = set()
    for fold in folds:
        if fold['legacy']['tail_events'] != fold['balanced']['tail_events']:
            raise ValueError('policies must use the same tail observations')
        tail_ids.update(fold['legacy']['tail_events'])
    supported = (len(folds) >= 2 and len(tail_ids) >= 3
                 and all(fold['legacy']['tail_events'] for fold in folds))
    means = {}
    for rule in ('legacy', 'balanced'):
        means[rule] = {}
        for metric in ('row_mae', 'event_equal_mae', 'tail_event_equal_mae'):
            values = [fold[rule][metric] for fold in folds]
            means[rule][metric] = float(np.mean(values)) if all(value is not None for value in values) else None
    finite = all(value is not None and np.isfinite(value) for row in means.values() for value in row.values())
    checks = {'sufficient_tail_events': bool(supported), 'finite_metrics': bool(finite)}
    checks['row_not_worse'] = bool(finite and means['balanced']['row_mae'] <= means['legacy']['row_mae'])
    checks['event_equal_not_worse'] = bool(finite and means['balanced']['event_equal_mae'] <= means['legacy']['event_equal_mae'])
    checks['tail_better'] = bool(finite and means['balanced']['tail_event_equal_mae'] < means['legacy']['tail_event_equal_mae'])
    return {'accepted': all(checks.values()), 'checks': checks,
            'independent_tail_events': len(tail_ids), 'annual_equal_means': means}
