"""Research-only interval-censored discrete hazard EM using LightGBM."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from haze_forecast.interval_hazard_likelihood import expectation, median_hours


def soft_risk_targets(events, risk):
    events, risk = np.asarray(events, dtype=float), np.asarray(risk, dtype=float)
    if (events.shape != risk.shape or not np.isfinite(events).all()
            or not np.isfinite(risk).all() or (events < 0).any()
            or (risk < 0).any() or (events > risk + 1e-12).any()):
        raise ValueError('Expected finite event counts between zero and at-risk counts')
    target = np.divide(events, risk, out=np.zeros_like(events), where=risk > 0)
    return np.clip(target, 0, 1), risk.copy()


class IntervalHazardEM:
    def __init__(self, rounds=300, max_em_steps=5):
        if not isinstance(rounds, int) or not isinstance(max_em_steps, int) or min(rounds, max_em_steps) < 1:
            raise ValueError('Positive integer iteration limits required')
        self.rounds = rounds
        self.max_em_steps = max_em_steps

    def fit_bounds(self, features, lower, upper):
        if not isinstance(features, pd.DataFrame) or features.empty or not features.columns.is_unique:
            raise ValueError('Nonempty unique feature columns required')
        if 'horizon_hours' in features:
            raise ValueError('Reserved horizon feature present')
        values = features.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('Infinite feature values are not supported')
        for bound in (lower, upper):
            if isinstance(bound, pd.Series) and not bound.index.equals(features.index):
                raise ValueError('Label index must match features')
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (lo.shape != (len(features),) or hi.shape != lo.shape or not np.isfinite(lo).all()
                or np.isnan(hi).any() or np.isneginf(hi).any() or not np.isfinite(hi).any()):
            raise ValueError('Aligned valid labels with at least one finite event required')
        observed_end = np.where(np.isfinite(hi), hi, lo - 1)
        self.horizon = int(max(1, observed_end.max()))
        self.initial_hazard = float(np.clip(1 / np.median(lo), 1e-6, 1 - 1e-6))
        hazards = np.full((len(features), self.horizon), self.initial_hazard)
        logp, events, risk = expectation(hazards, lo, hi)
        counts = observed_end.astype(int)
        owner = np.repeat(np.arange(len(features)), counts)
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        hours = np.arange(len(owner)) - offsets
        expanded = np.column_stack([values[owner], hours + 1.])
        self.feature_names = list(features.columns)
        self.model = None
        current = float(-logp.mean())
        self.trace = [{'step': 0, 'nll': current, 'accepted': True}]
        params = {'objective': 'cross_entropy', 'verbosity': -1, 'num_threads': 1,
                  'learning_rate': .04, 'num_leaves': 15, 'min_data_in_leaf': 40,
                  'feature_fraction': .8, 'lambda_l2': 2., 'seed': 42,
                  'force_col_wise': True}
        for step in range(1, self.max_em_steps + 1):
            target, weight = soft_risk_targets(events[owner, hours], risk[owner, hours])
            keep = weight > 0
            dataset = lgb.Dataset(expanded[keep], label=target[keep], weight=weight[keep])
            candidate = lgb.train(params, dataset, num_boost_round=self.rounds)
            proposal = hazards.copy()
            proposal[owner, hours] = np.clip(candidate.predict(expanded, num_threads=1), 1e-9, 1 - 1e-9)
            logp, next_events, next_risk = expectation(proposal, lo, hi)
            score = float(-logp.mean())
            accepted = bool(np.isfinite(score) and score <= current)
            self.trace.append({'step': step, 'nll': score, 'accepted': accepted})
            if not accepted:
                break
            self.model, hazards = candidate, proposal
            events, risk = next_events, next_risk
            improvement = current - score
            current = score
            if improvement < 1e-6:
                break
        return self

    def predict_hazards(self, features):
        if list(features.columns) != self.feature_names:
            raise ValueError('Feature names and order must match training')
        values = features.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('Infinite feature values are not supported')
        result = np.full((len(features), self.horizon), self.initial_hazard)
        if self.model is None:
            return result
        for start in range(0, len(features), 128):
            batch = values[start:start + 128]
            expanded = np.column_stack([np.repeat(batch, self.horizon, axis=0),
                                        np.tile(np.arange(1, self.horizon + 1), len(batch))])
            result[start:start + len(batch)] = np.clip(
                self.model.predict(expanded, num_threads=1).reshape(len(batch), self.horizon), 1e-9, 1 - 1e-9)
        return result

    def predict(self, features):
        return median_hours(self.predict_hazards(features))
