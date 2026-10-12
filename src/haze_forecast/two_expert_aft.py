"""Fixed two-expert interval AFT with training-only proposal acceptance."""

import json
from numbers import Integral

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit

from haze_forecast.interval_mixture_math import mixture_median, mixture_observation_terms
from haze_forecast.lognormal_aft_audit import lognormal_interval_terms
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor


def accept_training_update(current, proposed):
    if not np.isfinite([current, proposed]).all():
        raise ValueError('Training losses must be finite')
    return bool(proposed < current)


def trainable_components(responsibility):
    with np.errstate(under='ignore'):
        native = np.asarray(responsibility).astype(np.float32)
    return bool(np.isfinite(native).all() and (native.sum(axis=0, dtype=float) > 0).all())


class TwoExpertAFT:
    """Keep all attempted states; never select an update on validation labels."""

    def __init__(self, max_updates=5):
        if isinstance(max_updates, bool) or not isinstance(max_updates, Integral) or max_updates < 1:
            raise ValueError('Require a positive integer update limit')
        self.max_updates = int(max_updates)

    @staticmethod
    def gate_parameters():
        return {
            'objective': 'binary:logistic', 'eta': .05, 'max_depth': 2,
            'min_child_weight': 8, 'lambda': 3., 'subsample': 1., 'colsample_bytree': 1.,
            'tree_method': 'hist', 'seed': 42, 'nthread': 1, 'verbosity': 0,
        }

    @staticmethod
    def _numeric(features):
        if (not isinstance(features, pd.DataFrame) or features.empty
                or not features.columns.is_unique or not features.index.is_unique):
            raise ValueError('Require nonempty features with unique columns and index')
        numeric = features.astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError('Features must be finite or missing')
        return numeric

    @staticmethod
    def state_parameters(state, matrix):
        locations = np.column_stack([
            expert.predict(matrix, output_margin=True).astype(float) + offset
            for expert, offset in zip(state['experts'], state['offsets'], strict=True)
        ])
        gate = (np.zeros(matrix.num_row()) if state['gate'] is None
                else state['gate'].predict(matrix, output_margin=True).astype(float))
        return locations, gate

    def _matrix(self, features):
        if not hasattr(self, 'selected_index'):
            raise ValueError('Model must be fitted before prediction')
        numeric = self._numeric(features)
        if list(numeric.columns) != self.feature_names:
            raise ValueError('Feature names and order must match training')
        return xgb.DMatrix(numeric, missing=np.nan)

    def predict_state(self, features, index):
        matrix = self._matrix(features)
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.states):
            raise ValueError('Unknown mixture state')
        locations, gate = self.state_parameters(self.states[index], matrix)
        return mixture_median(locations, gate)

    def predict(self, features):
        if not hasattr(self, 'selected_index'):
            raise ValueError('Model must be fitted before prediction')
        return self.predict_state(features, self.selected_index)

    @staticmethod
    def _trace_row(index, nll, responsibility, locations, gate, accepted):
        probability = expit(gate)
        return {
            'state_index': index, 'mean_nll': float(np.mean(nll)), 'accepted': bool(accepted),
            'responsibility_mass': responsibility.sum(axis=0).tolist(),
            'responsibility_float32_mass': responsibility.astype(np.float32).sum(axis=0, dtype=float).tolist(),
            'gate_probability_min': float(probability.min()),
            'gate_probability_mean': float(probability.mean()),
            'gate_probability_max': float(probability.max()),
            'mean_abs_log_location_separation': float(np.mean(abs(locations[:, 1] - locations[:, 0]))),
        }

    def fit_bounds(self, features, lower, upper, baseline):
        numeric = self._numeric(features)
        for bound in (lower, upper):
            if isinstance(bound, pd.Series) and not bound.index.equals(features.index):
                raise ValueError('Label index must match training features')
        if (not isinstance(baseline, ObservedAFTSurvivalRegressor) or baseline.booster is None
                or baseline.booster.feature_names != list(features.columns)):
            raise ValueError('Require a fitted baseline with identical feature names and order')
        matrix = xgb.DMatrix(numeric, missing=np.nan)
        margin = baseline.booster.predict(matrix, output_margin=True).astype(float)
        baseline_nll, _ = lognormal_interval_terms(lower, upper, margin)
        self.baseline_training_nll = float(np.mean(baseline_nll))
        self.feature_names = list(features.columns)
        self.states = [{
            'experts': [baseline.booster, baseline.booster], 'offsets': [-.6, .6],
            'gate': None, 'expert_training_configs': None, 'gate_training_config': None,
        }]
        self.selected_index = 0
        locations, gate = self.state_parameters(self.states[0], matrix)
        nll, responsibility = mixture_observation_terms(lower, upper, locations, gate)
        initial = mixture_median(locations, gate)
        np.testing.assert_allclose(initial, np.exp(margin), rtol=1e-12, atol=1e-12)
        self.trace = [self._trace_row(0, nll, responsibility, locations, gate, True)]
        self.stop_reason = 'maximum_updates'
        for index in range(1, self.max_updates + 1):
            if not trainable_components(responsibility):
                self.stop_reason = 'nonfinite_or_zero_native_responsibility_mass'
                break
            experts, configs = [], []
            for component in range(2):
                expert = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500)
                expert.fit_bounds(numeric, lower, upper, sample_weight=responsibility[:, component])
                experts.append(expert.booster)
                configs.append(json.loads(expert.booster.save_config()))
            gate_matrix = xgb.DMatrix(numeric, label=responsibility[:, 1], missing=np.nan)
            gate_model = xgb.train(self.gate_parameters(), gate_matrix, num_boost_round=100)
            proposal = {
                'experts': experts, 'offsets': [0., 0.], 'gate': gate_model,
                'expert_training_configs': configs,
                'gate_training_config': json.loads(gate_model.save_config()),
            }
            new_locations, new_gate = self.state_parameters(proposal, matrix)
            new_nll, new_responsibility = mixture_observation_terms(lower, upper, new_locations, new_gate)
            accepted = accept_training_update(np.mean(nll), np.mean(new_nll))
            self.states.append(proposal)
            self.trace.append(self._trace_row(index, new_nll, new_responsibility,
                                              new_locations, new_gate, accepted))
            if not accepted:
                self.stop_reason = 'training_nll_not_improved'
                break
            self.selected_index = index
            nll, responsibility = new_nll, new_responsibility
        self.accepted_updates = self.selected_index
        return self
