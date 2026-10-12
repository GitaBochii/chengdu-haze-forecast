"""Fixed regularized linear AFT candidate, isolated from deployed tree models."""

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def linear_state(booster):
    native = json.loads(booster.save_raw(raw_format='json'))
    model = native['learner']['gradient_booster']
    if model['name'] != 'gblinear':
        raise ValueError('Require a native linear booster')
    weights = np.asarray(model['model']['weights'], dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite linear coefficients')
    return weights[:-1], float(weights[-1])


class _TailState(xgb.callback.TrainingCallback):
    def __init__(self):
        self.states = []

    def after_iteration(self, model, epoch, evals_log):
        if epoch >= 949:
            weights, intercept = linear_state(model)
            self.states.append({
                'iteration': epoch, 'weights': weights.tolist(), 'intercept': intercept,
                'native_nll': float(evals_log['train']['aft-nloglik'][-1]),
            })
        return False


class LinearIntervalAFT:
    """Train-only imputation/scaling and an unpenalized log-time intercept.

    The linear booster is deprecated in XGBoost 3.4.1. This version-frozen
    research candidate is intentionally not wired into the deployment factory.
    """

    @staticmethod
    def parameters():
        return {
            'booster': 'gblinear', 'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.2,
            'updater': 'coord_descent', 'feature_selector': 'cyclic',
            'lambda': 1.0, 'alpha': 0.0, 'eta': 0.1, 'base_score': 1.0,
            'nthread': 1, 'device': 'cpu', 'seed': 42, 'verbosity': 0,
        }

    @staticmethod
    def _numeric(frame):
        if (not isinstance(frame, pd.DataFrame) or frame.empty
                or not frame.columns.is_unique or not frame.index.is_unique):
            raise ValueError('Require nonempty features with unique columns and index')
        numeric = frame.astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError('Infinite features are not supported')
        return numeric

    def fit_bounds(self, features, lower, upper):
        numeric = self._numeric(features)
        bounds = []
        for values in (lower, upper):
            if isinstance(values, pd.Series) and not values.index.equals(features.index):
                raise ValueError('Label index must match the feature index')
            values = np.asarray(values, dtype=float)
            if values.shape != (len(features),):
                raise ValueError('Require aligned one-dimensional label bounds')
            bounds.append(values)
        lo, hi = bounds
        if (not np.isfinite(lo).all() or (lo <= 0).any()
                or np.isnan(hi).any() or (hi < lo).any()):
            raise ValueError('Require positive finite lower and upper >= lower')
        with np.errstate(over='ignore', under='ignore'):
            lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
        if (not np.isfinite(lo32).all() or (lo32 <= 0).any()
                or not np.array_equal(np.isfinite(hi), np.isfinite(hi32))
                or not np.array_equal(lo == hi, lo32 == hi32)):
            raise ValueError('Label bounds must preserve their meaning in float32')

        self.feature_names = list(numeric.columns)
        self.all_missing_features = numeric.columns[numeric.isna().all()].tolist()
        if len(self.all_missing_features) == numeric.shape[1]:
            raise ValueError('No observed training features')
        self.observed_features = [c for c in self.feature_names if c not in self.all_missing_features]
        self.imputer = SimpleImputer(strategy='median', add_indicator=True)
        filled = self.imputer.fit_transform(numeric[self.observed_features])
        names = self.imputer.get_feature_names_out().tolist()
        nonconstant = np.ptp(filled, axis=0) > 0
        unique = ~pd.DataFrame(filled).T.duplicated().to_numpy()
        self.keep_indices = np.flatnonzero(nonconstant & unique)
        if not len(self.keep_indices):
            raise ValueError('No nonconstant training features')
        self.kept_features = [names[i] for i in self.keep_indices]
        self.dropped_design_features = [name for i, name in enumerate(names)
                                        if i not in self.keep_indices]
        self.scaler = StandardScaler().fit(filled[:, self.keep_indices])
        transformed = self.transform(numeric)
        matrix = xgb.DMatrix(transformed)
        matrix.set_float_info('label_lower_bound', lo)
        matrix.set_float_info('label_upper_bound', hi)
        tail = _TailState()
        self.booster = xgb.train(
            self.parameters(), matrix, num_boost_round=1000,
            evals=[(matrix, 'train')], callbacks=[tail], verbose_eval=False,
        )
        self.training_trace = tail.states
        self.predict(numeric)
        return self

    def transform(self, features):
        if not hasattr(self, 'scaler'):
            raise ValueError('Model must be fitted before transformation')
        numeric = self._numeric(features)
        if list(numeric.columns) != self.feature_names:
            raise ValueError('Prediction columns must match training names and order')
        filled = self.imputer.transform(numeric[self.observed_features])
        matrix = self.scaler.transform(filled[:, self.keep_indices])
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite transformed features')
        return pd.DataFrame(matrix, columns=self.kept_features, index=features.index)

    def _prediction_matrix(self, features):
        if not hasattr(self, 'booster'):
            raise ValueError('Model must be fitted before prediction')
        return xgb.DMatrix(self.transform(features))

    def predict_margin(self, features):
        matrix = self._prediction_matrix(features)
        result = self.booster.predict(matrix, output_margin=True)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite log-time prediction')
        return result

    def predict(self, features):
        matrix = self._prediction_matrix(features)
        result = self.booster.predict(matrix)
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError('Nonfinite or nonpositive time prediction; do not drop origins')
        return result
