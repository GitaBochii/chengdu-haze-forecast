"""Focused tests for the independent learned-path audit."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

SPEC = importlib.util.spec_from_file_location('learned_independent', Path(__file__).with_name('verify_independent.py'))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class IndependentAuditTests(unittest.TestCase):
    def test_bootstrap_multiplicity_and_tree_average(self):
        bootstrap = np.array([[0, 0, 1, 2], [1, 2, 2, 3]])
        train_leaf = np.array([[7, 7], [7, 7], [7, 9], [9, 9]])
        actual = audit.recompute_weights(bootstrap, train_leaf, np.array([7, 9]))
        np.testing.assert_allclose(actual, np.array([1/4, 1/8, 1/8+1/3, 1/6]))
        self.assertAlmostEqual(actual.sum(), 1.)
        self.assertFalse(np.allclose(actual, np.array([1/6, 1/6, 1/3, 1/3])))

    def test_empty_bootstrap_leaf_fails(self):
        with self.assertRaises(ValueError):
            audit.recompute_weights(np.array([[0, 0]]), np.array([[1], [2]]), np.array([2]))

    def test_invalid_weight_distribution_fails(self):
        paths = np.zeros((2, 24, 3))
        for weights in (np.array([.2, .2]), np.array([-.1, 1.1]), np.array([np.nan, 1]), np.array([.5, .500001])):
            with self.assertRaises(ValueError):
                audit.distribution(weights, paths, np.array(['a', 'b']))

    def test_only_probability_roundoff_is_clipped(self):
        np.testing.assert_array_equal(audit.roundoff_bounds(np.array([-1e-15, 1e-15, .3, 1-1e-15, 1+1e-15])), [0., 0., .3, 1., 1.])
        for probability in (np.array([-.001]), np.array([1.001]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                audit.roundoff_bounds(probability)

    def test_300_tree_degenerate_certain_recovery_at_theta_one(self):
        weights = audit.recompute_weights(np.zeros((300, 1), dtype=int), np.ones((1, 300), dtype=int), np.ones(300, dtype=int))
        paths = np.full((1, 24, 3), 80.)
        products = audit.distribution(weights, paths, np.array(['same']))
        self.assertEqual(weights[0], 1.)
        self.assertEqual(products['pmf'][0], 1.)
        self.assertEqual(products['cdf'][-1], 1.)
        self.assertEqual(audit.decision(products['cdf'][None], 1.)[0], 1)
        self.assertEqual(audit.decision(products['cdf'][None], 1.000001)[0], 20)

    def test_censor_mass_and_mid_probability_are_not_snapped(self):
        paths = np.full((2, 24, 3), 80.)
        paths[1, :, 2] = 180.
        products = audit.distribution(np.array([.7, .3]), paths, np.array(['a', 'b']))
        self.assertAlmostEqual(products['pmf'][-1], .3)
        self.assertAlmostEqual(products['cdf'][-1], .7)
        self.assertEqual(audit.decision(products['cdf'][None], .70000001)[0], 20)
        self.assertEqual(audit.decision(products['cdf'][None], .69999999)[0], 1)

    def test_pathwise_not_recovery_of_mean(self):
        paths = np.zeros((3, 24, 3))
        paths[0, :, 2] = 200
        paths[1, :, 2] = 50
        paths[2, :, 2] = 50
        paths[2, :18, 2] = 200
        dist = audit.distribution(np.array([.2, .3, .5]), paths, np.array(['a', 'b', 'b']))
        np.testing.assert_allclose(dist['pmf'][[0, 18, 19]], [.3, .5, .2])
        self.assertAlmostEqual(dist['cdf'][-1], .8)
        self.assertAlmostEqual(dist['onset_probability'], .7)
        self.assertAlmostEqual(dist['effective_support'], 1/.38)
        self.assertEqual(dist['nonzero_support'], 3)
        self.assertEqual(dist['selected_group_count'], 2)
        self.assertAlmostEqual(dist['max_group_weight'], .8)

    def test_all_reject_and_missing_classes_fail_selection(self):
        groups = np.array(['a', 'a', 'b', 'b'])
        t = np.array([1, 2, 20, 20])
        rows = [dict(theta=1.000001, **audit.metric_schema(t, np.full(4, 20), groups))]
        self.assertIsNone(audit.choose(rows, 1., 'theta', require_detection=True))
        single = dict(theta=.5, **audit.metric_schema(np.array([1, 2]), np.array([1, 2]), groups[:2]))
        self.assertIsNone(audit.choose([single], 1., 'theta', require_detection=True))

    def test_deepcopy_preserves_additional_fields(self):
        values = {'probability': {'brier': .2, 'nested': [1, 2]}}
        detached = audit.detached_metrics(values)
        detached['probability']['nested'].append(3)
        detached['probability']['extra'] = .1
        self.assertEqual(values, {'probability': {'brier': .2, 'nested': [1, 2]}})

    def test_sha_and_freeze_chinese_path(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder)/'中文测试.txt'
            path.write_text('fixed', encoding='utf-8')
            frozen = audit.freeze_paths([path])
            audit.assert_frozen(frozen)
            path.write_text('changed', encoding='utf-8')
            with self.assertRaises(AssertionError):
                audit.assert_frozen(frozen)

    def test_no_origin_or_future_in_feature_routing(self):
        dates = pd.date_range('2018-01-01', periods=90, freq='h', tz='Asia/Shanghai')
        raw = pd.DataFrame({'timestamp': dates, 'pm2_5': np.arange(90.), 'pm10': np.arange(90.)+2,
                            'aqi': np.arange(90.)+20})
        _, _, first, _ = audit.rebuild(raw)
        changed = raw.copy()
        changed.loc[45:, ['pm2_5', 'pm10', 'aqi']] += 400
        _, _, second, _ = audit.rebuild(changed)
        left = np.flatnonzero(first['indices'] == 45)[0]
        right = np.flatnonzero(second['indices'] == 45)[0]
        np.testing.assert_array_equal(first['features'][left], second['features'][right])
        self.assertNotEqual(first['origin_aqi'][left], second['origin_aqi'][right])
        self.assertFalse(np.array_equal(first['paths'][left], second['paths'][right]))
        self.assertEqual(first['features'].shape[1], 76)

    def test_saved_forest_complete_independent_roundtrip(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(96, 76))
        query_features = rng.normal(size=(20, 76))
        paths = rng.uniform(30, 160, size=(96, 24, 3))
        labels = audit.recovery(paths[:, :, 2])
        groups = np.array([f'process{row//12}' for row in range(96)])
        forest = RandomForestRegressor(n_estimators=300, criterion='squared_error', bootstrap=True,
            min_samples_leaf=30, max_features=1., max_depth=None, random_state=42, n_jobs=4)
        forest.fit(features, paths[:, :, 2])
        bootstrap = np.asarray(forest.estimators_samples_, dtype=np.int32)
        train_leaf, query_leaf = forest.apply(features), forest.apply(query_features)
        dense = np.stack([audit.recompute_weights(bootstrap, train_leaf, leaf) for leaf in query_leaf])
        weights = sparse.csr_matrix(dense)
        products = [audit.distribution(row, paths, groups) for row in dense]
        pred = {key: np.stack([product[key] for product in products]) for key in ('pmf', 'cdf', 'onset_probability', 'path_mean')}
        pred.update(path_quantiles=np.stack([audit.weighted_quantiles(paths[row > 0], row[row > 0]) for row in dense]),
            forest_aqi_mean=forest.predict(query_features),
            unconditional_cdf=(labels[:, None] <= np.arange(1, 20)).mean(axis=0),
            unconditional_onset=np.any(paths[:, :, 2] > 100, axis=1).mean(),
            median_start=np.argmax(np.cumsum(pred['pmf'], axis=1) >= .5, axis=1)+1,
            covered=np.ones(20, dtype=bool))
        support = pd.DataFrame(dict(dev_row=np.arange(20),
            nonzero_support=[product['nonzero_support'] for product in products],
            effective_support=[product['effective_support'] for product in products],
            distinct_processes=[product['selected_group_count'] for product in products],
            max_process_weight=[product['max_group_weight'] for product in products]))
        usage = pd.DataFrame(dict(train_row=np.arange(96), query_support_count=(dense > 0).sum(axis=0), total_weight=dense.sum(axis=0)))
        train = dict(indices=np.arange(96), features=features, paths=paths, labels=labels, groups=groups)
        dev = dict(indices=np.arange(20), features=query_features)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            run = Path(folder)
            joblib.dump(forest, run/'forest.joblib')
            np.savez(run/'forest_structure.npz', bootstrap_indices=bootstrap, train_leaves=train_leaf, dev_leaves=query_leaf)
            sparse.save_npz(run/'weights.npz', weights)
            support.to_parquet(run/'support.parquet')
            usage.to_parquet(run/'train_support_usage.parquet')
            record = {'checks': []}
            cdf = audit.verify_forest(run, train, dev, pred, record)
            np.testing.assert_allclose(cdf, pred['cdf'])
            self.assertTrue(record['independently_reloaded_forest'])
            pred['forest_aqi_mean'][0, 0] += 1
            with self.assertRaises(AssertionError):
                audit.verify_forest(run, train, dev, pred, {'checks': []})


if __name__ == '__main__':
    unittest.main()
