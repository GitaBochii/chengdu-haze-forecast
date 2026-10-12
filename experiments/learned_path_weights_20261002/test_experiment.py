"""Small-data contract tests; no source evaluation or full training."""
import inspect
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

import run_experiment as main


class LeafWeightTests(unittest.TestCase):
    def test_bootstrap_multiplicity_is_not_unique_sample_average(self):
        bootstrap = np.array([[0, 1, 1]])
        train_leaves = np.array([[2], [2], [7]])
        query_leaves = np.array([[2]])
        index = main.leaf_index(bootstrap, train_leaves)
        weights = main.forest_weights(index, query_leaves, 3)
        np.testing.assert_allclose(weights.toarray(), [[1 / 3, 2 / 3, 0]])
        self.assertAlmostEqual(float((weights @ np.array([10., 20., 99.]))[0]), 50 / 3)

    def test_each_tree_normalized_before_equal_tree_average(self):
        bootstrap = np.array([[0, 0, 0], [1, 2, 2]])
        leaves = np.array([[2, 8], [7, 4], [7, 4]])
        weights = main.forest_weights(main.leaf_index(bootstrap, leaves), np.array([[2, 4]]), 3)
        np.testing.assert_allclose(weights.toarray(), [[.5, 1 / 6, 1 / 3]])

    def test_real_multitarget_forest_predict_equals_bootstrap_path_mean(self):
        rng = np.random.RandomState(42)
        x = rng.normal(size=(95, 4))
        y = rng.uniform(0, 200, size=(95, 24))
        query = rng.normal(size=(8, 4))
        rf = RandomForestRegressor(n_estimators=7, min_samples_leaf=3, random_state=42, n_jobs=1).fit(x, y)
        bootstrap, train_leaves, query_leaves = main.forest_structure(rf, x, query)
        weights = main.forest_weights(main.leaf_index(bootstrap, train_leaves), query_leaves, len(x))
        self.assertIsInstance(weights, sparse.csr_matrix)
        np.testing.assert_allclose(np.asarray(weights.sum(axis=1)).ravel(), 1, atol=1e-13)
        np.testing.assert_allclose(weights @ y, rf.predict(query), atol=1e-10, rtol=1e-12)
        self.assertTrue((weights.data > 0).all())

    def test_weights_api_has_no_query_future_or_issue_aqi(self):
        self.assertEqual(list(inspect.signature(main.forest_weights).parameters), ['index', 'query_leaves', 'n_train'])

    def test_sparse_and_model_roundtrip_unicode_directory(self):
        x = np.arange(60, dtype=float).reshape(20, 3)
        y = np.repeat(np.arange(20)[:, None], 24, axis=1)
        rf = RandomForestRegressor(n_estimators=3, min_samples_leaf=2, random_state=42).fit(x, y)
        b, t, q = main.forest_structure(rf, x, x[:3])
        w = main.forest_weights(main.leaf_index(b, t), q, len(x))
        with tempfile.TemporaryDirectory(prefix='learned-path-') as tmp:
            directory = Path(tmp) / '\u6a21\u578b'
            directory.mkdir()
            sparse.save_npz(directory / 'weights.npz', w)
            joblib.dump(rf, directory / 'forest.joblib')
            np.testing.assert_array_equal(sparse.load_npz(directory / 'weights.npz').toarray(), w.toarray())
            np.testing.assert_array_equal(joblib.load(directory / 'forest.joblib').predict(x), rf.predict(x))

    def test_reload_equivalence_allows_only_floating_accumulation_noise(self):
        expected = np.zeros((1, 24))
        self.assertAlmostEqual(main.assert_forest_equivalent(expected, expected + 9e-11), 9e-11)
        with self.assertRaises(AssertionError):
            main.assert_forest_equivalent(expected, expected + 1.1e-10)


class PathDistributionTests(unittest.TestCase):
    def fixture(self):
        paths = np.full((3, 24, 3), 150.)
        paths[0, :6, 2] = 80
        paths[1, 18:, 2] = 90
        return paths

    def test_recovery_preserves_late_start_and_censored_tail(self):
        paths = self.fixture()
        np.testing.assert_array_equal(main.recovery(paths[:, :, 2]), [1, 19, 20])

    def test_weighted_distribution_is_pathwise_not_mean_curve(self):
        paths = self.fixture()
        w = sparse.csr_matrix([[.25, .25, .5]])
        products = main.path_products(w, paths)
        self.assertEqual(products['pmf'].shape, (1, 20))
        self.assertAlmostEqual(products['pmf'][0, 0], .25)
        self.assertAlmostEqual(products['pmf'][0, 18], .25)
        self.assertAlmostEqual(products['pmf'][0, 19], .5)
        self.assertAlmostEqual(products['cdf'][0, 18], .5)
        np.testing.assert_array_equal(main.decision(products['cdf'], 1.000001), [20])
        self.assertEqual(main.recovery(products['path_mean'][:, :, 2])[0], 20)
        self.assertEqual(products['onset_probability'][0], 1.)
        np.testing.assert_allclose(products['path_mean'], (w @ paths.reshape(3, 72)).reshape(1, 24, 3))
        self.assertEqual(products['path_quantiles'].shape, (1, 3, 24, 3))

    def test_support_counts_bootstrap_weighted_process_mass(self):
        w = sparse.csr_matrix([[.2, .3, .5], [0, 1., 0]])
        table, usage = main.support_diagnostics(w, np.array(['a', 'a', 'b']))
        np.testing.assert_array_equal(table.nonzero_support, [3, 1])
        np.testing.assert_array_equal(table.distinct_processes, [2, 1])
        np.testing.assert_allclose(table.effective_support, [1 / .38, 1])
        np.testing.assert_allclose(table.max_process_weight, [.5, 1])
        np.testing.assert_array_equal(usage.query_support_count, [1, 2, 1])

    def test_invalid_nonfinite_or_non24_paths_rejected(self):
        with self.assertRaises(ValueError):
            main.recovery(np.full((1, 23), 100.))
        with self.assertRaises(ValueError):
            main.recovery(np.full((1, 24), np.nan))

    def test_roundoff_does_not_exceed_one_or_drop_reliability_bin(self):
        paths = np.full((1, 24, 3), 80.)
        products = main.path_products(sparse.csr_matrix([[1 + 1e-15]]), paths)
        self.assertLessEqual(products['cdf'].max(), 1)
        paths[:, :, 2] = 150
        products = main.path_products(sparse.csr_matrix([[1 + 1e-15]]), paths)
        self.assertLessEqual(products['onset_probability'].max(), 1)

    def test_300_trees_deterministic_recovery_keeps_exact_theta_one(self):
        bootstrap = np.zeros((300, 1), dtype=np.int32)
        leaves = np.ones((1, 300), dtype=np.int32)
        weights = main.forest_weights(main.leaf_index(bootstrap, leaves), leaves, 1)
        products = main.path_products(weights, np.full((1, 24, 3), 80.))
        self.assertEqual(float(weights.sum()), 1.)
        self.assertEqual(products['cdf'][0, 18], 1.)
        np.testing.assert_array_equal(main.decision(products['cdf'], 1.), [1])
        np.testing.assert_array_equal(main.decision(products['cdf'], 1.000001), [20])

    def test_300_trees_censored_mass_is_not_snapped_into_recovery(self):
        bootstrap = np.repeat(np.array([[0, 1, 1]]), 300, axis=0)
        leaves = np.repeat(np.array([[1], [1], [7]]), 300, axis=1)
        query_leaves = np.ones((1, 300), dtype=np.int32)
        weights = main.forest_weights(main.leaf_index(bootstrap, leaves), query_leaves, 3)
        paths = np.full((3, 24, 3), 150.)
        paths[0, :, 2] = 80.
        products = main.path_products(weights, paths)
        self.assertAlmostEqual(float(weights.sum()), 1., places=15)
        self.assertAlmostEqual(products['cdf'][0, 18], 1 / 3, places=14)
        self.assertAlmostEqual(products['pmf'][0, 19], 2 / 3, places=14)
        np.testing.assert_array_equal(main.decision(products['cdf'], 1.), [20])

    def test_probability_endpoint_policy_does_not_relax_midrange_theta(self):
        probability = np.array([1 - 4e-15, 4e-15, .5 - 4e-15])
        fixed = main.probability_endpoints(probability)
        np.testing.assert_array_equal(fixed[:2], [1., 0.])
        self.assertEqual(fixed[2], probability[2])
        np.testing.assert_array_equal(main.decision(np.full((1, 19), fixed[2]), .5), [20])


class SelectionTests(unittest.TestCase):
    def test_all_rejection_and_missing_positive_or_negative_fail(self):
        g = np.array(['a', 'a'])
        rows = [dict(theta=1.000001, **main.metrics(np.array([1, 20]), np.array([20, 20]), g))]
        self.assertIsNone(main.select_theta(rows, 1.))
        rows = [dict(theta=.5, **main.metrics(np.array([1, 2]), np.array([1, 2]), g))]
        self.assertIsNone(main.select_theta(rows, 1.))
        rows = [dict(theta=.5, **main.metrics(np.array([20, 20]), np.array([20, 20]), g))]
        self.assertIsNone(main.select_theta(rows, 1.))

    def test_tie_prefers_higher_theta_and_does_not_relax_early_limit(self):
        base = main.metrics(np.array([1, 20]), np.array([1, 20]), np.array(['a', 'b']))
        rows = [dict(theta=.4, **base), dict(theta=.5, **base)]
        self.assertEqual(main.select_theta(rows, 0.)['theta'], .5)
        rows[1]['early_fraction'] = .1
        self.assertEqual(main.select_theta(rows, 0.)['theta'], .4)

    def test_frozen_configs_and_thresholds(self):
        self.assertEqual(main.FOREST_CONFIG, dict(n_estimators=300, criterion='squared_error', bootstrap=True,
            min_samples_leaf=30, max_features=1.0, max_depth=None, random_state=42, n_jobs=4))
        self.assertEqual(main.LGBM_CONFIG['n_estimators'], 300)
        self.assertEqual(main.LGBM_CONFIG['objective'], 'regression_l1')
        self.assertEqual(main.THRESHOLDS, [round(i * .05, 2) for i in range(1, 21)] + [1.000001])

    def test_buffer_uses_own_raw_early_limit_and_reports_all_102_candidates(self):
        curve = np.stack([np.full(24, 90.), np.full(24, 110.)])
        truth = np.array([1, 20])
        delta, rows = main.select_buffer(curve, truth, np.array(['a', 'b']))
        self.assertEqual(len(rows), 102)
        self.assertEqual(delta, 10)
        self.assertEqual(rows[10]['mae'], 0)

    def test_integrated_probability_metric_includes_process_equal(self):
        truth = np.array([1, 20, 20])
        cdf = np.zeros((3, 19))
        score = main.probability_metrics(cdf, truth, np.array(['a', 'b', 'b']))
        self.assertAlmostEqual(score['brier_mean_1_19'], 1 / 3)
        self.assertAlmostEqual(score['group_equal_brier_mean_1_19'], .5)

    def test_empty_clean_population_and_missing_negative_are_failures_not_nan(self):
        summary = main.clean_metrics(np.array([.2]), np.array([True]), np.array([20]),
            np.array([20]), np.array(['a']), .3)
        self.assertIsNone(summary['fpr'])
        self.assertEqual(summary['negative_n'], 0)

    def test_package_freeze_contains_installed_code_record(self):
        files = main.dependency_files()
        self.assertTrue(any(p.name == '_forest.py' for p in files))
        self.assertGreaterEqual(sum(p.name == 'RECORD' for p in files), 7)

    def test_artifact_contract_names_are_explicit(self):
        self.assertIn('weights.npz', main.ARTIFACT_SCHEMA)
        self.assertIn('prediction.npz', main.ARTIFACT_SCHEMA)
        self.assertEqual(main.ARTIFACT_SCHEMA['forest_structure.npz']['bootstrap_indices'], '(300,n_train) int32')


if __name__ == '__main__':
    unittest.main()
