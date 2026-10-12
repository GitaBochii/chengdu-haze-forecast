"""Discrete capped interval likelihood tests, including true tail semantics."""
import math
import unittest

import torch

from experiments.prithvi_wxc_probe_20260924.interval_loss import interval_survival_nll
from experiments.prithvi_wxc_probe_20260924.run_real_fusion_smoke import exact_survival_nll


class IntervalLossTests(unittest.TestCase):
    def test_exact_matches_existing_nll(self):
        torch.manual_seed(17)
        logits = torch.randn(3, 67, dtype=torch.float64)
        for hours in (1, 46, 67):
            actual = interval_survival_nll(logits, [hours]*3, [hours]*3)
            torch.testing.assert_close(actual, exact_survival_nll(logits, hours))

    def test_closed_interval_probability_not_midpoint(self):
        value = interval_survival_nll(torch.zeros(1, 67, dtype=torch.float64), [3], [5])
        self.assertAlmostEqual(value.item(), -math.log(1/8 + 1/16 + 1/32))

    def test_right_censor_includes_lower_bound(self):
        value = interval_survival_nll(torch.zeros(1, 67, dtype=torch.float64), [4], [float('inf')])
        self.assertAlmostEqual(value.item(), 3*math.log(2))

    def test_tail_projection(self):
        logits = torch.zeros(4, 67, dtype=torch.float64)
        value = interval_survival_nll(logits, [68, 84, 70, 67],
                                      [68, 84, float('inf'), float('inf')], reduction='none')
        torch.testing.assert_close(value, torch.tensor([67,67,67,66], dtype=torch.float64)*math.log(2))

    def test_full_uncertainty_zero_and_zero_gradient(self):
        logits = torch.randn(2, 67, requires_grad=True)
        loss = interval_survival_nll(logits, [1,1], [float('inf'), 68])
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0.)

    def test_invalid_bounds(self):
        for low, high in [(0,2),(3,2),(80,70),(1.5,3),(1,2.5),(1.000000001,3),
                          (float('inf'),float('inf')),(1,float('nan')),(1,-float('inf'))]:
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                interval_survival_nll(torch.zeros(1,67), [low], [high])
        with self.assertRaises(ValueError):
            interval_survival_nll(torch.zeros(2,67), [1], [2])
        with self.assertRaises(ValueError):
            interval_survival_nll(torch.zeros(1,66), [1], [2])
        with self.assertRaises(ValueError):
            interval_survival_nll(torch.full((1,67), float('nan')), [1], [2])

    def test_extreme_logits_finite_gradient(self):
        logits = torch.tensor([[10000.]*67, [-10000.]*67], requires_grad=True)
        loss = interval_survival_nll(logits, [60,2], [float('inf'), 5])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
