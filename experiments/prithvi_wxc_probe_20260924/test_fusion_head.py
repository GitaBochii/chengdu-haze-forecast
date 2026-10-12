"""Tensor contract tests; no pretrained-inference or accuracy claims."""
import unittest
import numpy as np
import torch
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from experiments.prithvi_wxc_probe_20260924.fusion_head import SpatialRecoveryFusion


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.base = RecoveryPM25GNN(17, np.array([[104., 30.], [104.1, 30.1]]),
                                   torch.zeros(2), torch.ones(2))
        self.model = SpatialRecoveryFusion(self.base, embedding_dim=8)
        self.x = torch.randn(2, 24, 2, 17)
        self.active = torch.ones(2, 2, 2, dtype=torch.bool)
        self.embedding = torch.randn(2, 8)
        self.present = torch.ones(2, dtype=torch.bool)

    def test_initial_identity_and_frozen_base(self):
        curves, logits = self.model(self.x, self.active, self.embedding, self.present)
        expected_curves, expected_logits = self.base(self.x, self.active)
        torch.testing.assert_close(curves, expected_curves, rtol=0, atol=0)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        self.assertEqual(logits.shape, (2, 67))
        self.model.train()
        self.assertFalse(self.base.training)
        self.assertTrue(all(not p.requires_grad for p in self.base.parameters()))

    def test_learns_embedding_and_missing_strict_bypass(self):
        optimizer = torch.optim.SGD(self.model.residual.parameters(), lr=0.1)
        _, logits = self.model(self.x, self.active, self.embedding, self.present)
        logits.square().mean().backward()
        self.assertGreater(self.model.residual[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        optimizer.step()
        emb = self.embedding.clone().requires_grad_()
        _, out = self.model(self.x, self.active, emb, self.present)
        _, changed = self.model(self.x, self.active, -emb, self.present)
        self.assertFalse(torch.equal(out, changed))
        out.sum().backward()
        self.assertGreater(emb.grad.abs().sum().item(), 0)
        missing = torch.full_like(emb, float('nan'))
        _, fallback = self.model(self.x, self.active, missing, ~self.present)
        _, baseline = self.base(self.x, self.active)
        torch.testing.assert_close(fallback, baseline, rtol=0, atol=0)

    def test_rejects_invalid_observed_and_shape(self):
        for embedding, present in [(self.embedding[:, :7], self.present),
                                   (self.embedding, self.present.float()),
                                   (torch.full_like(self.embedding, float('nan')), self.present)]:
            with self.assertRaises(ValueError):
                self.model(self.x, self.active, embedding, present)

    def test_optional_trainable_base(self):
        model = SpatialRecoveryFusion(self.base, embedding_dim=8, freeze_baseline=False)
        self.assertTrue(all(p.requires_grad for p in self.base.parameters()))
        _, output = model(self.x, self.active, self.embedding, self.present)
        output.sum().backward()
        self.assertIsNotNone(self.base.risk_head.weight.grad)


if __name__ == '__main__':
    unittest.main()
