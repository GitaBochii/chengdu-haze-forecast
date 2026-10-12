import tempfile
import unittest
import json
from pathlib import Path
import torch
from experiments.prithvi_wxc_probe_20260924.run_real_fusion_smoke import exact_survival_nll, preflight


class SmokeTests(unittest.TestCase):
    def test_exact_nll(self):
        logits = torch.zeros(1, 67, requires_grad=True)
        loss = exact_survival_nll(logits, 46)
        torch.testing.assert_close(loss, torch.tensor(46.) * torch.log(torch.tensor(2.)))
        loss.backward()
        self.assertEqual(logits.grad[0, 45].item(), -0.5)
        self.assertEqual(logits.grad[0, 46:].abs().sum().item(), 0)

    def test_invalid_targets(self):
        for value in (0, 68, 2.5, True):
            with self.assertRaises(ValueError):
                exact_survival_nll(torch.zeros(1, 67), value)

    def test_missing_artifacts_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            result = preflight(Path(temp))
            self.assertFalse(result['ready'])
            self.assertIn('missing', result['reason'])

    def test_unverified_or_hash_mismatched_artifacts_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ['pretrained_spatial_embedding.pt', 'official_batch.pt', 'paired_sample_report.json']:
                (folder / name).write_bytes(b'not trusted')
            path = folder / 'pretrained_encoder_report.json'
            path.write_text(json.dumps({'status': 'random_architecture_passed'}))
            self.assertEqual(preflight(folder)['reason'], 'unverified_pretrained_encoder')
            path.write_text(json.dumps({'status': 'real_pretrained_encoder_passed',
                                       'pretrained_inference_done': True,
                                       'revision': 'e2f62b3e19890f1a5e225bb123acb4b33df85ef8',
                                       'load': {'strict_load': True}, 'future_truth_used': False,
                                       'embedding_file_sha256': '0' * 64}))
            self.assertIn('hash_mismatch', preflight(folder)['reason'])


if __name__ == '__main__':
    unittest.main()
