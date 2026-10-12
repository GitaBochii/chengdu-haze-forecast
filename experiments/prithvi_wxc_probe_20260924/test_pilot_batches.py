import tempfile
from pathlib import Path
import unittest
import hashlib
import torch
from experiments.prithvi_wxc_probe_20260924.prepare_pilot_batches import check_sources, validate_batch


class BatchTests(unittest.TestCase):
    def test_missing_partial_not_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.partial').write_bytes(b'ab')
            entries = [{'path': 'a', 'size': 2, 'lfs': {'oid': hashlib.sha256(b'ab').hexdigest()}}]
            self.assertEqual(check_sources(root, entries)['status'], 'missing')

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a').write_bytes(b'xx')
            entries = [{'path': 'a', 'size': 2, 'lfs': {'oid': hashlib.sha256(b'ab').hexdigest()}}]
            self.assertEqual(check_sources(root, entries)['status'], 'failed')

    def test_batch_rejects_y_shape_and_nonfinite(self):
        batch = {'x': torch.zeros(1, 2, 160, 30, 32), 'static': torch.zeros(1, 10, 30, 32),
                 'climate': torch.zeros(1, 160, 30, 32), 'input_time': torch.tensor([6.]),
                 'lead_time': torch.tensor([6.])}
        validate_batch(batch)
        for changed in [dict(batch, y=torch.zeros(1)), dict(batch, x=torch.zeros(1)),
                        dict(batch, lead_time=torch.tensor([float('nan')]))]:
            with self.assertRaises(ValueError):
                validate_batch(changed)


if __name__ == '__main__':
    unittest.main()
