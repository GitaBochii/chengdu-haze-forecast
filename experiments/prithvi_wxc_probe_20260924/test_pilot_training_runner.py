import copy
import numpy as np
import pytest
import torch

from experiments.prithvi_wxc_probe_20260924.run_pilot_training import (
    make_matched_models, require_complete_rows, verify_source_hashes)


def test_gate_rejects_missing_duplicate_or_changed_partition():
    frozen = [dict(timestamp=str(i), partition='train') for i in range(25)]
    rows = [dict(**r, status='ready') for r in frozen]
    assert len(require_complete_rows(rows, frozen)) == 25
    for broken in [rows[:-1], rows[:-1] + [rows[0]]]:
        with pytest.raises(ValueError):
            require_complete_rows(broken, frozen)
    broken = copy.deepcopy(rows)
    broken[0]['status'] = 'awaiting_batch'
    with pytest.raises(ValueError):
        require_complete_rows(broken, frozen)
    broken[0]['status'] = 'ready'
    broken[0]['partition'] = 'test'
    with pytest.raises(ValueError):
        require_complete_rows(broken, frozen)


def test_matched_initialization_and_trainable_pollution_branches():
    coords = np.array([[104., 30.], [104.1, 30.1]])
    models = make_matched_models(24, coords, np.zeros(17), np.ones(17))
    original = models['baseline'].state_dict()
    for arm in ('capacity_control', 'spatial_fusion'):
        model = models[arm]
        assert not model.freeze_baseline
        assert all(p.requires_grad for p in model.baseline.parameters())
        for k, value in model.baseline.state_dict().items():
            torch.testing.assert_close(value, original[k], rtol=0, atol=0)
    for left, right in zip(models['capacity_control'].residual.parameters(),
                           models['spatial_fusion'].residual.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_source_guard_rejects_changes(tmp_path):
    import hashlib
    path = tmp_path/'source'
    path.write_bytes(b'original')
    expected = {'source':hashlib.sha256(b'original').hexdigest()}
    assert verify_source_hashes(tmp_path, expected) == expected
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        verify_source_hashes(tmp_path, expected)
