import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def fixture_inputs(tmp_path):
    batch = {key: np.array([6], dtype=np.float32) for key in
             ('x', 'static', 'climate', 'input_time', 'lead_time')}
    candidates = [dict(origin=f'2020-01-02T0{hour}:00:00+00:00',
                       latest='2020-01-02T00:00:00+00:00', partition='train',
                       year_start_boundary_uncertain=False) for hour in (1, 2)]
    source = tmp_path / 'source.txt'
    source.write_text('frozen')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    audit = dict(status='all_planned_batches_readable_not_training_ready',
                 source_sha256={str(source): digest}, candidates=candidates,
                 validated_candidates=2, unique_weather_pairs=1,
                 batch_shapes={k: [1] for k in batch},
                 pair_array_sha256={candidates[0]['latest']: {
                     k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in batch.items()}})
    path = tmp_path / 'audit.json'
    path.write_text(json.dumps(audit))

    class Loader:
        hashes = {str(source): digest}
        def load(self, origin):
            return batch, next(r for r in candidates if r['origin'] == origin)
        def verify_unchanged(self):
            pass
    return path, audit, Loader(), batch, source


def test_export_preserves_candidates_and_exact_arrays(tmp_path):
    from export_full_encoder_batches import export_batches
    path, audit, loader, batch, _ = fixture_inputs(tmp_path)
    result = export_batches(path, tmp_path / 'out', loader)
    assert result['status'] == 'complete'
    assert result['candidates'] == audit['candidates']
    assert len(result['rows']) == 1
    row = result['rows'][0]
    assert row['batch_array_sha256'] == audit['pair_array_sha256'][row['pair_id']]
    with np.load(tmp_path / 'out' / row['path']) as saved:
        assert set(saved.files) == set(batch)
        for key in batch:
            np.testing.assert_array_equal(saved[key], batch[key])


def test_changed_source_fails_without_complete_manifest(tmp_path):
    from export_full_encoder_batches import export_batches
    path, _, loader, _, source = fixture_inputs(tmp_path)
    source.write_text('changed')
    with pytest.raises(ValueError, match='Source checksum'):
        export_batches(path, tmp_path / 'out', loader)
    assert not (tmp_path / 'out' / 'manifest.json').exists()


def test_changed_array_fails(tmp_path):
    from export_full_encoder_batches import export_batches
    path, _, loader, batch, _ = fixture_inputs(tmp_path)
    batch['x'][0] = 9
    with pytest.raises(ValueError, match='Array checksum'):
        export_batches(path, tmp_path / 'out', loader)


def test_missing_pair_mapping_fails(tmp_path):
    from export_full_encoder_batches import export_batches
    path, audit, loader, _, _ = fixture_inputs(tmp_path)
    audit['pair_array_sha256'] = {}
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='Pair coverage'):
        export_batches(path, tmp_path / 'out', loader)
