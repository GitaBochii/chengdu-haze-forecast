"""Read and validate every preserved origin through the lazy batch loader."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_history_batch_loader import FullHistoryBatchLoader, describe_origin
from raw_dynamic_adapter import HERE
from audit_full_import import parquet, derive_requirements
from scripts.import_merra_history import atomic_json


def main():
    loader = FullHistoryBatchLoader()
    source = loader.base / 'support_aligned_plan/candidate_hours.parquet'
    candidates = parquet(source)
    derive_requirements(candidates)
    if len(candidates) != len(loader.origins):
        raise ValueError('Candidate count mismatch')
    for name in ('full_history_batch_loader.py', 'raw_inference_batch.py', 'audit_full_history_batches.py'):
        loader.verify(HERE / name)
    rows, pairs = [], {}
    shapes = {'x': (1, 2, 160, 30, 32), 'static': (1, 10, 30, 32),
              'climate': (1, 160, 30, 32), 'input_time': (1,), 'lead_time': (1,)}
    for candidate in candidates:
        batch, metadata = loader.load(candidate['timestamp'])
        if metadata['training_allowed'] is not False or metadata['operational_availability_verified'] is not False:
            raise ValueError('Unexpected training or operational authorization')
        if metadata['partition'] != candidate['partition']:
            raise ValueError('Partition changed')
        for key, column in [('earlier', 'support_earlier_utc'), ('latest', 'support_latest_utc'),
                            ('support_end', 'support_end_utc')]:
            if datetime.fromisoformat(metadata[key]) != datetime.fromisoformat(candidate[column]):
                raise ValueError('Support plan mismatch')
        if datetime.fromisoformat(metadata['support_end']) > datetime.fromisoformat(metadata['origin']):
            raise ValueError('Weather physical support extends beyond pollution origin')
        if set(batch) != set(shapes):
            raise ValueError('Inference keys changed')
        digests = {}
        for key, shape in shapes.items():
            value = batch[key]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid batch array: ' + key)
            digests[key] = hashlib.sha256(value.tobytes(order='C')).hexdigest()
        if batch['input_time'].item() != 6 or batch['lead_time'].item() != 6:
            raise ValueError('Unexpected interval')
        pair = metadata['latest']
        if pair in pairs and pairs[pair] != digests:
            raise ValueError('Shared weather pair produced inconsistent input')
        pairs[pair] = digests
        rows.append(metadata)
        if len(rows) % 1000 == 0:
            print(json.dumps({'validated_origins': len(rows), 'required': len(candidates)}), flush=True)
    loader.verify_unchanged()
    report = {'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'all_planned_batches_readable_not_training_ready',
              'validated_candidates': len(rows), 'unique_weather_pairs': len(pairs),
              'batch_shapes': {k: list(v) for k, v in shapes.items()},
              'counts_by_partition': dict(Counter(row['partition'] for row in rows)),
              'boundary_by_partition': dict(Counter(row['partition'] for row in rows if row['year_start_boundary_uncertain'])),
              'all_physical_support_ends_at_or_before_origin': True,
              'source_sha256': loader.hashes, 'pair_array_sha256': pairs, 'candidates': rows,
              'training_allowed': False, 'full_history_preprocessing_verified': False,
              'operational_availability_verified': False,
              'scope': 'Every frozen origin assembled and numerically checked; not encoder execution or predictive accuracy evaluation.'}
    destination = loader.base / 'full_batch_read_audit.json'
    atomic_json(destination, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'pair_array_sha256', 'candidates')}, indent=2))


if __name__ == '__main__':
    main()
