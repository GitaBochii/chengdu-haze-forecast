"""Export audited NumPy batches without importing torch or changing source data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_history_batch_loader import FullHistoryBatchLoader
from raw_dynamic_adapter import HERE
from scripts.download_merra_regional_sample import file_hash
from scripts.import_merra_history import atomic_json


KEYS = {'x', 'static', 'climate', 'input_time', 'lead_time'}


def verify_sources(sources):
    for path, digest in sources.items():
        if file_hash(Path(path)) != digest:
            raise ValueError('Source checksum mismatch: ' + path)


def validate_arrays(batch, expected, shapes):
    if set(batch) != KEYS or set(expected) != KEYS or set(shapes) != KEYS:
        raise ValueError('Exactly five batch keys required')
    result = {}
    for key in sorted(KEYS):
        value = batch[key]
        if value.dtype != np.float32 or list(value.shape) != shapes[key] or not np.isfinite(value).all():
            raise ValueError('Invalid batch array: ' + key)
        result[key] = hashlib.sha256(value.tobytes(order='C')).hexdigest()
        if result[key] != expected[key]:
            raise ValueError('Array checksum mismatch: ' + key)
    return result


def export_batches(audit_path, output, loader=None):
    audit_path, output = Path(audit_path).resolve(), Path(output).resolve()
    audit_digest = file_hash(audit_path)
    audit = json.loads(audit_path.read_text('utf-8'))
    if audit['status'] != 'all_planned_batches_readable_not_training_ready':
        raise ValueError('Complete batch-read audit required')
    sources = dict(audit['source_sha256'])
    sources[str(audit_path)] = audit_digest
    sources[str(Path(__file__).resolve())] = file_hash(Path(__file__))
    verify_sources(sources)
    candidates = audit['candidates']
    if len(candidates) != audit['validated_candidates'] or len({r['origin'] for r in candidates}) != len(candidates):
        raise ValueError('Candidate coverage mismatch')
    representatives = {}
    for row in candidates:
        representatives.setdefault(row['latest'], row)
    if set(representatives) != set(audit['pair_array_sha256']) or len(representatives) != audit['unique_weather_pairs']:
        raise ValueError('Pair coverage mismatch')
    loader = loader or FullHistoryBatchLoader()
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='running', source_sha256=sources, rows=[], candidates=candidates,
                    training_performed=False, operational_availability_verified=False)
    atomic_json(output / 'manifest.json', manifest)
    for pair_id, row in sorted(representatives.items()):
        batch, metadata = loader.load(row['origin'])
        if any(metadata.get(key) != value for key, value in row.items()):
            raise ValueError('Candidate metadata changed')
        expected = audit['pair_array_sha256'][pair_id]
        digests = validate_arrays(batch, expected, audit['batch_shapes'])
        name = 'batch_' + datetime.fromisoformat(pair_id).strftime('%Y%m%dT%H%M%SZ') + '.npz'
        target = output / name
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **batch)
        with np.load(temporary, allow_pickle=False) as saved:
            validate_arrays(saved, expected, audit['batch_shapes'])
        temporary.replace(target)
        manifest['rows'].append(dict(pair_id=pair_id, origin=row['origin'], path=name,
                                     sha256=file_hash(target), batch_array_sha256=digests))
        if len(manifest['rows']) % 100 == 0:
            atomic_json(output / 'manifest.json', manifest)
            print(json.dumps({'exported_pairs': len(manifest['rows']), 'required': len(representatives)}), flush=True)
    loader.verify_unchanged()
    for path, digest in loader.hashes.items():
        if path not in sources or sources[path] != digest:
            raise ValueError('Loader used unverified or changed source: ' + path)
    verify_sources(sources)
    for row in manifest['rows']:
        if file_hash(output / row['path']) != row['sha256']:
            raise ValueError('Export checksum mismatch')
    manifest.update(status='complete', unique_weather_pairs=len(manifest['rows']),
                    candidate_count=len(candidates), completed_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    result = export_batches(HERE / 'full_history/full_batch_read_audit.json',
                            HERE / 'full_history/encoder_batches')
    print(json.dumps({k: result[k] for k in ('status', 'unique_weather_pairs', 'candidate_count')}))
