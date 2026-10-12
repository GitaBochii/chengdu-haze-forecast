"""Frozen full-history regional encoder extraction, separate from training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from encoder_loading import build_encoder, load_encoder_state
from spatial_encoder import encode_spatial

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_feature(value, identity):
    feature = value.get('embedding')
    return (all(value.get(k) == v for k, v in identity.items())
            and isinstance(feature, torch.Tensor) and feature.shape == (1, 2560)
            and feature.is_floating_point() and bool(torch.isfinite(feature).all())
            and bool(feature.std() > 0))


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), 'utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError('Positive extraction limit required')
    folder = HERE / 'full_history/encoder_batches'
    source = folder / 'manifest.json'
    manifest = json.loads(source.read_text('utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Complete exported input manifest required')
    for path, digest in manifest['source_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Input provenance changed: ' + path)
    dependencies = {str(HERE / name): sha(HERE / name) for name in (
        'official_batch.pt', 'real_sample/config.yaml', 'encoder_loading.py', 'spatial_encoder.py',
        'storage_loader.py', 'tensor_descriptors.json', 'model_storage_inventory.json',
        'vendor/PrithviWxC/model.py', 'encode_full_history.py')}
    dependencies[str(source)] = sha(source)
    identity_base = {'dependency_sha256': dependencies, 'torch_version': str(torch.__version__),
                     'pooling': 'mean_global_and_local_tokens', 'precision': 'bfloat16_parameters_fp32_scalers'}
    output = HERE / 'full_history/encoder_embeddings'
    output.mkdir(exist_ok=True)
    rows, produced, model, load_report = [], 0, None, None
    report = {**identity_base, 'status': 'running', 'rows': rows, 'expected_pairs': len(manifest['rows']),
              'accuracy_evaluated': False, 'training_performed': False}
    write_report(output / 'manifest.json', report)
    for item in manifest['rows']:
        name = Path(item['path']).stem + '.pt'
        path = output / name
        batch_path = (folder / item['path']).resolve()
        if not batch_path.is_relative_to(folder.resolve()) or sha(batch_path) != item['sha256']:
            raise ValueError('Exported batch checksum mismatch')
        identity = {**identity_base, 'pair_id': item['pair_id'], 'batch_sha256': item['sha256']}
        row = {'pair_id': item['pair_id'], 'path': name, 'batch_sha256': item['sha256']}
        if path.exists() and valid_feature(torch.load(path, weights_only=True, map_location='cpu'), identity):
            rows.append({**row, 'status': 'ready', 'sha256': sha(path)})
            continue
        if args.limit is not None and produced >= args.limit:
            rows.append({**row, 'status': 'deferred'})
            continue
        with np.load(batch_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {'x', 'static', 'climate', 'input_time', 'lead_time'}:
                raise ValueError('Unexpected inference fields')
            batch = {}
            for key in arrays.files:
                value = arrays[key]
                if hashlib.sha256(value.tobytes(order='C')).hexdigest() != item['batch_array_sha256'][key]:
                    raise ValueError('Batch array hash mismatch')
                batch[key] = torch.from_numpy(value.copy())
        if model is None:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for frozen extraction')
            values = torch.load(HERE / 'official_batch.pt', weights_only=True, map_location='cpu')
            config = yaml.safe_load((HERE / 'real_sample/config.yaml').read_text())['params']
            model = build_encoder(config, values, (30, 32))
            free, _ = torch.cuda.mem_get_info()
            if free < sum(p.numel()*2 for p in model.parameters()) + 1024**3:
                raise RuntimeError('Insufficient free GPU memory')
            descriptions = json.loads((HERE / 'tensor_descriptors.json').read_text())['model_state']
            weights = json.loads((HERE / 'model_storage_inventory.json').read_text())
            records = {r['key']: r for r in weights['records']}
            load_report = load_encoder_state(model, descriptions, records, HERE / 'weights/model_storage', 'cuda', torch.bfloat16)
        started = time.perf_counter()
        batch = {k: v.to('cuda') for k, v in batch.items()}
        torch.manual_seed(24)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            tokens = encode_spatial(model, batch)
            feature = tokens.float().mean(dim=(1, 2)).cpu()
        value = {**identity, 'embedding': feature}
        if not valid_feature(value, identity):
            raise ValueError('Invalid encoded feature')
        temporary = path.with_suffix('.tmp')
        torch.save(value, temporary)
        temporary.replace(path)
        produced += 1
        rows.append({**row, 'status': 'ready', 'sha256': sha(path), 'seconds': time.perf_counter()-started})
        report.update(load_report=load_report, ready_pairs=sum(r['status']=='ready' for r in rows))
        write_report(output / 'manifest.json', report)
        print(json.dumps({'ready': report['ready_pairs'], 'expected': len(manifest['rows']), 'seconds': rows[-1]['seconds']}), flush=True)
    if any(sha(Path(path)) != digest for path, digest in dependencies.items()):
        raise ValueError('Encoder dependency changed')
    count = sum(row['status'] == 'ready' for row in rows)
    report.update(status='complete' if count == len(manifest['rows']) else 'partial_limit', ready_pairs=count)
    write_report(output / 'manifest.json', report)
    print(json.dumps({'status': report['status'], 'ready_pairs': count}), flush=True)


if __name__ == '__main__':
    main()
