"""Frozen full-history offline comparison; no deployment or test-driven selection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiments.prithvi_wxc_probe_20260924.run_pilot_training import (
    make_matched_models, forward, sha, verify_source_hashes, SOURCE_HASHES)
from experiments.prithvi_wxc_probe_20260924.pilot_training_data import prepare_pollution_histories
from experiments.prithvi_wxc_probe_20260924.pilot_training_utils import (
    fit_spatial_pca, transform_spatial, error_summary)
from experiments.prithvi_wxc_probe_20260924.interval_loss import interval_survival_nll
from haze_forecast.recovery_survival import hazard_median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARTITIONS = ('train', 'dev', 'retrospective_test', 'recent_audit')


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, data):
    path.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False), 'utf-8')


def check_groups(frame):
    if frame.timestamp.duplicated().any() or not set(frame.partition).issubset(PARTITIONS):
        raise ValueError('Unique origins and fixed partitions required')
    for key in ('episode_id', 'continuity_block', *(['pair_id'] if 'pair_id' in frame.columns else [])):
        if frame.groupby(key).partition.nunique().gt(1).any():
            raise ValueError('Cross-partition ' + key)


def select_population(frame):
    keep = (~frame.cross_partition_block & frame.label_mature_within_partition
            & frame.history_within_partition & ~frame.year_start_boundary_uncertain)
    selected, excluded = frame.loc[keep].copy(), frame.loc[~keep].copy()
    check_groups(selected)
    return selected, excluded


def unique_train_pair_mask(partitions, pairs):
    seen, keep = set(), []
    for partition, pair in zip(partitions, pairs, strict=True):
        use = partition == 'train' and pair not in seen
        keep.append(use)
        if use:
            seen.add(pair)
    return torch.tensor(keep, dtype=torch.bool)


def sample_blocks(blocks, rng, size):
    return np.asarray([rng.choice(blocks[k]) for k in rng.integers(len(blocks), size=size)])


def merge_hashes(target, additions):
    for path, digest in additions.items():
        if path in target and target[path] != digest:
            raise ValueError('Conflicting provenance: ' + path)
        target[path] = digest


def load_population():
    base = HERE / 'full_history'
    candidate = base / 'support_aligned_plan/candidate_hours.parquet'
    batch_path = base / 'encoder_batches/manifest.json'
    embedding_path = base / 'encoder_embeddings/manifest.json'
    batches = json.loads(batch_path.read_text('utf-8'))
    report = json.loads(embedding_path.read_text('utf-8'))
    if (batches['status'] != 'complete' or report['status'] != 'complete'
            or len(batches['rows']) != 1639 or len(report['rows']) != 1639):
        raise ValueError('Complete 1639-pair encoded population required')
    dependencies = dict(batches['source_sha256'])
    merge_hashes(dependencies, report['dependency_sha256'])
    merge_hashes(dependencies, {str(path): sha(path) for path in (candidate, batch_path, embedding_path)})
    for path, expected in dependencies.items():
        if sha(Path(path)) != expected:
            raise ValueError('Changed input dependency: ' + path)
    source_rows = {r['pair_id']: r for r in batches['rows']}
    features = {}
    for row in report['rows']:
        if row['status'] != 'ready' or row['pair_id'] in features:
            raise ValueError('Non-ready or duplicate embedding')
        path = (embedding_path.parent / row['path']).resolve()
        if not path.is_relative_to(embedding_path.parent.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Embedding checksum mismatch')
        stored = torch.load(path, weights_only=True, map_location='cpu')
        for key in ('dependency_sha256', 'torch_version', 'pooling', 'precision'):
            if stored[key] != report[key]:
                raise ValueError('Embedding provenance mismatch')
        pair = row['pair_id']
        if stored['pair_id'] != pair or stored['batch_sha256'] != source_rows[pair]['sha256']:
            raise ValueError('Embedding batch mismatch')
        feature = stored['embedding']
        if feature.shape != (1, 2560) or not torch.isfinite(feature).all() or feature.std() <= 0:
            raise ValueError('Invalid embedding')
        features[pair] = feature.float()
        dependencies[str(path)] = row['sha256']
    if set(features) != set(source_rows):
        raise ValueError('Incomplete pair coverage')
    frame = pd.read_parquet(candidate)
    metadata = {pd.Timestamp(r['origin']): r for r in batches['candidates']}
    if len(frame) != 8080 or len(metadata) != 8080 or set(frame.timestamp) != set(metadata):
        raise ValueError('Candidate coverage changed')
    frame['year_start_boundary_uncertain'] = [metadata[t]['year_start_boundary_uncertain'] for t in frame.timestamp]
    frame['pair_id'] = [metadata[t]['latest'] for t in frame.timestamp]
    if any(metadata[t]['partition'] != part for t, part in zip(frame.timestamp, frame.partition)):
        raise ValueError('Candidate partition changed')
    selected, excluded = select_population(frame)
    selected = selected.reset_index(drop=True)
    embeddings = torch.cat([features[p] for p in selected.pair_id])
    return selected, excluded, embeddings, dependencies


def predict(model, arm, data, z, device):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z), 64):
            stop = start + 64
            outputs.append(forward(model, arm, data['x'][start:stop].to(device),
                data['active'][start:stop].to(device), z[start:stop].to(device)).cpu())
    return torch.cat(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    frame, excluded, embeddings, dependencies = load_population()
    source = ROOT / 'data/processed/regional_air_models_full_history'
    source_hashes = verify_source_hashes(source, SOURCE_HASHES)
    counts = frame.groupby('partition').size().to_dict()
    if set(counts) != set(PARTITIONS):
        raise ValueError('All fixed partitions required')
    if args.preflight_only:
        print(json.dumps({'status': 'preflight_passed', 'selected': counts, 'excluded': len(excluded)}))
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = frame.to_dict('records')
    prepared_rows = [dict(r, partition=r['partition'] if r['partition'] in ('train','dev') else 'test') for r in rows]
    data = prepare_pollution_histories(pd.read_parquet(source/'station_hourly.parquet'),
                                     pd.read_parquet(source/'weather_proxy.parquet'), prepared_rows)
    pca_mask = unique_train_pair_mask(frame.partition, frame.pair_id)
    pca = fit_spatial_pca(embeddings, pca_mask)
    z = transform_spatial(embeddings, pca)
    train = frame.partition.eq('train')
    blocks = [group.index.to_numpy() for _, group in frame.loc[train].groupby('continuity_block')]
    low = torch.tensor(frame.lower_bound_hours.to_numpy(), dtype=torch.float32)
    high = torch.tensor(frame.upper_bound_hours.to_numpy(), dtype=torch.float32)
    output = HERE/'full_history/training'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output/'selected.parquet', index=False)
    excluded.to_parquet(output/'excluded.parquet', index=False)
    code = ['run_full_history_training.py', 'FULL_HISTORY_TRAINING_PROTOCOL.md', 'run_pilot_training.py',
            'pilot_training_data.py', 'pilot_training_utils.py', 'interval_loss.py', 'origin_only.py', 'fusion_head.py']
    provenance = {'input_sha256': dependencies, 'source_sha256': source_hashes,
        'code_sha256': {name: sha(HERE/name) for name in code}, 'torch_version': str(torch.__version__),
        'device': str(device), 'updates': 500, 'batch_size': 64, 'pca_unique_train_pairs': int(pca_mask.sum())}
    shared = ('src/haze_forecast/regional_recovery_network.py', 'src/haze_forecast/regional_air_networks.py',
        'src/haze_forecast/regional_air_experiment.py', 'src/haze_forecast/regional_air_scoring.py',
        'src/haze_forecast/recovery_survival.py',
        'experiments/regional_air_models/vendor/PM25-GNN/model/cells.py',
        'experiments/regional_air_models/vendor/PM25-GNN/model/PM25_GNN.py')
    provenance['shared_code_sha256'] = {name:sha(ROOT/name) for name in shared}
    save_json(output/'protocol.json', provenance)
    torch.save({'pca':pca, 'z':z, 'scaler_mean':torch.from_numpy(data['scaler_mean']),
        'scaler_scale':torch.from_numpy(data['scaler_scale']), 'nodes':data['nodes'],
        'training_timestamps':data['training_timestamps'], 'provenance':provenance}, output/'preprocessing.pt')
    runs = []
    for seed in (24, 42, 2026):
        models = make_matched_models(seed, data['coords'], data['scaler_mean'], data['scaler_scale'])
        for arm, model in models.items():
            torch.manual_seed(seed)
            model.to(device).train()
            rng = np.random.default_rng(seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            losses = []
            for step in range(500):
                ix = sample_blocks(blocks, rng, 64)
                optimizer.zero_grad(set_to_none=True)
                logits = forward(model, arm, data['x'][ix].to(device), data['active'][ix].to(device), z[ix].to(device))
                loss = interval_survival_nll(logits, low[ix].to(device), high[ix].to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step+1) % 100 == 0:
                    print(json.dumps({'seed':seed, 'arm':arm, 'step':step+1, 'loss':losses[-1]}), flush=True)
            logits = predict(model, arm, data, z, device)
            predictions = hazard_median(logits).numpy()
            checkpoint = output/f'{seed}_{arm}.pt'
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'seed':seed, 'arm':arm, 'provenance':provenance}, checkpoint)
            model.cpu()
            restored = make_matched_models(seed, data['coords'], data['scaler_mean'], data['scaler_scale'])[arm]
            restored.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict'], strict=True)
            restored.to(device)
            torch.testing.assert_close(predict(restored, arm, data, z, device), logits, rtol=0, atol=0)
            checks = {'checkpoint_exact_logit_roundtrip': True}
            if arm != 'baseline':
                with torch.no_grad():
                    x, active = data['x'][:64].to(device), data['active'][:64].to(device)
                    missing = restored(x, active, torch.full_like(z[:64].to(device), float('nan')),
                        torch.zeros(len(x), dtype=torch.bool, device=device))[1]
                    torch.testing.assert_close(missing, restored.baseline(x, active)[1], rtol=0, atol=0)
                    checks['missing_exact_baseline_fallback'] = True
                    if arm == 'spatial_fusion':
                        change = (forward(restored, arm, x, active, z[:64].to(device))
                                  - forward(restored, arm, x, active, -z[:64].to(device))).abs().max()
                        checks['feature_intervention_max_logit_change'] = float(change)
                        if change <= 0:
                            raise ValueError('Spatial features have no effect')
            restored.cpu()
            prediction_frame = frame.copy()
            prediction_frame['predicted_capped_median_hours'] = predictions
            prediction_frame.to_parquet(output/f'{seed}_{arm}_predictions.parquet', index=False)
            summaries = {part:error_summary([r for r in rows if r['partition']==part],
                predictions[frame.partition.eq(part).to_numpy()]) for part in PARTITIONS}
            record = {'seed':seed, 'arm':arm, 'summary':summaries, 'checks':checks,
                'training_loss':losses, 'checkpoint_sha256':sha(checkpoint)}
            save_json(output/f'{seed}_{arm}.json', record)
            runs.append(record)
            print(json.dumps({'seed':seed, 'arm':arm, 'summary':summaries}), flush=True)
    comparisons = []
    for seed in (24,42,2026):
        by_arm = {r['arm']:r for r in runs if r['seed']==seed}
        for part in PARTITIONS:
            for control in ('baseline','capacity_control'):
                fusion, base = by_arm['spatial_fusion']['summary'][part], by_arm[control]['summary'][part]
                comparisons.append({'seed':seed, 'partition':part, 'control':control,
                    'fusion_minus_control':{key:fusion[key]-base[key] for key in (
                        'exact_row_mae_capped_hours','row_error_lower_hours','row_error_upper_hours',
                        'block_error_lower_hours','block_error_upper_hours')
                        if fusion[key] is not None and base[key] is not None}})
    for path, digest in dependencies.items():
        if sha(Path(path)) != digest:
            raise ValueError('Input changed during experiment')
    verify_source_hashes(source, SOURCE_HASHES)
    if any(sha(HERE/name) != digest for name, digest in provenance['code_sha256'].items()):
        raise ValueError('Experiment code changed during training')
    if any(sha(ROOT/name) != digest for name, digest in provenance['shared_code_sha256'].items()):
        raise ValueError('Shared model code changed during training')
    save_json(output/'report.json', {'status':'completed_offline_fixed_protocol', 'runs':runs,
        'comparisons':comparisons, 'provenance':provenance, 'counts':counts, 'excluded_count':len(excluded),
        'elapsed_seconds':time.perf_counter()-started, 'deployed':False, 'blind_test':False,
        'limitations':['Offline reanalysis, not real-time availability.',
                       'Year-start uncertain rows excluded before training and evaluation.',
                       'New purged population and fixed protocol; do not compare directly to prior MAE.',
                       '2025 previously inspected; 2026 has few independent blocks.']})
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
