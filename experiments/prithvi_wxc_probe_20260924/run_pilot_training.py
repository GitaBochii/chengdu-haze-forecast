"""Fixed-budget offline pilot; never substitutes missing final-audit dates."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.prithvi_wxc_probe_20260924.origin_only import RecoveryOriginOnly
from haze_forecast.recovery_survival import hazard_median
from experiments.prithvi_wxc_probe_20260924.fusion_head import SpatialRecoveryFusion
from experiments.prithvi_wxc_probe_20260924.interval_loss import interval_survival_nll
from experiments.prithvi_wxc_probe_20260924.pilot_training_utils import (
    fit_spatial_pca, transform_spatial, block_weights, error_summary)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISION = 'e2f62b3e19890f1a5e225bb123acb4b33df85ef8'
CANDIDATE_SHA = '036c3760cbc27eed1bdb15d887c836b31b19672a3c71481a4e19a0596750da81'
SOURCE_HASHES = {
    'station_hourly.parquet':'45ce6dbcf96ce804253e0756361d7303fae0d90f8b357096911f21299a6a8dc1',
    'weather_proxy.parquet':'ef074edd8c17d53cf0144d100f2350659b56870a145d427e4fbdd1048eb8a35a',
    'city_hourly.parquet':'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_hashes(folder, expected):
    actual = {name:sha(folder/name) for name in expected}
    if actual != expected:
        raise ValueError('Frozen pollution source changed; new experiment version required')
    return actual


def require_complete_rows(rows, frozen):
    result = {r['timestamp']: r for r in rows}
    if (len(rows) != len(frozen) or len(result) != len(rows)
            or set(result) != {r['timestamp'] for r in frozen}
            or any(result[r['timestamp']]['partition'] != r['partition'] for r in frozen)
            or any(r['status'] != 'ready' for r in rows)):
        raise ValueError('Every frozen origin must be present exactly once, ready and in its original split')
    return result


def make_matched_models(seed, coords, mean, scale):
    torch.manual_seed(seed)
    baseline = RecoveryOriginOnly(17, coords, torch.as_tensor(mean[-2:], dtype=torch.float32),
                              torch.as_tensor(scale[-2:], dtype=torch.float32))
    control = SpatialRecoveryFusion(copy.deepcopy(baseline), 4, hidden_dim=16, freeze_baseline=False)
    return dict(baseline=baseline, capacity_control=control, spatial_fusion=copy.deepcopy(control))


def load_embeddings():
    manifest = HERE / 'pilot_candidate_audit.json'
    if sha(manifest) != CANDIDATE_SHA:
        raise ValueError('Frozen candidate manifest changed')
    frozen = json.loads(manifest.read_text('utf-8'))['rows']
    report = json.loads((HERE / 'pilot_embedding_report.json').read_text('utf-8'))
    if report['candidate_manifest_sha256'] != CANDIDATE_SHA or report['revision'] != REVISION:
        raise ValueError('Embedding provenance mismatch')
    by_time = require_complete_rows(report['rows'], frozen)
    for path, expected in report['dependency_sha256'].items():
        full = (HERE / path).resolve()
        if not full.is_relative_to(HERE) or sha(full) != expected:
            raise ValueError('Encoder dependency changed')
    if sha(HERE / 'model_storage_inventory.json') != report['weights_inventory_sha256']:
        raise ValueError('Weights inventory changed')
    batches = json.loads((HERE / 'pilot_batch_report.json').read_text('utf-8'))
    batch_rows = require_complete_rows(batches['rows'], frozen)
    features = []
    for candidate in frozen:
        row = by_time[candidate['timestamp']]
        path = (HERE / row['embedding_path']).resolve()
        if not path.is_relative_to(HERE) or sha(path) != row['embedding_sha256']:
            raise ValueError('Embedding file checksum mismatch')
        stored = torch.load(path, weights_only=True, map_location='cpu')
        for key in ('candidate_manifest_sha256', 'weights_inventory_sha256', 'revision',
                    'dependency_sha256', 'torch_version', 'pooling'):
            if stored.get(key) != report[key]:
                raise ValueError('Stored embedding provenance mismatch')
        if any(stored.get(k) != candidate[k] for k in ('timestamp', 'date', 'partition')):
            raise ValueError('Stored embedding identity mismatch')
        source = batch_rows[candidate['timestamp']]
        batch_path = (HERE / source['batch_path']).resolve()
        if (not batch_path.is_relative_to(HERE) or sha(batch_path) != source['batch_sha256']
                or stored['batch_sha256'] != source['batch_sha256']):
            raise ValueError('Batch checksum mismatch')
        feature = stored['embedding']
        if feature.shape != (1, 2560) or not torch.isfinite(feature).all() or feature.std() <= 0:
            raise ValueError('Invalid spatial feature')
        features.append(feature.float())
    for group in ('episode_id', 'continuity_block'):
        groups = {part: {r[group] for r in frozen if r['partition'] == part}
                  for part in ('train', 'dev', 'test')}
        if any(groups[a] & groups[b] for a, b in [('train','dev'),('train','test'),('dev','test')]):
            raise ValueError('Cross-partition group leakage')
    return frozen, torch.cat(features)


def forward(model, arm, x, active, z):
    if arm == 'baseline':
        return model(x, active)[1]
    features = z if arm == 'spatial_fusion' else torch.zeros_like(z)
    return model(x, active, features, torch.ones(len(x), dtype=torch.bool, device=x.device))[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    rows, embeddings = load_embeddings()
    if args.preflight_only:
        print(json.dumps(dict(status='complete_inputs_verified', rows=len(rows))))
        return
    from experiments.prithvi_wxc_probe_20260924.pilot_training_data import prepare_pollution_histories
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    source = ROOT / 'data/processed/regional_air_models_full_history'
    source_hashes = verify_source_hashes(source, SOURCE_HASHES)
    paths = [source / name for name in ('station_hourly.parquet', 'weather_proxy.parquet')]
    data = prepare_pollution_histories(*(pd.read_parquet(p) for p in paths), rows)
    train = torch.tensor([r['partition'] == 'train' for r in rows])
    pca = fit_spatial_pca(embeddings, train)
    z = transform_spatial(embeddings, pca)
    weights = block_weights([r['continuity_block'] for r in rows if r['partition'] == 'train'])
    low = torch.tensor([r['lower_bound_hours'] for r in rows])
    high = torch.tensor([r['upper_bound_hours'] for r in rows])
    output = HERE / 'pilot_training' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(candidate_sha256=CANDIDATE_SHA, recipe_sha256=sha(HERE/'TRAINING_RECIPE.md'),
                      embedding_report_sha256=sha(HERE/'pilot_embedding_report.json'),
                      source_sha256=source_hashes, torch_version=str(torch.__version__),
                      code_sha256={name:sha(HERE/name) for name in
                                   ('run_pilot_training.py','pilot_training_data.py',
                                    'pilot_training_utils.py','interval_loss.py','fusion_head.py', 'origin_only.py')})
    provenance['shared_code_sha256'] = {name:sha(ROOT/name) for name in
        ('src/haze_forecast/regional_recovery_network.py','src/haze_forecast/regional_air_networks.py',
         'src/haze_forecast/regional_air_experiment.py','src/haze_forecast/regional_air_scoring.py',
         'src/haze_forecast/recovery_survival.py',
         'experiments/regional_air_models/vendor/PM25-GNN/model/cells.py',
         'experiments/regional_air_models/vendor/PM25-GNN/model/PM25_GNN.py')}
    torch.save(dict(pca=pca, embeddings=embeddings, spatial_features=z, candidates=rows,
                   **{k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                              for k,v in data.items()}, provenance=provenance), output/'preprocessing.pt')
    device = torch.device('cpu')
    print(f'Run directory: {output}; device: {device}', flush=True)
    x, active, z = data['x'].to(device), data['active'].to(device), z.to(device)
    train, low, high, weights = train.to(device), low.to(device), high.to(device), weights.to(device)
    records = []
    for seed in (24, 42, 2026):
        models = make_matched_models(seed, data['coords'], data['scaler_mean'], data['scaler_scale'])
        for arm, model in models.items():
            model.to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            losses = []
            for step in range(100):
                optimizer.zero_grad(set_to_none=True)
                logits = forward(model, arm, x[train], active[train], z[train])
                loss = (interval_survival_nll(logits, low[train], high[train], reduction='none') * weights).sum()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step+1) % 25 == 0:
                    print(f'{seed} {arm} step {step+1}/100 loss {losses[-1]:.6f}', flush=True)
            model.eval()
            with torch.no_grad():
                logits = forward(model, arm, x, active, z)
                median = hazard_median(logits).cpu().tolist()
                checks = {}
                if arm != 'baseline':
                    missing = model(x, active, torch.full_like(z, float('nan')),
                                    torch.zeros(len(x), dtype=torch.bool, device=device))[1]
                    base = model.baseline(x, active)[1]
                    torch.testing.assert_close(missing, base, rtol=0, atol=0)
                    checks['missing_exact_baseline_fallback'] = True
                if arm == 'spatial_fusion':
                    altered = forward(model, arm, x, active, -z)
                    checks['feature_intervention_max_logit_change'] = float((logits-altered).abs().max())
                    if checks['feature_intervention_max_logit_change'] <= 0:
                        raise ValueError('Spatial feature intervention had no effect')
            prediction = [dict(**r, predicted_capped_median_hours=p) for r,p in zip(rows, median)]
            summary = {part:error_summary([r for r in rows if r['partition']==part],
                        [p for r,p in zip(rows,median) if r['partition']==part])
                       for part in ('train','dev','test')}
            checkpoint = output/f'{seed}_{arm}.pt'
            torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                            seed=seed, arm=arm, updates=100, provenance=provenance), checkpoint)
            restored = make_matched_models(seed, data['coords'], data['scaler_mean'], data['scaler_scale'])[arm]
            restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')['state_dict'], strict=True)
            restored.to(device).eval()
            with torch.no_grad():
                reproduced = forward(restored, arm, x, active, z)
            torch.testing.assert_close(reproduced, logits, rtol=0, atol=0)
            checks['checkpoint_exact_logit_roundtrip'] = True
            del restored
            record = dict(seed=seed, arm=arm, training_nll=losses, summary=summary,
                          checks=checks, predictions=prediction, checkpoint_sha256=sha(checkpoint))
            (output/f'{seed}_{arm}.json').write_text(json.dumps(record, indent=2, allow_nan=False), 'utf-8')
            records.append(record)
            model.cpu()
    report = dict(status='offline_multisample_pilot_completed', provenance=provenance, device=str(device), runs=records,
                  deployed=False, real_time_validated=False, generalization_proven=False)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','provenance')}, indent=2))


if __name__ == '__main__':
    main()
