"""Frozen-checkpoint shared-task gradient audit; never performs optimizer updates."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from haze_forecast.recovery_survival import masked_survival_loss
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_head import RecoveryWindows, load_experiment
from scripts.benchmark_regional_air_models import hashes, masked_loss, write_json

ROOT = Path(__file__).resolve().parents[1]


def gradient_metrics(params, survival, auxiliary, weight):
    shared_s, shared_a, combined = [], [], []
    for p, s, a in zip(params, survival, auxiliary, strict=True):
        if s is not None and a is not None:
            shared_s.append(s.detach().flatten().double())
            shared_a.append(a.detach().flatten().double())
        s = torch.zeros_like(p) if s is None else s
        a = torch.zeros_like(p) if a is None else a
        combined.append((s + weight * a).detach().flatten().double())
    global_norm = float(torch.cat(combined).norm())
    result = {'combined_global_norm': global_norm,
              'clip_multiplier': min(1., 1. / (global_norm + 1e-6)),
              'shared_cosine': None, 'weighted_aux_shared_ratio': None,
              'survival_shared_norm': 0., 'auxiliary_shared_norm': 0.,
              'shared_parameter_elements': 0}
    if shared_s:
        s, a = torch.cat(shared_s), torch.cat(shared_a)
        sn, an = float(s.norm()), float(a.norm())
        result.update(survival_shared_norm=sn, auxiliary_shared_norm=an,
                      shared_parameter_elements=s.numel())
        if sn > 0:
            result['weighted_aux_shared_ratio'] = weight * an / sn
        if sn > 0 and an > 0:
            result['shared_cosine'] = float(torch.dot(s, a) / (sn * an))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.data = 'data/processed/regional_air_models_full_history'
    args.history, args.horizon, args.validation_year = 24, 72, 2024
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'artifacts/task01_recovery_head/seed42_v1/recovery_head_state.pt'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    before, deployed = hashes(ROOT / args.data), hashes(deployment)
    torch.set_num_threads(2)
    panel, city, _, eligible, partitions = load_experiment(args)
    origins = np.concatenate([partitions[y] for y in range(2014, 2024)])
    dataset = RecoveryWindows(panel, origins, 24, 72, city.aqi.to_numpy(), eligible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    model = RecoveryPM25GNN(len(panel['feature_names']), panel['coords'],
                           panel['scaler'].mean[-2:], panel['scaler'].scale[-2:]).to(device)
    # One fixed permutation is reused at both snapshots, including the final small batch.
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    pd.DataFrame({'order': np.arange(len(order)), 'origin': origins[order],
                  'timestamp': panel['timestamps'][origins[order]]}).to_parquet(
                      output / 'batch_order.parquet', index=False)
    report = {'training_years': '2014-2023', 'batch_size': 256, 'auxiliary_weight': .1,
              'checkpoint_sha256': checkpoint_hash, 'snapshots': {},
              'limitations': ['frozen_snapshots_not_optimizer_trajectory',
                              'gradient_cosine_not_causal_generalization_evidence',
                              'checkpoint_selected_on_2024', 'fixed_audit_permutation_not_replayed_training']}
    started = time.monotonic()
    for name in ('initial', 'selected_epoch2'):
        if name != 'initial':
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.train()
        params = tuple(model.parameters())
        rows = []
        loader = DataLoader(dataset, batch_size=256, sampler=order)
        for index, batch in enumerate(loader):
            x, y, mask, active, target, risk, valid = [v.to(device) for v in batch]
            curves, logits = model(x, active)
            loss_s = masked_survival_loss(logits, target, risk, valid)
            loss_a = masked_loss(curves, y, mask)
            gs = (torch.autograd.grad(loss_s, params, retain_graph=True, allow_unused=True)
                  if loss_s is not None else (None,) * len(params))
            ga = (torch.autograd.grad(loss_a, params, allow_unused=True)
                  if loss_a is not None else (None,) * len(params))
            rows.append({'batch': index, 'origins': len(x),
                         'informative_origins': int((valid & risk.any(dim=1)).sum()),
                         'concentration_elements': int(mask.sum()),
                         **gradient_metrics(params, gs, ga, .1)})
        for key, value in model.state_dict().items():
            assert torch.equal(value, frozen[key]), f'mutated model state: {key}'
        frame = pd.DataFrame(rows)
        frame.to_csv(output / f'{name}_batches.csv', index=False)
        report['snapshots'][name] = {
            'batches': len(frame), 'origins': int(frame.origins.sum()),
            'informative_origins': int(frame.informative_origins.sum()),
            'zero_survival_batches': int(frame.informative_origins.eq(0).sum()),
            'negative_cosine_batches': int(frame.shared_cosine.lt(0).sum()),
            'defined_cosine_batches': int(frame.shared_cosine.notna().sum()),
            'clipped_batches': int(frame.clip_multiplier.lt(1).sum()),
            'summary': frame[['informative_origins', 'shared_cosine', 'weighted_aux_shared_ratio',
                              'survival_shared_norm', 'auxiliary_shared_norm',
                              'clip_multiplier']].describe().to_dict()}
        print({'snapshot': name, **report['snapshots'][name]}, flush=True)
    assert hashes(ROOT / args.data) == before
    assert hashes(deployment) == deployed
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    report.update(inputs_unchanged=True, deployment_unchanged=True, checkpoint_unchanged=True,
                  elapsed_seconds=time.monotonic() - started)
    write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()
