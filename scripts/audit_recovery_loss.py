"""Read-only training-period likelihood and output-gradient contribution audit."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_head import RecoveryWindows, load_experiment, prediction_arrays
from scripts.benchmark_regional_air_models import hashes, write_json

ROOT = Path(__file__).resolve().parents[1]


def contribution_arrays(logits, target, mask, eligible):
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    observed = np.asarray(mask, dtype=bool) & np.asarray(eligible, dtype=bool)[:, None]
    terms = np.logaddexp(0, logits) - target * logits
    probability = np.exp(-np.logaddexp(0, -logits))
    return {'nll': (terms * observed).sum(axis=1),
            'gradient': (probability - target) * observed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.data = 'data/processed/regional_air_models_full_history'
    args.history, args.horizon, args.validation_year, args.batch_size = 24, 72, 2024, 256
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'artifacts/task01_recovery_head/seed42_v1/recovery_head_state.pt'
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    before, deployed = hashes(ROOT / args.data), hashes(deployment)
    torch.set_num_threads(2)
    panel, city, labels, eligible, partitions = load_experiment(args)
    origins = np.concatenate([partitions[y] for y in range(2014, 2024)])
    dataset = RecoveryWindows(panel, origins, 24, 72, city.aqi.to_numpy(), eligible)
    informative = dataset.eligible & dataset.risk_mask.any(axis=1)
    exact = dataset.hazard.any(axis=1)
    observed = dataset.risk_mask.sum(axis=1)
    group = np.select([~informative, exact & (observed <= 24), exact,
                       ~exact & (observed == 67)],
                      ['excluded', 'exact_1_24h', 'exact_25_67h', 'observed_tail_68h'],
                      default='prefix_censored')
    rows = pd.DataFrame({'timestamp': panel['timestamps'][origins],
                         'event_id': labels.reindex(panel['timestamps'][origins]).event_id.to_numpy(),
                         'group': group, 'observed_bins': observed,
                         'informative': informative})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    model = RecoveryPM25GNN(len(panel['feature_names']), panel['coords'],
                           panel['scaler'].mean[-2:], panel['scaler'].scale[-2:]).to(device)
    report = {'training_years': '2014-2023', 'label_audit': dataset.audit(),
              'checkpoint_sha256': checkpoint_hash, 'snapshots': {},
              'limitations': ['output_logit_gradient_not_parameter_gradient',
                              'snapshot_not_accumulated_optimizer_updates',
                              'censoring_is_not_known_long_duration',
                              'selected_checkpoint_used_2024_validation']}
    for name in ('initial', 'selected_epoch2'):
        if name != 'initial':
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        _, logits = prediction_arrays(model, dataset, args, device)
        values = contribution_arrays(logits.numpy(), dataset.hazard, dataset.risk_mask, dataset.eligible)
        frame = rows.copy()
        frame['nll'] = values['nll']
        frame['gradient_l1'] = np.abs(values['gradient']).sum(axis=1)
        # Positive BCE derivatives reduce hazard under gradient descent; negatives increase it.
        frame['reduce_hazard_gradient'] = np.maximum(values['gradient'], 0).sum(axis=1)
        frame['increase_hazard_gradient'] = np.maximum(-values['gradient'], 0).sum(axis=1)
        frame.to_parquet(output / f'{name}_origins.parquet', index=False)
        kept = frame.loc[informative]
        columns = ['nll', 'gradient_l1', 'reduce_hazard_gradient', 'increase_hazard_gradient']
        grouped = kept.groupby('group')[columns].sum()
        grouped['rows'] = kept.groupby('group').size()
        for col in ('nll', 'gradient_l1'):
            grouped[f'{col}_share'] = grouped[col] / kept[col].sum()
        events = kept.dropna(subset=['event_id']).groupby('event_id')[columns].sum()
        events['rows'] = kept.dropna(subset=['event_id']).groupby('event_id').size()
        events.sort_values('nll', ascending=False).to_csv(output / f'{name}_events.csv')
        bins = pd.DataFrame({'bin': np.arange(1, 68),
                             'risk_rows': (dataset.risk_mask & informative[:, None]).sum(axis=0),
                             'gradient_signed': values['gradient'].sum(axis=0),
                             'gradient_l1': np.abs(values['gradient']).sum(axis=0)})
        bins.to_csv(output / f'{name}_bins.csv', index=False)
        report['snapshots'][name] = {
            'groups': grouped.to_dict(orient='index'), 'nll_per_informative_origin': kept.nll.mean(),
            'event_id_missing_rows': int(kept.event_id.isna().sum()),
            'top5_event_nll_share_all_informative': events.nlargest(5, 'nll').nll.sum() / kept.nll.sum(),
            'known_events': len(events)}
        print({'snapshot': name, **report['snapshots'][name]}, flush=True)
    assert hashes(ROOT / args.data) == before
    assert hashes(deployment) == deployed
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    report.update(inputs_unchanged=True, deployment_unchanged=True, checkpoint_unchanged=True)
    write_json(output / 'report.json', report)


if __name__ == '__main__':
    main()
