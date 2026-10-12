"""Nested chronological comparison of two checkpoint rules on one training trajectory."""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from haze_forecast.recovery_selection import development_gate, selection_keys, selection_metrics
from haze_forecast.recovery_survival import hazard_distribution, hazard_median, masked_survival_loss
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_head import (
    RecoveryWindows,
    duration_records,
    load_experiment,
    prediction_arrays,
)
from scripts.benchmark_regional_air_models import hashes, masked_loss, write_json

ROOT = Path(__file__).resolve().parents[1]
RULES = ('legacy', 'balanced')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen(output, frozen):
    for name, expected in frozen.items():
        if file_hash(output / name) != expected:
            raise RuntimeError(f'frozen artifact changed: {name}')


def train_selection_pair(model, training, validation, panel, labels, city_aqi, args, output, device):
    """Both selectors observe every epoch; ties retain the first checkpoint."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=generator)
    best = {rule: None for rule in RULES}
    selections, trace = {}, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        nll_total, nll_count, aux_total, aux_count = 0., 0, 0., 0
        for batch in loader:
            x, y, mask, active, target, risk_mask, eligible = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            curves, logits = model(x, active)
            survival = masked_survival_loss(logits, target, risk_mask, eligible)
            auxiliary = masked_loss(curves, y, mask)
            terms = []
            if survival is not None:
                terms.append(survival)
                count = int((eligible & risk_mask.any(dim=1)).sum())
                nll_total += float(survival.detach()) * count
                nll_count += count
            if auxiliary is not None:
                terms.append(args.auxiliary_weight * auxiliary)
                count = int(mask.sum())
                aux_total += float(auxiliary.detach()) * count
                aux_count += count
            if not terms:
                continue
            loss = sum(terms)
            if not torch.isfinite(loss):
                raise ValueError('non-finite recovery training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if not nll_count:
            raise ValueError('no informative survival training observations')
        _, logits = prediction_arrays(model, validation, args, device)
        records = duration_records(validation.origins, panel, labels, city_aqi,
                                   hazard_median(logits).numpy(), args.horizon)
        metrics = selection_metrics(records)
        keys = selection_keys(metrics)
        row = {'epoch': epoch, 'training_survival_nll': nll_total / nll_count,
               'training_concentration_l1': aux_total / aux_count if aux_count else None,
               'validation_year': args.validation_year, 'validation': metrics,
               'selection_keys': keys, 'elapsed_seconds': time.monotonic() - started}
        trace.append(row)
        write_json(output / 'training_trace.json', trace)
        for rule in RULES:
            key = tuple(keys[rule])
            if best[rule] is None or key < best[rule]:
                best[rule] = key
                state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                torch.save(state, output / f'{rule}_state.pt')
                selections[rule] = {'epoch': epoch, 'validation_year': args.validation_year,
                                    'validation_key': key, 'metrics': metrics, 'rule': rule,
                                    'seed': args.seed}
        print(row, flush=True)
    for rule in RULES:
        selections[rule]['candidate_epochs'] = list(range(1, args.epochs + 1))
        write_json(output / f'{rule}_selection.json', selections[rule])
    frozen = {name: file_hash(output / name) for rule in RULES
              for name in (f'{rule}_state.pt', f'{rule}_selection.json')}
    write_json(output / 'frozen_before_evaluation.json', frozen)
    return {'trace': trace, 'selections': selections, 'frozen': frozen}


def evaluate_selection_pair(model, dataset, panel, labels, city_aqi, args, output, device, year, frozen):
    verify_frozen(output, frozen)
    result, paired = {}, None
    for rule in RULES:
        model.load_state_dict(torch.load(output / f'{rule}_state.pt', map_location=device, weights_only=True))
        _, logits = prediction_arrays(model, dataset, args, device)
        records = duration_records(dataset.origins, panel, labels, city_aqi,
                                   hazard_median(logits).numpy(), args.horizon)
        identity = records[['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']]
        if paired is not None:
            pd.testing.assert_frame_equal(identity, paired)
        paired = identity.copy()
        probability = hazard_distribution(logits).numpy()
        records['probability_recovered_by_67h'] = 1 - probability[:, -1]
        records.to_parquet(output / f'{rule}_{year}_recovery.parquet', index=False)
        np.savez_compressed(output / f'{rule}_{year}_distribution.npz',
                            origin_ns=panel['timestamps'][dataset.origins].asi8,
                            probability=probability, logits=logits.numpy(),
                            target=dataset.hazard, risk_mask=dataset.risk_mask, eligible=dataset.eligible)
        result[rule] = selection_metrics(records)
    verify_frozen(output, frozen)
    result['paired_rows'] = True
    write_json(output / f'metrics_{year}.json', result)
    return result


def run_fold(args, validation_year, test_years, output, device):
    output.mkdir(parents=True, exist_ok=False)
    args = argparse.Namespace(**vars(args))
    args.validation_year = validation_year
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    panel, city, labels, eligible, partitions = load_experiment(args)
    aqi = city.aqi.to_numpy()
    training_origins = np.concatenate([partitions[year] for year in range(2014, validation_year)])
    training = RecoveryWindows(panel, training_origins, args.history, args.horizon, aqi, eligible)
    validation = RecoveryWindows(panel, partitions[validation_year], args.history, args.horizon, aqi, eligible)
    protocol = {'configuration': vars(args), 'validation_year': validation_year,
                'test_years': list(test_years), 'nodes': panel['nodes'],
                'feature_names': panel['feature_names'], 'coordinates_lon_lat': panel['coords'].tolist(),
                'scaler': {'mean': panel['scaler'].mean.tolist(), 'scale': panel['scaler'].scale.tolist()},
                'training_windows_by_year': {year: len(partitions[year]) for year in range(2014, validation_year)},
                'partitions': {'training': len(training), 'validation': len(validation),
                               **{str(year): len(partitions[year]) for year in test_years}},
                'training_label_audit': training.audit(), 'validation_label_audit': validation.audit()}
    write_json(output / 'protocol.json', protocol)
    model = RecoveryPM25GNN(len(panel['feature_names']), panel['coords'],
                            panel['scaler'].mean[-2:], panel['scaler'].scale[-2:],
                            history=args.history, horizon=args.horizon).to(device)
    trained = train_selection_pair(model, training, validation, panel, labels, aqi, args, output, device)
    report = {**protocol, 'epochs_ran': len(trained['trace']), 'selections': trained['selections'],
              'frozen': trained['frozen'], 'years': {}}
    for year in test_years:
        dataset = RecoveryWindows(panel, partitions[year], args.history, args.horizon, aqi, eligible)
        report['years'][str(year)] = evaluate_selection_pair(
            model, dataset, panel, labels, aqi, args, output, device, year, trained['frozen'])
    write_json(output / 'report.json', report)
    return report


def run_experiment(args):
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    inputs_before, deployed_before = hashes(ROOT / args.data), hashes(deployment)
    report = {'configuration': vars(args), 'inputs_before': inputs_before,
              'deployment_before': deployed_before, 'development_gate': None,
              'final_ran': False, 'folds': {}, 'decision': 'running',
              'limitations': ['single_seed', 'previously_inspected_retrospective_years',
                               'ERA5_not_operational', 'activity_mask_pooling_only',
                               'tail68_not_exact_recovery', 'same_event_may_enter_both_groups']}
    write_json(output / 'protocol.json', report)
    try:
        torch.set_num_threads(2)
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        pairs = []
        for validation_year in (2022, 2023):
            fold = run_fold(args, validation_year, [validation_year + 1],
                            output / f'validation_{validation_year}', device)
            report['folds'][str(validation_year)] = fold
            pairs.append(fold['years'][str(validation_year + 1)])
            write_json(output / 'progress.json', report)
        gate = development_gate(pairs)
        report['development_gate'] = gate
        write_json(output / 'development_gate.json', gate)
        rule_selection = {'selected_rule': 'balanced' if gate['accepted'] else 'legacy',
                          'development_gate': gate, 'final_authorized': bool(gate['accepted'])}
        write_json(output / 'rule_selection.json', rule_selection)
        frozen_rule = file_hash(output / 'rule_selection.json')
        report['rule_selection_sha256'] = frozen_rule
        if gate['accepted']:
            report['final_started'] = True
            report['folds']['2024'] = run_fold(args, 2024, [2025, 2026], output / 'validation_2024', device)
            report['final_ran'] = True
            report['decision'] = 'research_only_final_completed'
        else:
            report['decision'] = 'reject_development_gate'
        if file_hash(output / 'rule_selection.json') != frozen_rule:
            raise RuntimeError('frozen rule selection changed')
    except Exception as exc:
        report['decision'] = 'error'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['inputs_after'], report['deployment_after'] = hashes(ROOT / args.data), hashes(deployment)
        report['inputs_unchanged'] = report['inputs_after'] == inputs_before
        report['deployment_unchanged'] = report['deployment_after'] == deployed_before
        if not report['inputs_unchanged'] or not report['deployment_unchanged']:
            report['decision'] = 'error'
            report['integrity_error'] = 'input or deployment changed during experiment'
        write_json(output / 'report.json', report)
    if not report['inputs_unchanged'] or not report['deployment_unchanged']:
        raise RuntimeError(report['integrity_error'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', default='data/processed/regional_air_models_full_history')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--auxiliary-weight', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.history, args.horizon = 24, 72
    if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0 or args.auxiliary_weight <= 0:
        parser.error('training counts and objective weights must be positive')
    run_experiment(args)


if __name__ == '__main__':
    main()
