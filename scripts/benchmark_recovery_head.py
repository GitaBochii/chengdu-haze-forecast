"""Isolated recovery-head pilot using the existing chronological regional protocol."""

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

from haze_forecast.labels import label_city_data
from haze_forecast.recovery_survival import (
    causal_recovery_eligibility,
    hazard_distribution,
    hazard_median,
    masked_survival_loss,
    recovery_hazard_labels,
    validate_hourly_grid,
)
from haze_forecast.regional_air_experiment import (
    boundary_event_mask,
    prepare_panel,
    purge_event_windows,
    recovery_metrics,
    sequence_origins,
)
from haze_forecast.regional_air_scoring import active_node_mask, verified_recovery_mask
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_regional_air_models import Windows, evaluate, hashes, masked_loss, write_json

ROOT = Path(__file__).resolve().parents[1]


class RecoveryWindows(Windows):
    def __init__(self, panel, origins, history, horizon, city_aqi, eligible):
        super().__init__(panel, origins, history, horizon)
        positions = origins[:, None] + np.arange(1, horizon + 1)
        self.hazard, self.risk_mask = recovery_hazard_labels(city_aqi[positions])
        self.active = active_node_mask(panel['targets'], origins, lookback=history)
        self.eligible = eligible[origins]

    def __getitem__(self, item):
        return (*super().__getitem__(item), torch.from_numpy(self.active[item]),
                torch.from_numpy(self.hazard[item]), torch.from_numpy(self.risk_mask[item]),
                torch.tensor(bool(self.eligible[item]), dtype=torch.bool))

    def audit(self):
        informative = self.eligible & self.risk_mask.any(axis=1)
        return {'windows': len(self), 'eligible': int(self.eligible.sum()),
                'informative_eligible': int(informative.sum()),
                'exact_recoveries': int((self.hazard.sum(axis=1).astype(bool) & informative).sum()),
                'censored_informative': int((~self.hazard.sum(axis=1).astype(bool) & informative).sum()),
                'eligible_without_observed_risk': int((self.eligible & ~informative).sum())}


def duration_records(origins, panel, labels, city_aqi, predicted, horizon):
    times = panel['timestamps']
    event = labels.reindex(times[origins])
    positions = origins[:, None] + np.arange(1, horizon + 1)
    records = pd.DataFrame({
        'timestamp': times[origins], 'event_id': event['event_id'].to_numpy(),
        'actual_hours': event['remaining_hours'].to_numpy(), 'predicted_hours': predicted,
    })
    records['label_verified_in_scoring_window'] = verified_recovery_mask(
        records.actual_hours.to_numpy(), city_aqi[positions])
    return records


def prediction_arrays(model, dataset, args, device):
    model.eval()
    curves, logits = [], []
    with torch.no_grad():
        for x, _, _, active, *_ in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            curve, risk = model(x.to(device), active.to(device))
            curves.append(curve.cpu())
            logits.append(risk.cpu())
    return torch.cat(curves), torch.cat(logits)


def train(model, training, validation, panel, labels, city_aqi, args, output, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, generator=generator)
    trace, best, best_state, stale = [], (float('inf'), float('inf')), None, 0
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
        medians = hazard_median(logits).numpy()
        records = duration_records(validation.origins, panel, labels, city_aqi, medians, args.horizon)
        metrics = recovery_metrics(records.loc[records.label_verified_in_scoring_window], args.horizon)
        if not metrics['event_rows']:
            raise ValueError('validation contains no verifiable recovery events')
        key = (metrics['event_equal_restricted_time_mae'], metrics['restricted_time_mae'])
        row = {'epoch': epoch, 'training_survival_nll': nll_total / nll_count,
               'training_concentration_l1': aux_total / aux_count if aux_count else None,
               'validation_2024_duration': metrics, 'elapsed_seconds': time.monotonic() - started}
        trace.append(row)
        write_json(output / 'training_trace.json', trace)
        print(row, flush=True)
        if key < best:
            best, stale = key, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, output / 'recovery_head_state.pt')
            write_json(output / 'selection.json', {'epoch': epoch, 'validation_key': key,
                                                  'rule': 'event_equal_mae_then_row_mae', 'seed': args.seed})
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return trace


def load_experiment(args):
    data = ROOT / args.data
    stations = pd.read_parquet(data / 'station_hourly.parquet')
    weather = pd.read_parquet(data / 'weather_proxy.parquet')
    city_frame = pd.read_parquet(data / 'city_hourly.parquet')
    validate_hourly_grid(city_frame.timestamp)
    labels = label_city_data(city_frame).set_index('timestamp')
    start = pd.Timestamp('2014-01-01', tz='Asia/Shanghai')
    validation_year = getattr(args, 'validation_year', 2024)
    if not 2015 <= validation_year <= 2024:
        raise ValueError('validation year must follow training start and precede final evaluation')
    cutoff = pd.Timestamp(f'{validation_year}-01-01', tz='Asia/Shanghai')
    panel = prepare_panel(stations, weather, cutoff, training_start=start, ozone_column='o3_8h')
    times = panel['timestamps']
    validate_hourly_grid(times)
    city = city_frame.set_index('timestamp').reindex(times)
    blocked = boundary_event_mask(labels.reindex(times), [start, *[
        pd.Timestamp(f'{year}-01-01', tz=times.tz) for year in range(validation_year, 2027)]])
    known_weather = weather.set_index('timestamp').reindex(times)[
        ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']].notna().all(axis=1).to_numpy()

    def origins(year):
        selected = sequence_origins(times, history=args.history, horizon=args.horizon,
                                    start=pd.Timestamp(f'{year}-01-01', tz=times.tz),
                                    end=min(pd.Timestamp(f'{year+1}-01-01', tz=times.tz),
                                            times[-1] + pd.Timedelta(hours=1)), stride=6)
        selected = purge_event_windows(selected, blocked, history=args.history, horizon=args.horizon)
        selected = selected[known_weather[selected]]
        active = active_node_mask(panel['targets'], selected, lookback=args.history)
        return selected[active.any(axis=1).all(axis=1)]

    partitions = {year: origins(year) for year in range(2014, 2027)}
    if any(not len(values) for values in partitions.values()):
        raise ValueError('all declared years must contribute valid windows')
    eligible = causal_recovery_eligibility(city.aqi.to_numpy(), labels.reindex(times).particle_primary.to_numpy())
    return panel, city, labels, eligible, partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data', default='data/processed/regional_air_models_full_history')
    parser.add_argument('--reference', default='artifacts/regional_air_models/history_structure_seed42_v1')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--auxiliary-weight', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.history, args.horizon = 24, 72
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.auxiliary_weight <= 0:
        parser.error('training counts and auxiliary weight must be positive')
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    inputs_before = hashes(ROOT / args.data)
    reference_before = hashes(ROOT / args.reference)
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    deployed_before = hashes(deployment)
    protocol = {'configuration': vars(args), 'inputs_before': inputs_before,
                'reference_before': reference_before,
                'deployment_before': deployed_before,
                'protocol': '2014-2023 train; 2024 duration selects checkpoint; freeze before 2025/2026',
                'limitations': ['single_seed_retrospective_exploration', 'ERA5_not_operational_weather',
                               'tail68_is_not_exact_recovery', 'activity_masks_pooling_only',
                               'validation_and_test_years_previously_inspected'],
                'label': 'first6_observed_AQI_le100_start; censor_at_first_missing_prefix',
                'eligibility': 'causal_active_particle_event_or_current_particle_pollution'}
    write_json(output / 'protocol.json', protocol)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    panel, city, labels, eligible, partitions = load_experiment(args)
    aqi = city.aqi.to_numpy()
    training_origins = np.concatenate([partitions[year] for year in range(2014, 2024)])
    training = RecoveryWindows(panel, training_origins, 24, 72, aqi, eligible)
    validation = RecoveryWindows(panel, partitions[2024], 24, 72, aqi, eligible)
    protocol.update({'nodes': panel['nodes'], 'feature_names': panel['feature_names'],
                     'coordinates_lon_lat': panel['coords'].tolist(),
                     'scaler': {'mean': panel['scaler'].mean.tolist(), 'scale': panel['scaler'].scale.tolist()},
                     'training_windows_by_year': {year: len(partitions[year]) for year in range(2014, 2024)},
                     'partitions': {'training': len(training), 'validation': len(validation),
                                    '2025': len(partitions[2025]), '2026': len(partitions[2026])},
                     'training_label_audit': training.audit(), 'validation_label_audit': validation.audit()})
    write_json(output / 'protocol.json', protocol)
    model = RecoveryPM25GNN(len(panel['feature_names']), panel['coords'],
                            panel['scaler'].mean[-2:], panel['scaler'].scale[-2:]).to(device)
    trace = train(model, training, validation, panel, labels, aqi, args, output, device)
    frozen_checkpoint_hash = hashlib.sha256((output / 'recovery_head_state.pt').read_bytes()).hexdigest()
    frozen_selection_hash = hashlib.sha256((output / 'selection.json').read_bytes()).hexdigest()
    write_json(output / 'frozen_before_evaluation.json',
               {'checkpoint_sha256': frozen_checkpoint_hash, 'selection_sha256': frozen_selection_hash})
    report = {**protocol, 'epochs_ran': len(trace), 'parameters': sum(p.numel() for p in model.parameters()),
              'checkpoint_sha256': frozen_checkpoint_hash, 'years': {}}
    reference = ROOT / args.reference
    import json
    reference_report = json.loads((reference / 'report.json').read_text('utf-8'))
    for key in ('nodes', 'feature_names', 'coordinates_lon_lat', 'scaler', 'partitions'):
        if report[key] != reference_report[key]:
            raise ValueError(f'reference mismatch: {key}')
    if inputs_before != reference_report['input_data_sha256']:
        raise ValueError('reference input hashes do not match')
    for year in (2025, 2026):
        selected = partitions[year]
        dataset = RecoveryWindows(panel, selected, 24, 72, aqi, eligible)
        curve, logits = prediction_arrays(model, dataset, args, device)
        medians = hazard_median(logits).numpy()
        records = duration_records(selected, panel, labels, aqi, medians, 72)
        records['model'] = 'recovery_head'
        probabilities = hazard_distribution(logits).numpy()
        records['probability_recovered_by_67h'] = 1 - probabilities[:, -1]
        records.to_parquet(output / f'recovery_head_{year}_recovery.parquet', index=False)
        np.savez_compressed(output / f'recovery_head_{year}_distribution.npz',
                            origin_ns=panel['timestamps'][selected].asi8, logits=logits.numpy(),
                            probability=probabilities, target=dataset.hazard, risk_mask=dataset.risk_mask,
                            eligible=dataset.eligible)
        verified = records.label_verified_in_scoring_window
        metrics = recovery_metrics(records.loc[verified], 72)
        reference_rows = pd.read_parquet(reference / f'pm25_gnn_history_{year}_recovery.parquet')
        for key in ('timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window'):
            pd.testing.assert_series_equal(records[key], reference_rows[key], check_names=False)
        baseline = recovery_metrics(reference_rows.loc[verified], 72)
        physical_curve = np.maximum(0, curve.numpy() * panel['scaler'].scale[[0, 2]]
                                    + panel['scaler'].mean[[0, 2]])
        curve_result = evaluate(physical_curve, panel, selected, city, labels, args, output,
                                'auxiliary_curve', year)
        nll = masked_survival_loss(logits, torch.from_numpy(dataset.hazard),
                                   torch.from_numpy(dataset.risk_mask), torch.from_numpy(dataset.eligible))
        known_exact = records.actual_hours.notna() & verified
        binary_truth = (records.loc[known_exact, 'actual_hours'].to_numpy() < 68).astype(float)
        binary_probability = records.loc[known_exact, 'probability_recovered_by_67h'].to_numpy()
        report['years'][year] = {'duration': metrics, 'reference_history_gnn_duration': baseline,
                                 'auxiliary_curve': curve_result,
                                 'label_audit': dataset.audit(),
                                 'eligible_observed_prefix_nll': float(nll) if nll is not None else None,
                                 'verified_event_brier_by_67h': float(np.mean((binary_truth - binary_probability)**2)),
                                 'paired_reference': True}
        write_json(output / 'progress.json', report)
        print({'year': year, 'duration': metrics, 'reference': baseline}, flush=True)
    if hashes(ROOT / args.data) != inputs_before or hashes(deployment) != deployed_before:
        raise RuntimeError('input or deployment changed during experiment')
    if hashes(reference) != reference_before:
        raise RuntimeError('reference artifacts changed during experiment')
    if hashlib.sha256((output / 'recovery_head_state.pt').read_bytes()).hexdigest() != frozen_checkpoint_hash:
        raise RuntimeError('checkpoint mutated during evaluation')
    if hashlib.sha256((output / 'selection.json').read_bytes()).hexdigest() != frozen_selection_hash:
        raise RuntimeError('selection mutated during evaluation')
    promising = all(values['duration'][metric] < values['reference_history_gnn_duration'][metric]
                    for values in report['years'].values()
                    for metric in ('restricted_time_mae', 'event_equal_restricted_time_mae'))
    report.update({'decision': 'research_candidate_only' if promising else 'reject',
                   'deployment_unchanged': True, 'inputs_unchanged': True,
                   'selection_sha256': frozen_selection_hash})
    write_json(output / 'report.json', report)
    print(f"Report: {output / 'report.json'}", flush=True)


if __name__ == '__main__':
    main()
