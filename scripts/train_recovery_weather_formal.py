"""Gated three-arm training: 2014-2022 history, 2023 checkpoint selection only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from haze_forecast.operational_gefs_availability import availability_state
from haze_forecast.recovery_selection import selection_metrics
from haze_forecast.recovery_survival import hazard_distribution, masked_survival_loss
from haze_forecast.recovery_weather_inputs import (
    WEATHER_FEATURES,
    WeatherScaler,
    build_weather_inputs,
)
from haze_forecast.recovery_weather_network import (
    WeatherControlRecoveryPM25GNN,
    WeatherRecoveryPM25GNN,
)
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from scripts.benchmark_recovery_head import ROOT, RecoveryWindows, duration_records, load_experiment
from scripts.benchmark_regional_air_models import hashes, masked_loss, write_json

INPUT_FILES = ('station_hourly.parquet', 'weather_proxy.parquet', 'city_hourly.parquet')
MANIFEST_SHA256 = 'f7a77f53e8b8278c3cb650846867d66dadb13d8dd29b96235b05c7e972cd0dc2'
ARMS = {'base': RecoveryPM25GNN, 'masked_values': WeatherControlRecoveryPM25GNN,
        'weather': WeatherRecoveryPM25GNN}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def actual_input_hashes(data):
    return {name: sha(data / name) for name in INPUT_FILES}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_ready_manifest(gate_path, manifest_path, results, input_hashes):
    gate = json.loads(gate_path.read_text('utf-8'))
    require(gate.get('status') == 'ready_for_formal_training', 'offline gate is not training ready')
    require(all(gate.get(key) == value for key, value in
                {'passed_objects': 4926, 'failed_objects': 0, 'pending_objects': 0,
                 'complete_origins': 208}.items()), 'full collection with zero failures is required')
    require(sha(manifest_path) == gate.get('manifest_sha256'), 'resolved manifest hash mismatch')
    require(gate['manifest_sha256'] == MANIFEST_SHA256, 'manifest differs from frozen resolved candidates')
    require(input_hashes == gate.get('input_data_sha256'), 'offline gate history input hash mismatch')
    mapping = gate.get('results_sha256', {})
    paths = {path.name: path for path in results.glob('*.json')}
    require(len(mapping) == 4926 and set(paths) == set(mapping), 'complete result hash manifest required')
    require(all(sha(paths[name]) == expected for name, expected in mapping.items()), 'collection result hash mismatch')
    plan = pd.read_csv(manifest_path)
    require(len(plan) == 4992 and plan.url.nunique() == 4926, 'fixed candidate count mismatch')
    require(not plan.duplicated(['issue_utc', 'valid_utc']).any(), 'duplicate issue/valid weather keys')
    times = pd.to_datetime(plan.issue_utc, utc=True)
    unique = pd.DatetimeIndex(times.unique()).sort_values()
    require({int(y): int(n) for y, n in pd.Series(unique.year).value_counts().items()} == {2022: 101, 2023: 107},
            'fixed weather origin years/counts mismatch')
    for _, rows in plan.groupby('issue_utc'):
        require(sorted(rows.ahead_hours.tolist()) == list(range(3, 73, 3)), 'complete 24-step forecast required')
    require((plan.partition.to_numpy() == np.where(times.dt.year.eq(2022), 'train', 'validation')).all(),
            'weather partition labels mismatch')
    records = {}
    for path in paths.values():
        record = json.loads(path.read_text('utf-8'))
        require(record.get('status') == 'pass' and record.get('manifest_sha256') == gate['manifest_sha256'],
                'failed or stale collection record')
        url = record['probe']['url']
        require(url not in records, 'duplicate collection URL')
        records[url] = record
    require(set(records) == set(plan.url), 'collection must exactly cover resolved URL set')
    return gate, plan, records


def collected_frame(plan, records):
    verified_files, rows = {}, []
    for row in plan.to_dict('records'):
        record = records[row['url']]
        head, probe = record['head'], record['probe']
        require(head['url'] == row['url'] and head['etag'] == probe['etag'], 'object identity mismatch')
        for key in ('run_utc', 'valid_utc', 'lead_hours'):
            require(probe[key] == row[key], f'deduplicated forecast identity mismatch: {key}')
        issue, run, valid = [pd.Timestamp(row[key]) for key in ('issue_utc', 'run_utc', 'valid_utc')]
        require(availability_state(head, issue, run)[0] == 'available', 'unavailable forecast in ready collection')
        require((valid - issue).total_seconds() / 3600 == row['ahead_hours'], 'invalid forecast offset')
        require(set(record['fields']) == set(WEATHER_FEATURES), 'all four fields are required')
        values = {}
        for name in WEATHER_FEATURES:
            field = record['fields'][name]
            path = (ROOT / field['path']).resolve()
            require(path.is_relative_to((ROOT / 'artifacts/task01_recovery_head').resolve()), 'field outside artifact root')
            if path not in verified_files:
                verified_files[path] = sha(path)
            require(verified_files[path] == field['sha256'] and path.stat().st_size == field['bytes'], 'field hash/size mismatch')
            require(np.isfinite(field['value']) and abs(field['value']) < 1e10, 'invalid collected weather value')
            values[name] = field['value']
        rows.append({'issue_utc': issue, 'run_utc': run, 'valid_utc': valid,
                     'available_at_utc': pd.Timestamp(head['last_modified']), **values})
    return pd.DataFrame(rows), {str(path.relative_to(ROOT)): digest for path, digest in verified_files.items()}


def fit_weather(frame, origins):
    raw, mask = build_weather_inputs(frame, origins)
    require(mask.all(), 'complete observed forecast cells required for fixed weather origins')
    training_rows = np.asarray(pd.DatetimeIndex(origins).tz_convert('Asia/Shanghai').year == 2022)
    scaler = WeatherScaler.fit(raw, mask, training_rows)
    return scaler.transform(raw, mask).astype(np.float32), mask, scaler


class FormalWeatherWindows(RecoveryWindows):
    def __init__(self, panel, origins, city_aqi, eligible, weather_by_origin):
        super().__init__(panel, origins, 24, 72, city_aqi, eligible)
        self.weather_by_origin = weather_by_origin
        self.missing_weather = (torch.zeros(24, 4), torch.zeros(24, 4, dtype=torch.bool))

    def __getitem__(self, item):
        weather = self.weather_by_origin.get(int(self.origins[item]), self.missing_weather)
        return (*super().__getitem__(item), *weather)


def fixed_orders(count, epochs, seed):
    generator = torch.Generator().manual_seed(seed)
    return np.stack([torch.randperm(count, generator=generator).numpy() for _ in range(epochs)])


def select_key(metrics):
    key = (metrics['event_equal_mae'], metrics['row_mae'])
    require(np.isfinite(key).all(), 'finite validation selection metrics required')
    return key


def forward_batch(model, batch, device):
    x, y, mask, active, target, risk, eligible, weather, weather_mask = [value.to(device) for value in batch]
    if isinstance(model, WeatherRecoveryPM25GNN):
        curves, logits = model(x, active, weather, weather_mask)
    else:
        curves, logits = model(x, active)
    return curves, logits, y, mask, target, risk, eligible


def train_arm(model, dataset, orders, args, output, device, validate):
    require(orders.shape == (args.epochs, len(dataset)), 'batch order shape mismatch')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best, selection, trace = None, None, []
    started = time.monotonic()
    generator = torch.Generator().manual_seed(getattr(args, 'seed', 42))
    for epoch, order in enumerate(orders, 1):
        model.train()
        survival_sum, survival_count, auxiliary_sum, auxiliary_count, windows = 0., 0, 0., 0, 0
        for batch in DataLoader(dataset, batch_size=args.batch_size, sampler=order.tolist(), generator=generator):
            optimizer.zero_grad(set_to_none=True)
            curves, logits, y, mask, target, risk, eligible = forward_batch(model, batch, device)
            survival = masked_survival_loss(logits, target, risk, eligible)
            auxiliary = masked_loss(curves, y, mask)
            windows += len(y)
            losses = []
            if survival is not None:
                count = int((eligible & risk.any(dim=1)).sum())
                survival_sum += float(survival.detach()) * count
                survival_count += count
                losses.append(survival)
            if auxiliary is not None:
                count = int(mask.sum())
                auxiliary_sum += float(auxiliary.detach()) * count
                auxiliary_count += count
                losses.append(args.auxiliary_weight * auxiliary)
            if losses:
                loss = sum(losses)
                require(bool(torch.isfinite(loss)), 'non-finite formal training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
        require(survival_count > 0 and windows == len(dataset), 'all history windows must be retained')
        model.eval()
        with torch.no_grad():
            metrics = validate(model)
        key = select_key(metrics)
        row = {'epoch': epoch, 'training_windows': windows,
               'informative_survival_origins': survival_count, 'concentration_elements': auxiliary_count,
               'training_survival_nll': survival_sum / survival_count,
               'training_concentration_l1': auxiliary_sum / auxiliary_count if auxiliary_count else None,
               'validation_2023': metrics, 'selection_key': key, 'elapsed_seconds': time.monotonic() - started}
        trace.append(row)
        write_json(output / 'training_trace.json', trace)
        if best is None or key < best:
            best = key
            torch.save(model.state_dict(), output / 'selected_state.pt')
            selection = {'epoch': epoch, 'validation_year': 2023, 'validation_key': key,
                         'rule': 'event_equal_mae_then_row_mae_earliest_tie',
                         'candidate_epochs': list(range(1, args.epochs + 1))}
            write_json(output / 'selection.json', selection)
        print({'arm': output.name, **row}, flush=True)
    model.load_state_dict(torch.load(output / 'selected_state.pt', map_location=device, weights_only=True))
    return {'selection': selection, 'epochs_ran': len(trace),
            'checkpoint_sha256': sha(output / 'selected_state.pt'),
            'selection_sha256': sha(output / 'selection.json')}


def validated_distribution(logits):
    require(logits.ndim == 2 and logits.shape[0] > 0 and logits.shape[1] == 67
            and bool(torch.isfinite(logits).all()), 'validation requires nonempty finite logits [batch, 67]')
    probability = hazard_distribution(logits)
    require(probability.shape == (len(logits), 68) and bool(torch.isfinite(probability).all())
            and bool(((probability >= 0) & (probability <= 1)).all())
            and torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]), atol=2e-6, rtol=2e-6),
            'invalid validation probability distribution')
    return probability


def validation_outputs(model, dataset, panel, labels, aqi, args, device):
    pieces = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                generator=torch.Generator().manual_seed(args.seed)):
            pieces.append(forward_batch(model, batch, device)[1].cpu())
    logits = torch.cat(pieces)
    probability = validated_distribution(logits)
    medians = (probability.cumsum(1) < .5).sum(1).clamp_max(67) + 1
    frame = duration_records(dataset.origins, panel, labels, aqi, medians.numpy(), 72)
    return frame, logits, selection_metrics(frame), probability


def state_digest(model, names=None):
    state = model.state_dict()
    digest = hashlib.sha256()
    for name in state if names is None else names:
        value = state[name].detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}:'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gate', default='artifacts/task01_recovery_head/future_weather_collection_v1/training_readiness_gate.json')
    parser.add_argument('--manifest', default='artifacts/task01_recovery_head/future_weather_fallback_v1/resolved_candidate_requests.csv')
    parser.add_argument('--collection', default='artifacts/task01_recovery_head/future_weather_collection_v1')
    parser.add_argument('--data', default='data/processed/regional_air_models_full_history')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.history, args.horizon, args.validation_year = 24, 72, 2023
    args.seed, args.epochs, args.batch_size = 42, 12, 256
    args.learning_rate, args.auxiliary_weight = .001, .1
    inputs_before = actual_input_hashes(ROOT / args.data)
    gate_path, manifest_path = ROOT / args.gate, ROOT / args.manifest
    gate, plan, records = load_ready_manifest(gate_path, manifest_path, ROOT / args.collection / 'results', inputs_before)
    gate_hash = sha(gate_path)
    frame, field_hashes = collected_frame(plan, records)
    times = pd.DatetimeIndex(pd.to_datetime(plan.issue_utc.unique(), utc=True)).sort_values()
    weather, weather_mask, scaler = fit_weather(frame, times)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    panel, city, labels, eligible, partitions = load_experiment(args)
    selected = panel['timestamps'].get_indexer(times)
    require((selected >= 0).all(), 'weather origin absent from historical panel')
    for year in (2022, 2023):
        expected = partitions[year][eligible[partitions[year]]]
        np.testing.assert_array_equal(selected[times.tz_convert('Asia/Shanghai').year == year], expected)
    mapping = {int(origin): (torch.from_numpy(weather[i]), torch.from_numpy(weather_mask[i]))
               for i, origin in enumerate(selected)}
    train_origins = np.concatenate([partitions[y] for y in range(2014, 2023)])
    require(panel['timestamps'][train_origins.max() + 72] < pd.Timestamp('2023-01-01', tz='Asia/Shanghai'),
            'training target crosses validation boundary')
    training = FormalWeatherWindows(panel, train_origins, city.aqi.to_numpy(), eligible, mapping)
    validation = FormalWeatherWindows(panel, partitions[2023], city.aqi.to_numpy(), eligible, mapping)
    orders = fixed_orders(len(training), 12, 42)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'batch_orders.npy', orders)
    np.savez_compressed(output / 'weather_inputs.npz', origin_ns=times.asi8, normalized=weather,
                        mask=weather_mask, training_rows=np.asarray(times.tz_convert('Asia/Shanghai').year == 2022),
                        mean=scaler.mean, scale=scaler.scale)
    all_origins = np.concatenate([train_origins, partitions[2023]])
    pd.DataFrame({'timestamp': panel['timestamps'][all_origins], 'origin': all_origins,
                  'partition': ['train'] * len(training) + ['validation'] * len(validation),
                  'survival_eligible': eligible[all_origins],
                  'future_weather_available': np.isin(all_origins, selected)}).to_parquet(output / 'windows.parquet', index=False)
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    deployment_before = hashes(deployment)
    report = {'configuration': vars(args), 'status': 'running', 'arms': {},
              'protocol': '2014-2022 all history windows train; 2023 selects checkpoint; no independent test',
              'weather_origins': {'training_2022': 101, 'validation_2023': 107},
              'inputs_before': inputs_before, 'deployment_before': deployment_before,
              'gate_sha256': gate_hash, 'manifest_sha256': gate['manifest_sha256'],
              'results_sha256': gate['results_sha256'], 'field_sha256': field_hashes,
              'batch_orders_sha256': sha(output / 'batch_orders.npy'),
              'weather_inputs_sha256': sha(output / 'weather_inputs.npz'),
              'nodes': panel['nodes'], 'feature_names': panel['feature_names'],
              'history_scaler': {'mean': panel['scaler'].mean.tolist(), 'scale': panel['scaler'].scale.tolist()},
              'weather_scaler': {'mean': scaler.mean.tolist(), 'scale': scaler.scale.tolist(), 'fit_year': 2022},
              'training_windows_by_year': {year: len(partitions[year]) for year in range(2014, 2023)},
              'training_label_audit': training.audit(), 'validation_label_audit': validation.audit(),
              'limitations': ['2023_checkpoint_selection_not_independent_test', 'previously_inspected_years',
                             'single_seed', 'archive_timestamp_proxy_not_first_publication',
                             'future_weather_absent_outside_fixed_2022_2023_origins',
                             'masked_values_control_retains_missingness', 'tail68_is_not_exact_recovery',
                             'no_deployment_or_2025_2026_error_evaluation']}
    write_json(output / 'protocol.json', report)
    models, base_names, initial_full, initial_base = {}, None, {}, {}
    try:
        for arm, cls in ARMS.items():
            torch.manual_seed(42)
            models[arm] = cls(len(panel['feature_names']), panel['coords'], panel['scaler'].mean[-2:],
                              panel['scaler'].scale[-2:]).to(device)
            if base_names is None:
                base_names = list(models[arm].state_dict())
            initial_full[arm] = state_digest(models[arm])
            initial_base[arm] = state_digest(models[arm], base_names)
        require(len(set(initial_base.values())) == 1, 'base initial weights differ')
        require(initial_full['masked_values'] == initial_full['weather'], 'weather control initial weights differ')
        for arm in ('masked_values', 'weather'):
            for name in base_names:
                require(torch.equal(models['base'].state_dict()[name], models[arm].state_dict()[name]),
                        f'base initial parameter mismatch: {arm}/{name}')
        for name, value in models['masked_values'].state_dict().items():
            require(torch.equal(value, models['weather'].state_dict()[name]), f'matched initial parameter mismatch: {name}')
        report['initial_state_sha256'], report['initial_base_state_sha256'] = initial_full, initial_base
        for arm, model in models.items():
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            folder = output / arm
            folder.mkdir()
            torch.save(model.state_dict(), folder / 'initial_state.pt')

            def validate(current):
                return validation_outputs(current, validation, panel, labels, city.aqi.to_numpy(), args, device)[2]

            arm_report = train_arm(model, training, orders, args, folder, device, validate)
            model.eval()
            predicted, logits, metrics, checked_probability = validation_outputs(
                model, validation, panel, labels, city.aqi.to_numpy(), args, device)
            probability = checked_probability.numpy()
            predicted['probability_recovered_by_67h'] = 1 - probability[:, -1]
            predicted.to_parquet(folder / 'validation_2023_recovery.parquet', index=False)
            np.savez_compressed(folder / 'validation_2023_distribution.npz', origin_ns=panel['timestamps'][validation.origins].asi8,
                                logits=logits.numpy(), probability=probability, target=validation.hazard,
                                risk_mask=validation.risk_mask, eligible=validation.eligible)
            require(sha(folder / 'selected_state.pt') == arm_report['checkpoint_sha256'], 'selected checkpoint changed during validation')
            arm_report.update(validation_2023=metrics, parameters=sum(p.numel() for p in model.parameters()),
                              initial_state_sha256=sha(folder / 'initial_state.pt'))
            report['arms'][arm] = arm_report
            write_json(folder / 'report.json', arm_report)
            write_json(output / 'progress.json', report)
        report['status'] = 'completed_train_validation_only'
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['inputs_unchanged'] = actual_input_hashes(ROOT / args.data) == inputs_before
        report['deployment_unchanged'] = hashes(deployment) == deployment_before
        report['gate_unchanged'] = sha(gate_path) == gate_hash
        report['manifest_unchanged'] = sha(manifest_path) == gate['manifest_sha256']
        report['results_unchanged'] = all(sha(ROOT / args.collection / 'results' / name) == digest
                                          for name, digest in gate['results_sha256'].items())
        report['fields_unchanged'] = all(sha(ROOT / path) == digest for path, digest in field_hashes.items())
        integrity = all(report[key] for key in ('inputs_unchanged', 'deployment_unchanged', 'gate_unchanged',
                                               'manifest_unchanged', 'results_unchanged', 'fields_unchanged'))
        if not integrity:
            report['status'] = 'error'
        write_json(output / 'report.json', report)
    require(integrity, 'formal input provenance changed during training')
    print(f'Completed train/validation only: {output / "report.json"}')


if __name__ == '__main__':
    main()
