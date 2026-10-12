"""Frozen three-arm retraining with separately attested 2020-2023 forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch

from haze_forecast.recovery_weather_inputs import WeatherScaler, build_weather_inputs
from scripts.benchmark_recovery_head import ROOT, load_experiment
from scripts.benchmark_regional_air_models import hashes, write_json
from scripts.train_recovery_weather_formal import (
    ARMS,
    FormalWeatherWindows,
    actual_input_hashes,
    collected_frame,
    fixed_orders,
    load_ready_manifest,
    require,
    sha,
    state_digest,
    train_arm,
    validation_outputs,
)
from scripts.verify_weather_expansion_ready import (
    COUNTS as EXPANSION_COUNTS,
)
from scripts.verify_weather_expansion_ready import (
    MANIFEST_SHA256 as EXPANSION_MANIFEST_SHA256,
)
from scripts.verify_weather_training_ready import check_plan

BASE = ROOT / 'artifacts/task01_recovery_head'
REFERENCE = BASE / 'future_weather_formal_v1'
SOURCE_PATHS = {
    'original': ('future_weather_collection_v1',
                 'future_weather_fallback_v1/resolved_candidate_requests.csv'),
    'expansion': ('future_weather_expansion_collection_v1',
                  'future_weather_expansion_resolved_v1/accepted_manifest.csv'),
}
ORIGIN_COUNTS = {2020: 60, 2021: 118, 2022: 101, 2023: 107}
SOURCE_CODE = (
    'scripts/train_recovery_weather_expanded.py',
    'scripts/train_recovery_weather_formal.py',
    'scripts/verify_recovery_weather_expanded.py',
    'scripts/verify_recovery_weather_formal.py',
    'scripts/verify_recovery_auxiliary.py',
    'scripts/verify_weather_expansion_ready.py',
    'scripts/verify_weather_training_ready.py',
    'scripts/benchmark_recovery_head.py',
    'scripts/benchmark_regional_air_models.py',
    'src/haze_forecast/recovery_weather_inputs.py',
    'src/haze_forecast/recovery_weather_network.py',
    'src/haze_forecast/regional_recovery_network.py',
    'src/haze_forecast/regional_air_networks.py',
    'src/haze_forecast/regional_air_experiment.py',
    'src/haze_forecast/regional_air_scoring.py',
    'src/haze_forecast/recovery_survival.py',
    'src/haze_forecast/recovery_selection.py',
    'src/haze_forecast/operational_gefs_availability.py',
    'src/haze_forecast/labels.py',
    'experiments/regional_air_models/vendor/PM25-GNN/model/cells.py',
    'experiments/regional_air_models/vendor/PM25-GNN/model/PM25_GNN.py',
)


def hashes_unchanged(expected):
    return all((ROOT / path).is_file() and sha(ROOT / path) == digest
               for path, digest in expected.items())


def weather_training_rows(origins):
    years = pd.DatetimeIndex(origins).tz_convert('Asia/Shanghai').year
    return np.isin(years, [2020, 2021, 2022])


def fit_expanded_weather(frame, origins):
    raw, mask = build_weather_inputs(frame, origins)
    require(mask.all(), 'complete observed forecast cells required')
    scaler = WeatherScaler.fit(raw, mask, weather_training_rows(origins))
    return scaler.transform(raw, mask).astype(np.float32), mask, scaler


def load_expansion_manifest(gate_path, manifest_path, results, input_hashes):
    gate = json.loads(gate_path.read_text('utf-8'))
    require(gate.get('status') == 'ready_for_formal_training', 'expansion gate is not ready')
    require(all(gate.get(key) == value for key, value in
                {'passed_objects': EXPANSION_COUNTS[2], 'failed_objects': 0,
                 'pending_objects': 0, 'complete_origins': EXPANSION_COUNTS[1],
                 'manifest_relationships': EXPANSION_COUNTS[0],
                 'verified_fields': EXPANSION_COUNTS[2] * 4}.items()),
            'expansion collection is incomplete')
    require(sha(manifest_path) == gate.get('manifest_sha256') == EXPANSION_MANIFEST_SHA256,
            'expansion manifest differs from fixed accepted candidates')
    require(input_hashes == gate.get('input_data_sha256'), 'expansion historical input hash mismatch')
    mapping = gate.get('results_sha256', {})
    paths = {path.name: path for path in results.glob('*.json')}
    require(len(mapping) == EXPANSION_COUNTS[2] and set(paths) == set(mapping),
            'expansion complete result hash manifest required')
    require(all(sha(paths[name]) == digest for name, digest in mapping.items()), 'expansion result hash mismatch')
    plan = pd.read_csv(manifest_path)
    check_plan(plan, EXPANSION_COUNTS)
    years = pd.to_datetime(plan.issue_utc, utc=True).dt.tz_convert('Asia/Shanghai').dt.year
    require(np.array_equal(plan.year, years), 'expansion year identity mismatch')
    require(plan.groupby('year').issue_utc.nunique().to_dict() == {year: ORIGIN_COUNTS[year] for year in (2020, 2021)},
            'fixed expansion year counts mismatch')
    records = {}
    for path in paths.values():
        record = json.loads(path.read_text('utf-8'))
        require(record.get('status') == 'pass' and record.get('manifest_sha256') == EXPANSION_MANIFEST_SHA256,
                'failed or stale expansion result')
        url = record['probe']['url']
        require(path.name == hashlib.sha256(url.encode()).hexdigest() + '.json', 'expansion filename URL mismatch')
        require(url not in records, 'duplicate expansion URL')
        records[url] = record
    require(set(records) == set(plan.url), 'expansion URL set mismatch')
    return gate, plan, records


def load_sources(input_hashes):
    sources, frames, fields, seen_urls = {}, [], {}, set()
    for name, (collection_name, manifest_name) in SOURCE_PATHS.items():
        collection, manifest = BASE / collection_name, BASE / manifest_name
        gate_path = collection / 'training_readiness_gate.json'
        gate_hash = sha(gate_path)
        loader = load_ready_manifest if name == 'original' else load_expansion_manifest
        gate, plan, records = loader(gate_path, manifest, collection / 'results', input_hashes)
        require(sha(gate_path) == gate_hash, 'gate changed while loading')
        require(not seen_urls.intersection(records), 'original and expansion URL overlap')
        seen_urls.update(records)
        evidence_hashes = gate.get('evidence_sha256', {})
        require(evidence_hashes and hashes_unchanged(evidence_hashes), 'gate evidence changed')
        frame, source_fields = collected_frame(plan, records)
        require(source_fields == gate.get('grib_sha256'), 'field set differs from independent gate')
        require(not set(fields).intersection(source_fields), 'cross-source GRIB path reuse')
        fields.update(source_fields)
        frames.append(frame)
        sources[name] = {
            'gate_path': str(gate_path.relative_to(ROOT)),
            'manifest_path': str(manifest.relative_to(ROOT)),
            'collection': str(collection.relative_to(ROOT)),
            'gate_sha256': gate_hash, 'manifest_sha256': gate['manifest_sha256'],
            'results_sha256': gate['results_sha256'], 'evidence_sha256': evidence_hashes,
            'grib_sha256': source_fields,
        }
    frame = pd.concat(frames, ignore_index=True).sort_values(['issue_utc', 'valid_utc']).reset_index(drop=True)
    require(len(frame) == 9264 and len(seen_urls) == 9110 and frame.issue_utc.nunique() == 386,
            'combined frozen coverage mismatch')
    require(not frame.duplicated(['issue_utc', 'valid_utc']).any(), 'combined duplicate origin/valid')
    times = pd.DatetimeIndex(frame.issue_utc.unique()).sort_values()
    counts = pd.Series(times.tz_convert('Asia/Shanghai').year).value_counts().to_dict()
    require(counts == ORIGIN_COUNTS, 'combined weather origin year counts mismatch')
    return sources, frame, fields


def sources_unchanged(sources):
    for source in sources.values():
        if not hashes_unchanged({source['gate_path']: source['gate_sha256'],
                                 source['manifest_path']: source['manifest_sha256'],
                                 **source['evidence_sha256'], **source['grib_sha256']}):
            return False
        folder = ROOT / source['collection'] / 'results'
        if {path.name for path in folder.glob('*.json')} != set(source['results_sha256']):
            return False
        if not all(sha(folder / name) == digest for name, digest in source['results_sha256'].items()):
            return False
    return True


def reference_snapshot():
    names = ['report.json', 'windows.parquet', 'batch_orders.npy']
    names.extend(f'{arm}/{name}' for arm in ARMS for name in
                 ('initial_state.pt', 'validation_2023_distribution.npz', 'validation_2023_recovery.parquet'))
    return {str((REFERENCE / name).relative_to(ROOT)): sha(REFERENCE / name) for name in names}


def weather_label_audit(dataset):
    covered = np.array([int(origin) in dataset.weather_by_origin for origin in dataset.origins])
    informative = covered & dataset.eligible & dataset.risk_mask.any(axis=1)
    exact = dataset.hazard.any(axis=1) & informative
    censored = informative & ~exact
    complete = censored & dataset.risk_mask.all(axis=1)
    return {'covered_origins': int(covered.sum()), 'informative': int(informative.sum()),
            'exact': int(exact.sum()), 'censored': int(censored.sum()),
            'full_horizon_no_recovery': int(complete.sum()),
            'partial_prefix_censored': int((censored & ~complete).sum()),
            'risk_bins': int(dataset.risk_mask[informative].sum())}


def prepare_experiment(args, frame):
    times = pd.DatetimeIndex(frame.issue_utc.unique()).sort_values()
    weather, mask, scaler = fit_expanded_weather(frame, times)
    panel, city, labels, eligible, partitions = load_experiment(args)
    selected = panel['timestamps'].get_indexer(times)
    require((selected >= 0).all() and len(set(selected)) == 386, 'weather origin absent or duplicated in panel')
    years = times.tz_convert('Asia/Shanghai').year
    for year in (2022, 2023):
        np.testing.assert_array_equal(selected[years == year], partitions[year][eligible[partitions[year]]])
    for year in (2020, 2021):
        origins = selected[years == year]
        require(len(origins) == ORIGIN_COUNTS[year] and np.isin(origins, partitions[year]).all()
                and eligible[origins].all(), 'expansion origins outside eligible training partition')
    mapping = {int(origin): (torch.from_numpy(weather[i]), torch.from_numpy(mask[i]))
               for i, origin in enumerate(selected)}
    train_origins = np.concatenate([partitions[y] for y in range(2014, 2023)])
    require(len(train_origins) == 12337 and len(partitions[2023]) == 1427, 'historical window count changed')
    require(panel['timestamps'][train_origins.max() + 72] < pd.Timestamp('2023-01-01', tz='Asia/Shanghai'),
            'training target crosses validation boundary')
    training = FormalWeatherWindows(panel, train_origins, city.aqi.to_numpy(), eligible, mapping)
    validation = FormalWeatherWindows(panel, partitions[2023], city.aqi.to_numpy(), eligible, mapping)
    reference = json.loads((REFERENCE / 'report.json').read_text('utf-8'))
    require(reference['status'] == 'completed_train_validation_only', 'reference run not complete')
    require(actual_input_hashes(ROOT / args.data) == reference['inputs_before'], 'history differs from reference')
    require(training.audit() == reference['training_label_audit']
            and validation.audit() == reference['validation_label_audit'], 'reference label audit changed')
    require(panel['nodes'] == reference['nodes'] and panel['feature_names'] == reference['feature_names'],
            'reference station/features changed')
    np.testing.assert_array_equal(panel['scaler'].mean, reference['history_scaler']['mean'])
    np.testing.assert_array_equal(panel['scaler'].scale, reference['history_scaler']['scale'])
    with np.load(REFERENCE / 'base/validation_2023_distribution.npz', allow_pickle=False) as saved:
        for key, value in (('origin_ns', panel['timestamps'][validation.origins].asi8),
                           ('target', validation.hazard), ('risk_mask', validation.risk_mask),
                           ('eligible', validation.eligible)):
            np.testing.assert_array_equal(saved[key], value)
    current = window_frame(panel, training, validation, eligible, selected)
    columns = ['timestamp', 'origin', 'partition', 'survival_eligible']
    pd.testing.assert_frame_equal(current[columns], pd.read_parquet(REFERENCE / 'windows.parquet')[columns])
    require(current.groupby('partition').future_weather_available.sum().to_dict() == {'train': 279, 'validation': 107},
            'weather training/validation coverage changed')
    return panel, city, labels, eligible, partitions, times, weather, mask, scaler, training, validation


def window_frame(panel, training, validation, eligible, selected):
    origins = np.concatenate([training.origins, validation.origins])
    return pd.DataFrame({'timestamp': panel['timestamps'][origins], 'origin': origins,
                         'partition': ['train'] * len(training) + ['validation'] * len(validation),
                         'survival_eligible': eligible[origins],
                         'future_weather_available': np.isin(origins, selected)})


def initialize_models(panel, device):
    models, full, shared = {}, {}, {}
    base_names = None
    for arm, cls in ARMS.items():
        torch.manual_seed(42)
        model = cls(len(panel['feature_names']), panel['coords'], panel['scaler'].mean[-2:],
                    panel['scaler'].scale[-2:]).to(device)
        if base_names is None:
            base_names = list(model.state_dict())
        reference = torch.load(REFERENCE / arm / 'initial_state.pt', map_location=device, weights_only=True)
        require(list(reference) == list(model.state_dict()), 'reference model structure changed')
        for name, tensor in model.state_dict().items():
            require(torch.equal(tensor, reference[name]), f'reference initial weight changed: {arm}/{name}')
        full[arm], shared[arm] = state_digest(model), state_digest(model, base_names)
        models[arm] = model
    require(len(set(shared.values())) == 1, 'shared initial weights differ')
    require(full['masked_values'] == full['weather'], 'weather/control initial weights differ')
    for arm in ('masked_values', 'weather'):
        for name in base_names:
            require(torch.equal(models['base'].state_dict()[name], models[arm].state_dict()[name]),
                    f'base initial parameter mismatch: {arm}/{name}')
    return models, full, shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/task01_recovery_head/future_weather_expanded_formal_v1')
    parser.add_argument('--data', default='data/processed/regional_air_models_full_history')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.history, args.horizon, args.validation_year = 24, 72, 2023
    args.seed, args.epochs, args.batch_size = 42, 12, 256
    args.learning_rate, args.auxiliary_weight = .001, .1
    output = (ROOT / args.output).resolve()
    require(output.is_relative_to(BASE.resolve()) and not output.exists(), 'new isolated output directory required')
    source_code = {path: sha(ROOT / path) for path in SOURCE_CODE}
    reference_hashes = reference_snapshot()
    inputs_before = actual_input_hashes(ROOT / args.data)
    deployment = ROOT / 'artifacts/cnemc_2025_deployment_candidate'
    deployment_before = hashes(deployment)
    sources, frame, field_hashes = load_sources(inputs_before)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    prepared = prepare_experiment(args, frame)
    panel, city, labels, eligible, partitions, times, weather, weather_mask, scaler, training, validation = prepared
    orders = fixed_orders(len(training), 12, 42)
    np.testing.assert_array_equal(orders, np.load(REFERENCE / 'batch_orders.npy', allow_pickle=False))
    require(hashes_unchanged(source_code) and hashes_unchanged(reference_hashes), 'source/reference changed during preparation')
    models, initial_full, initial_base = initialize_models(panel, device)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'batch_orders.npy', orders)
    np.savez_compressed(output / 'weather_inputs.npz', origin_ns=times.asi8, normalized=weather,
                        mask=weather_mask, training_rows=weather_training_rows(times),
                        mean=scaler.mean, scale=scaler.scale)
    np.savez_compressed(output / 'training_targets.npz', origin_ns=panel['timestamps'][training.origins].asi8,
                        target=training.hazard, risk_mask=training.risk_mask, eligible=training.eligible)
    windows = window_frame(panel, training, validation, eligible, panel['timestamps'].get_indexer(times))
    windows.to_parquet(output / 'windows.parquet', index=False)
    report = {'configuration': vars(args), 'status': 'running', 'arms': {},
              'protocol': '2014-2022 all history windows train; 2023 selects checkpoint; no independent test',
              'experiment': 'expanded_weather_2020_2022_training_fixed_2023_validation',
              'weather_origins': {'by_year': ORIGIN_COUNTS, 'train': 279, 'validation': 107},
              'inputs_before': inputs_before, 'deployment_before': deployment_before,
              'sources': sources, 'field_sha256': field_hashes,
              'source_code_sha256': source_code, 'reference_v1_sha256': reference_hashes,
              'batch_orders_sha256': sha(output / 'batch_orders.npy'),
              'weather_inputs_sha256': sha(output / 'weather_inputs.npz'),
              'training_targets_sha256': sha(output / 'training_targets.npz'),
              'windows_sha256': sha(output / 'windows.parquet'),
              'nodes': panel['nodes'], 'feature_names': panel['feature_names'],
              'history_scaler': {'mean': panel['scaler'].mean.tolist(), 'scale': panel['scaler'].scale.tolist()},
              'weather_scaler': {'mean': scaler.mean.tolist(), 'scale': scaler.scale.tolist(), 'fit_years': [2020, 2021, 2022]},
              'training_windows_by_year': {year: len(partitions[year]) for year in range(2014, 2023)},
              'training_label_audit': training.audit(), 'validation_label_audit': validation.audit(),
              'weather_training_label_audit': weather_label_audit(training),
              'weather_validation_label_audit': weather_label_audit(validation),
              'initial_state_sha256': initial_full, 'initial_base_state_sha256': initial_base,
              'limitations': ['2023_checkpoint_selection_not_independent_test', 'previously_inspected_years',
                              'single_seed', 'archive_timestamp_proxy_not_first_publication',
                              'future_weather_absent_outside_fixed_386_origins',
                              'masked_values_control_retains_missingness', 'tail68_is_not_exact_recovery',
                              'no_deployment_or_2025_2026_error_evaluation']}
    write_json(output / 'protocol.json', report)
    try:
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
            predicted, logits, metrics, probability = validation_outputs(
                model, validation, panel, labels, city.aqi.to_numpy(), args, device)
            probability = probability.numpy()
            predicted['probability_recovered_by_67h'] = 1 - probability[:, -1]
            predicted.to_parquet(folder / 'validation_2023_recovery.parquet', index=False)
            np.savez_compressed(folder / 'validation_2023_distribution.npz',
                                origin_ns=panel['timestamps'][validation.origins].asi8,
                                logits=logits.numpy(), probability=probability, target=validation.hazard,
                                risk_mask=validation.risk_mask, eligible=validation.eligible)
            require(sha(folder / 'selected_state.pt') == arm_report['checkpoint_sha256'],
                    'selected checkpoint changed during validation')
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
        checks = {'inputs_unchanged': actual_input_hashes(ROOT / args.data) == inputs_before,
                  'deployment_unchanged': hashes(deployment) == deployment_before,
                  'sources_unchanged': sources_unchanged(sources),
                  'source_code_unchanged': hashes_unchanged(source_code),
                  'reference_v1_unchanged': hashes_unchanged(reference_hashes)}
        report.update(checks)
        integrity = all(checks.values())
        if not integrity:
            report['status'] = 'error'
        write_json(output / 'report.json', report)
    require(integrity, 'expanded formal provenance changed during training')
    print(f'Completed expanded train/validation only: {output / "report.json"}')


if __name__ == '__main__':
    main()
