"""Frozen 2024/2025 diagnostic only; failed development gates are not waived."""

import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from haze_forecast.conditional_recovery import conditional_log_probability
from haze_forecast.history_state_residual import HistoryStateResidual
from scripts.benchmark_recovery_head import ROOT, duration_records, load_experiment
from scripts.benchmark_regional_air_models import write_json
from scripts.train_conditional_recovery import ARMS, score
from scripts.train_recovery_weather_expanded import hashes_unchanged
from scripts.train_recovery_weather_formal import (
    ARMS as BACKBONES,
)
from scripts.train_recovery_weather_formal import FormalWeatherWindows, actual_input_hashes, sha
from scripts.verify_recovery_auxiliary import independent_scores
from scripts.verify_recovery_weather_formal import tensor_digest


def main():
    started = time.monotonic()
    source = ROOT / 'artifacts/task01_recovery_head/future_weather_expanded_formal_v1'
    conditional = ROOT / 'artifacts/task01_recovery_head/conditional_recovery_v1'
    output = ROOT / 'artifacts/task01_recovery_head/conditional_crossyear_diagnostic_v1'
    assert not output.exists() and torch.cuda.is_available()
    original = json.loads((source / 'report.json').read_bytes())
    corrected = json.loads((conditional / 'report.json').read_bytes())
    verified = json.loads((conditional / 'independent_verification.json').read_bytes())
    assert original['status'] == 'completed_train_validation_only'
    assert corrected['status'] == 'completed_development_comparison'
    assert verified['verification'] == 'PASS'
    assert hashes_unchanged(original['source_code_sha256'])
    assert hashes_unchanged(corrected['source_sha256'])
    tracked = dict(original['source_code_sha256']) | dict(corrected['source_sha256'])
    for path in (source / 'report.json', conditional / 'report.json',
                 conditional / 'independent_verification.json',
                 ROOT / 'scripts/evaluate_conditional_crossyear.py'):
        tracked[str(path.relative_to(ROOT))] = sha(path)
    checkpoints = {'base': source / 'base/selected_state.pt', **{
        arm: conditional / arm / 'selected_state.pt' for arm in ARMS}}
    for arm, path in checkpoints.items():
        report = original if arm == 'base' else corrected
        assert sha(path) == report['arms'][arm]['checkpoint_sha256']
        tracked[str(path.relative_to(ROOT))] = sha(path)
    args = SimpleNamespace(**original['configuration'])
    assert args.validation_year == 2023 and args.history == 24 and args.horizon == 72
    inputs_before = actual_input_hashes(ROOT / args.data)
    assert inputs_before == original['inputs_before']
    torch.set_num_threads(2)
    panel, city, labels, eligible, partitions = load_experiment(args)
    for key in ('mean', 'scale'):
        np.testing.assert_array_equal(getattr(panel['scaler'], key), original['history_scaler'][key])
    base = BACKBONES['base'](len(panel['feature_names']), panel['coords'],
                             panel['scaler'].mean[-2:], panel['scaler'].scale[-2:]).cuda().eval()
    base.load_state_dict(torch.load(checkpoints['base'], weights_only=True))
    base.requires_grad_(False)
    models = {'base': base}
    for arm, nonlinear in ARMS.items():
        model = HistoryStateResidual(nonlinear=nonlinear).cuda().eval()
        model.load_state_dict(torch.load(checkpoints[arm], weights_only=True))
        model.requires_grad_(False)
        models[arm] = model
    digests_before = {arm: tensor_digest(model.state_dict()) for arm, model in models.items()}
    base_dev = corrected['baseline']
    development_checks = {}
    for arm in ARMS:
        metrics = corrected['arms'][arm]['metrics']
        checks = {'row_better': metrics['row_mae'] < base_dev['row_mae'],
                  'event_equal_better': metrics['event_equal_mae'] < base_dev['event_equal_mae'],
                  'tail_not_worse': metrics['tail_event_equal_mae'] <= base_dev['tail_event_equal_mae']}
        development_checks[arm] = {'accepted': all(checks.values()), 'checks': checks}
    assert not any(item['accepted'] for item in development_checks.values())
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running_diagnostic', 'evaluated_years': [2024, 2025],
              'configuration': {'validation_year': 2023, 'history': 24, 'horizon': 72,
                                'optimizer_updates': 0, 'reselection': False},
              'source_sha256': tracked, 'inputs_before': inputs_before,
              'model_digest_before': digests_before, 'development_checks': development_checks,
              'years': {}, 'limitations': [
                  'diagnostic_only_failed_development_gate_not_waived',
                  '2024_and_2025_previously_inspected_not_pristine_holdout',
                  'ERA5_retrospective_weather_not_operational_forecast',
                  'base_and_correction_selected_using_2023',
                  'full_history_loader_reads_input_and_constructs_partitions_through_2026',
                  'no_2026_inference_or_scoring', '68_hour_capped_metrics_not_exact_tail_recovery',
                  'no_training_no_reselection_no_deployment']}
    write_json(output / 'protocol.json', result)
    for year in (2024, 2025):
        origins = partitions[year]
        assert (panel['timestamps'][origins].year == year).all()
        dataset = FormalWeatherWindows(panel, origins, city.aqi.to_numpy(), eligible, {})
        states, logits = [], []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=256, shuffle=False):
                x, _, _, active, _, _, _, weather, mask = batch
                assert not mask.any() and not weather.any()
                _, state = base.forward_with_state(x.cuda(), active.cuda())
                states.append(state)
                logits.append(base.risk_head(state))
            state, logit = torch.cat(states), torch.cat(logits)
            base_probability = conditional_log_probability(logit, torch.zeros_like(logit))
        identity = duration_records(origins, panel, labels, city.aqi.to_numpy(),
                                    np.ones(len(origins)), args.horizon)
        folder = output / str(year)
        folder.mkdir()
        np.savez_compressed(folder / 'states.npz', state=state.cpu().numpy(),
                            logits=logit.cpu().numpy(), target=dataset.hazard,
                            risk_mask=dataset.risk_mask, eligible=dataset.eligible,
                            origin_ns=panel['timestamps'][origins].asi8, origin_indices=origins)
        identity.to_parquet(folder / 'identity.parquet', index=False)
        annual = {'label_audit': dataset.audit(), 'arms': {},
                  'scoring_origin_min': str(panel['timestamps'][origins].min()),
                  'scoring_origin_max': str(panel['timestamps'][origins].max()),
                  'input_year_rows': int((city.index.year == year).sum()),
                  'input_year_aqi_observed': int(city.loc[city.index.year == year, 'aqi'].notna().sum())}
        for arm, model in models.items():
            with torch.no_grad():
                probability = base_probability if arm == 'base' else conditional_log_probability(logit, model(state))
            assert torch.equal(probability[:, -1], base_probability[:, -1])
            frame, metrics = score(probability, identity)
            for key, value in independent_scores(frame).items():
                if key.endswith('mae'):
                    np.testing.assert_allclose(value, metrics[key], atol=1e-12)
            frame.to_parquet(folder / f'{arm}.parquet', index=False)
            np.save(folder / f'{arm}_log_probability.npy', probability.cpu().numpy())
            annual['arms'][arm] = metrics
        annual['output_sha256'] = {path.name: sha(path) for path in folder.iterdir()}
        result['years'][str(year)] = annual
        print(json.dumps({'year': year, **annual}, indent=2), flush=True)
    result['inputs_after'] = actual_input_hashes(ROOT / args.data)
    result['model_digest_after'] = {arm: tensor_digest(model.state_dict()) for arm, model in models.items()}
    result['sources_unchanged'] = hashes_unchanged(tracked)
    assert result['inputs_after'] == inputs_before
    assert result['model_digest_after'] == digests_before and result['sources_unchanged']
    result['status'] = 'completed_frozen_diagnostic_not_promotion'
    result['elapsed_seconds'] = time.monotonic() - started
    write_json(output / 'report.json', result)
    print(f"Completed frozen diagnostic in {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
