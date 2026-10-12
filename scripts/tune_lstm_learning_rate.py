"""Select one learning rate on 2023 development, then freeze annual evaluation."""

import json
import math
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from scripts.benchmark_lstm_direct_crossyear import (
    ARMS,
    METRICS,
    ROOT,
    SEEDS,
    YEARS,
    choose_epoch,
    digest,
    exact_gate,
    predict,
    split_members,
    write_json,
)
from scripts.lstm_lr_training import fit_lstm

PREVIOUS = ROOT / 'artifacts/task01_recovery_head/lstm_direct_crossyear_20261012'
OUT = ROOT / 'artifacts/task01_recovery_head/lstm_lr_tuning_20261012'
LEARNING_RATES = (.0003, .001, .003)


def rate_id(rate):
    return 'lr_' + str(rate).replace('.', 'p')


def select_learning_rate(records):
    expected = {(rate, seed) for rate in LEARNING_RATES for seed in SEEDS}
    keyed = {}
    for row in records:
        key = (row['learning_rate'], row['seed'])
        if key not in expected or key in keyed:
            raise ValueError('Unexpected or duplicate learning-rate candidate')
        score = float(row['scores']['exact_block_mae'])
        if not math.isfinite(score):
            raise ValueError('Development score must be finite')
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError('Require all nine development candidates')
    means = {rate: float(np.mean([keyed[rate, seed]['scores']['exact_block_mae']
                                 for seed in SEEDS])) for rate in LEARNING_RATES}
    selected = min(LEARNING_RATES, key=lambda rate:
                   (means[rate], abs(math.log(rate / .001)), rate))
    return {'selected_learning_rate': selected, 'selected_lr_id': rate_id(selected),
            'development_mean_scores': {rate_id(rate): means[rate] for rate in LEARNING_RATES},
            'selected_epochs': {str(seed): keyed[selected, seed]['selected_epoch'] for seed in SEEDS},
            'metric': 'exact_block_mae', 'learning_rates': list(LEARNING_RATES),
            'seeds': list(SEEDS)}


def development_inputs(year):
    folder = PREVIOUS / str(year)
    train = pd.read_parquet(folder / 'training.parquet')
    fitting, selection = split_members(train, year)
    expected = {2024: (6770, 396, 7166), 2025: (7166, 335, 7624)}[year]
    assert (int(fitting.sum()), int(selection.sum()), len(train)) == expected
    arrays = np.load(folder / 'selection_history_inputs.npz', allow_pickle=False)
    return train, fitting, selection, arrays['history'], arrays['snapshot']


def check_original_candidate(record, model):
    seed = record['seed']
    source = PREVIOUS / '2024' / f'lstm_history_seed{seed}'
    old = json.loads(source.with_name(source.name + '_run.json').read_bytes())
    assert record['selected_epoch'] == old['selected_epoch']
    assert len(record['selection_trace']) == len(old['selection_trace'])
    for new_row, old_row in zip(record['selection_trace'], old['selection_trace'], strict=True):
        assert new_row['epoch'] == old_row['epoch']
        for field in ('training_nll', 'selection_block_mae'):
            np.testing.assert_allclose(new_row[field], old_row[field], rtol=1e-12, atol=1e-10)
    old_weights = torch.load(source.with_name(source.name + '_selection.pt'),
                             map_location='cuda', weights_only=True)
    assert model.state_dict().keys() == old_weights.keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, old_weights[key]), key


def run_candidates():
    folder = OUT / 'candidates'
    folder.mkdir()
    train, fitting, selection, history, snapshot = development_inputs(2024)
    ntrain = len(train)
    development = train.loc[selection].copy()
    development_predictions = development.copy()
    records = {}
    # No final evaluation frame is loaded until this complete grid is frozen.
    for rate in LEARNING_RATES:
        for seed in SEEDS:
            prefix = f'{rate_id(rate)}_seed{seed}'
            print(json.dumps({'phase': 'development_2023', 'candidate': prefix}), flush=True)
            model, trace, epochs = fit_lstm(seed, history[:ntrain][fitting],
                snapshot[:ntrain][fitting], train.lower_bound_hours.to_numpy()[fitting],
                train.upper_bound_hours.to_numpy()[fitting], 60, rate,
                (history[:ntrain][selection], snapshot[:ntrain][selection], development))
            prediction = predict(model, history[:ntrain][selection], snapshot[:ntrain][selection])
            scores = score_predictions(development, prediction)
            np.testing.assert_allclose(scores['exact_block_mae'],
                trace[epochs - 1]['selection_block_mae'], rtol=1e-7, atol=1e-7)
            record = {'seed': seed, 'learning_rate': rate, 'lr_id': rate_id(rate),
                      'selected_epoch': epochs, 'selection_trace': trace, 'scores': scores,
                      'development_replay_mae': scores['exact_block_mae'],
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'budget_reached': len(trace) == 60}
            if rate == .001:
                check_original_candidate(record, model)
                record['original_configuration_reproduced'] = True
            torch.save(model.state_dict(), folder / f'{prefix}.pt')
            write_json(folder / f'{prefix}_run.json', record)
            development_predictions[prefix] = prediction
            records[prefix] = record
            print(json.dumps({'candidate': prefix, 'development_mae': scores['exact_block_mae'],
                              'selected_epoch': epochs}), flush=True)
            del model
            torch.cuda.empty_cache()
    development_predictions.to_parquet(folder / 'development_predictions.parquet', index=False)
    write_json(folder / 'report.json', {'runs': records})
    decision = select_learning_rate(list(records.values()))
    decision['before_evaluation'] = True
    decision['retrospective_exploration'] = True
    decision['source_sha256'] = json.loads((OUT / 'protocol.json').read_bytes())['input_sha256']
    decision['protocol_sha256'] = digest(OUT / 'protocol.json')
    write_json(OUT / 'decision.json', decision)
    print(json.dumps({'phase': 'decision_frozen', **decision}), flush=True)
    return decision, digest(OUT / 'decision.json'), records


def summary(scores):
    return {metric: {'mean': float(np.mean([scores[str(seed)][metric] for seed in SEEDS])),
                     'min': min(scores[str(seed)][metric] for seed in SEEDS),
                     'max': max(scores[str(seed)][metric] for seed in SEEDS)} for metric in METRICS}


def evaluate_year(year, decision, decision_hash, candidates):
    assert digest(OUT / 'decision.json') == decision_hash
    rate, folder = decision['selected_learning_rate'], OUT / str(year)
    folder.mkdir()
    train, fitting, selection, sh, sx = development_inputs(year)
    ntrain = len(train)
    final_arrays = np.load(PREVIOUS / str(year) / 'history_inputs.npz', allow_pickle=False)
    history, snapshot = final_arrays['history'], final_arrays['snapshot']
    old_cohort = pd.read_parquet(PREVIOUS / str(year) / 'scored_predictions.parquet')
    cohort = old_cohort.drop(columns=[f'{arm}_seed{seed}' for arm in ARMS for seed in SEEDS])
    old_predictions = pd.read_parquet(PREVIOUS / str(year) / 'all_predictions.parquet')
    predictions = old_predictions[['timestamp', 'native']].copy()
    expected_rows = {2024: 335, 2025: 266}[year]
    assert len(predictions) == len(cohort) == expected_rows
    assert predictions.timestamp.is_unique and set(predictions.timestamp) == set(cohort.timestamp)
    assert history.shape == (ntrain + expected_rows, 24, 24)
    assert snapshot.shape == (ntrain + expected_rows, 208)
    assert not set(train.continuity_block) & set(cohort.continuity_block)
    original_scores = {}
    for seed in SEEDS:
        column = f'original_lstm_seed{seed}'
        cohort[column] = old_cohort[f'lstm_history_seed{seed}']
        predictions[column] = old_predictions[f'lstm_history_seed{seed}']
        original_scores[str(seed)] = score_predictions(cohort, cohort[column].to_numpy())
    native = score_predictions(cohort, cohort.native.to_numpy())
    runs, tuned_scores = {}, {}
    for seed in SEEDS:
        prefix = f'lstm_tuned_seed{seed}'
        if year == 2024:
            candidate_prefix = f'{rate_id(rate)}_seed{seed}'
            candidate = candidates[candidate_prefix]
            epochs, trace = candidate['selected_epoch'], candidate['selection_trace']
            development_mae = candidate['development_replay_mae']
            selection_source = f'candidates/{candidate_prefix}.pt'
        else:
            print(json.dumps({'phase': 'development_2024', 'run': prefix, 'lr': rate}), flush=True)
            model, trace, epochs = fit_lstm(seed, sh[:ntrain][fitting], sx[:ntrain][fitting],
                train.lower_bound_hours.to_numpy()[fitting], train.upper_bound_hours.to_numpy()[fitting],
                60, rate, (sh[:ntrain][selection], sx[:ntrain][selection], train.loc[selection]))
            development_mae = score_predictions(train.loc[selection],
                predict(model, sh[:ntrain][selection], sx[:ntrain][selection]))['exact_block_mae']
            selection_source = f'{year}/{prefix}_selection.pt'
            torch.save(model.state_dict(), OUT / selection_source)
            del model
        assert epochs == choose_epoch(trace)
        np.testing.assert_allclose(development_mae, trace[epochs - 1]['selection_block_mae'],
                                   rtol=1e-7, atol=1e-7)
        print(json.dumps({'phase': 'refit', 'year': year, 'run': prefix, 'lr': rate,
                          'epochs': epochs}), flush=True)
        model, refit_trace, _ = fit_lstm(seed, history[:ntrain], snapshot[:ntrain],
            train.lower_bound_hours.to_numpy(), train.upper_bound_hours.to_numpy(), epochs, rate)
        prediction = predict(model, history[ntrain:], snapshot[ntrain:])
        torch.save(model.state_dict(), folder / f'{prefix}.pt')
        predictions[prefix] = prediction
        cohort[prefix] = pd.Series(prediction, index=predictions.timestamp).loc[cohort.timestamp].to_numpy()
        scores = score_predictions(cohort, cohort[prefix].to_numpy())
        run = {'year': year, 'seed': seed, 'learning_rate': rate, 'selected_epoch': epochs,
               'selection_source': selection_source, 'selection_trace': trace,
               'refit_trace': refit_trace, 'parameters': sum(p.numel() for p in model.parameters()),
               'development_replay_mae': development_mae, 'scores': scores,
               'paired_native': paired_error_summary(cohort, cohort.native, cohort[prefix]),
               'paired_original_lstm': paired_error_summary(cohort,
                    cohort[f'original_lstm_seed{seed}'], cohort[prefix])}
        if rate == .001:
            np.testing.assert_array_equal(prediction, old_predictions[f'lstm_history_seed{seed}'])
            run['original_configuration_reproduced'] = True
        write_json(folder / f'{prefix}_run.json', run)
        runs[prefix], tuned_scores[str(seed)] = run, scores
        print(json.dumps({'year': year, 'run': prefix, 'scores': scores}), flush=True)
        del model
        torch.cuda.empty_cache()
    cohort.to_parquet(folder / 'scored_predictions.parquet', index=False)
    predictions.to_parquet(folder / 'all_predictions.parquet', index=False)
    tuned_summary = summary(tuned_scores)
    gate_runs = {f'lstm_history_seed{seed}': runs[f'lstm_tuned_seed{seed}'] for seed in SEEDS}
    report = {'year': year, 'native': native, 'original_lstm': original_scores, 'runs': runs,
              'summary': tuned_summary, 'original_summary': summary(original_scores),
              'exact_advantage': exact_gate(native, gate_runs, {'lstm_history': tuned_summary}),
              'deployed': False}
    write_json(folder / 'report.json', report)
    return report


def frozen_inputs():
    protocol = json.loads((PREVIOUS / 'protocol.json').read_bytes())
    protection = json.loads((PREVIOUS / 'upstream_audit.json').read_bytes())
    verification = json.loads((PREVIOUS / 'verification.json').read_bytes())
    assert verification['passed'] is True
    expected = {**protocol['input_sha256'], **protection['input_sha256']}
    expected[str(ROOT / 'scripts/verify_lstm_direct_crossyear.py')] = verification['verifier_sha256']
    for relative, value in verification['artifact_sha256'].items():
        expected[str(PREVIOUS / relative)] = value
    deployment = protocol['deployment_sha256']
    for path, value in {**expected, **deployment}.items():
        assert digest(Path(path)) == value, path
    for path in PREVIOUS.rglob('*'):
        if path.is_file():
            expected[str(path)] = digest(path)
    for relative in ('scripts/tune_lstm_learning_rate.py', 'scripts/lstm_lr_training.py',
                     'docs/superpowers/specs/2026-10-12-lstm-learning-rate-design.md'):
        expected[str(ROOT / relative)] = digest(ROOT / relative)
    return expected, deployment, protocol


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the frozen numerical protocol')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hashes, deployment, previous_protocol = frozen_inputs()
    OUT.mkdir()
    write_json(OUT / 'protocol.json', {'input_sha256': hashes, 'deployment_sha256': deployment,
        'previous_archive': str(PREVIOUS), 'old_archive': previous_protocol['old_archive'],
        'features': previous_protocol['features'],
        'history_features': previous_protocol['history_features'],
        'lr_values': LEARNING_RATES, 'seeds': SEEDS, 'years': YEARS, 'max_epochs': 60,
        'patience': 8, 'batch_size': 128, 'scale': 1.2, 'weight_decay': .01, 'gradient_clip': 5.,
        'parameter_count': 10177, 'device': 'cuda', 'retrospective_exploration': True,
        'extra_development_budget': True, 'deployed': False})
    decision, decision_hash, candidates = run_candidates()
    reports = {str(year): evaluate_year(year, decision, decision_hash, candidates) for year in YEARS}
    assert digest(OUT / 'decision.json') == decision_hash
    for path, value in {**hashes, **deployment}.items():
        assert digest(Path(path)) == value, path
    report = {'decision_sha256': decision_hash, 'selected_learning_rate': decision['selected_learning_rate'],
              'years': reports, 'both_years_exact_advantage': all(
                  result['exact_advantage'] for result in reports.values()),
              'input_hashes_unchanged': True, 'deployed': False,
              'output_sha256': {str(path.relative_to(OUT)): digest(path)
                                for path in OUT.rglob('*') if path.is_file()}}
    write_json(OUT / 'report.json', report)
    print(json.dumps({'selected_learning_rate': decision['selected_learning_rate'],
                      'both_years_exact_advantage': report['both_years_exact_advantage'],
                      'deployed': False}), flush=True)


if __name__ == '__main__':
    main()
