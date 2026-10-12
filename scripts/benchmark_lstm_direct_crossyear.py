"""Frozen, three-seed recurrent duration comparison; never changes deployment."""

import json
import math
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import joblib
import numpy as np
import pandas as pd
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.direct_history_aft import DirectHistoryAFT, fit_scale, transform
from haze_forecast.direct_lstm_aft import DirectLSTMAFT
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.torch_interval_aft import interval_aft_nll
from scripts.benchmark_direct_history_aft import HISTORY, digest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/task01_recovery_head/direct_history_crossyear_20260918'
OUT = ROOT / 'artifacts/task01_recovery_head/lstm_direct_crossyear_20261012'
YEARS = (2024, 2025)
SEEDS = (24, 42, 2026)
ARMS = ('gru_history', 'lstm_history', 'lstm_repeat')
METRICS = ('exact_row_mae', 'exact_block_mae', 'tail24_block_mae',
           'all_row_error_lower', 'all_row_error_upper',
           'all_block_error_lower', 'all_block_error_upper')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def choose_epoch(trace):
    scores = [float(row['selection_block_mae']) for row in trace]
    if not scores or not all(math.isfinite(value) for value in scores):
        raise ValueError('Expected nonempty finite development scores')
    return int(trace[min(range(len(scores)), key=scores.__getitem__)]['epoch'])


def split_members(training, year):
    cut = pd.Timestamp(f'{year - 1}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    if not (training.timestamp.lt(end).all() and training.label_available_at.lt(end).all()):
        raise ValueError('Training origin or label not available before the evaluation year')
    selection = (training.timestamp.ge(cut) & training.label_available_at.lt(end)).to_numpy()
    fitting = (training.timestamp.lt(cut) & training.label_available_at.lt(cut)
               & ~training.continuity_block.isin(training.loc[selection, 'continuity_block']))
    return fitting.to_numpy(), selection


def normalize_arrays(history, snapshot, fitting, repeat=False):
    if repeat:
        history = np.repeat(history[:, -1:], history.shape[1], axis=1)
    if len(fitting) != len(history) or not np.asarray(fitting, dtype=bool).any():
        raise ValueError('Require a nonempty, aligned normalization population')
    hm, hs = fit_scale(history[fitting].reshape(-1, history.shape[-1]))
    xm, xs = fit_scale(snapshot[fitting])
    hk = np.isfinite(history[fitting]).any(axis=(0, 1))
    xk = np.isfinite(snapshot[fitting]).any(axis=0)
    scaler = {'history_mean': hm, 'history_scale': hs, 'history_supported': hk,
              'snapshot_mean': xm, 'snapshot_scale': xs, 'snapshot_supported': xk}
    return transform(history, hm, hs, hk), transform(snapshot, xm, xs, xk), scaler


def make_model(arm, history_dim, snapshot_dim):
    cls = DirectHistoryAFT if arm == 'gru_history' else DirectLSTMAFT
    return cls(history_dim, snapshot_dim).cuda()


def predict(model, history, snapshot):
    model.eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(history), 128):
            h = torch.from_numpy(history[start:start + 128]).cuda()
            x = torch.from_numpy(snapshot[start:start + 128]).cuda()
            pieces.append(model(h, x).double().exp().cpu().numpy())
    result = np.concatenate(pieces)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError('Invalid full-duration prediction')
    return result


def fit(arm, seed, history, snapshot, lower, upper, epochs, validation=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(arm, history.shape[-1], snapshot.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    tensors = [torch.as_tensor(a, device='cuda') for a in (history, snapshot, lower, upper)]
    generator = torch.Generator().manual_seed(seed)
    trace, best, best_state, best_epoch = [], math.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(history), generator=generator).cuda()
        total = 0.
        for ids in permutation.split(128):
            optimizer.zero_grad(set_to_none=True)
            mu = model(tensors[0][ids], tensors[1][ids])
            loss = interval_aft_nll(mu, tensors[2][ids], tensors[3][ids], scale=1.2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        entry = {'epoch': epoch, 'training_nll': total / len(history)}
        if validation is not None:
            hv, xv, frame = validation
            score = score_predictions(frame, predict(model, hv, xv))['exact_block_mae']
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite development score')
            entry['selection_block_mae'] = score
            if score < best:
                best, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
        trace.append(entry)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({'arm': arm, 'seed': seed, **entry}), flush=True)
        if validation is not None and epoch - best_epoch >= 8:
            break
    if validation is not None:
        assert choose_epoch(trace) == best_epoch and best_state is not None
        model.load_state_dict(best_state)
    return model, trace, best_epoch


def summarize(runs):
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for metric in METRICS:
            values = [runs[f'{arm}_seed{seed}']['scores'][metric] for seed in SEEDS]
            result[arm][metric] = {'mean': float(np.mean(values)),
                                   'min': min(values), 'max': max(values)}
    return result


def exact_gate(native, runs, summaries):
    mean = summaries['lstm_history']
    improvements = sum(all(runs[f'lstm_history_seed{s}']['scores'][k] < native[k]
                           for k in ('exact_row_mae', 'exact_block_mae')) for s in SEEDS)
    return bool(mean['exact_row_mae']['mean'] < native['exact_row_mae']
                and mean['exact_block_mae']['mean'] < native['exact_block_mae']
                and mean['tail24_block_mae']['mean'] <= native['tail24_block_mae']
                and improvements >= 2)


def preflight_year(year, names):
    source = OLD / str(year)
    training = pd.read_parquet(source / 'training.parquet')
    predictions = pd.read_parquet(source / 'all_predictions.parquet')
    cohort = pd.read_parquet(source / 'scored_predictions.parquet')
    fitting, selection = split_members(training, year)
    expected = {2024: (6770, 396, 7166, 335), 2025: (7166, 335, 7624, 266)}[year]
    assert (int(fitting.sum()), int(selection.sum()), len(training), len(predictions)) == expected
    assert not set(training.continuity_block) & set(cohort.continuity_block)
    assert predictions.timestamp.is_unique and set(predictions.timestamp) == set(cohort.timestamp)
    for mask, filename in ((fitting, 'selection_training.parquet'),
                           (selection, 'selection_rows.parquet')):
        pd.testing.assert_frame_equal(training.loc[mask].reset_index(drop=True),
                                      pd.read_parquet(source / filename).reset_index(drop=True))
    raw = np.load(source / 'raw_inputs.npz', allow_pickle=False)
    total = len(training) + len(predictions)
    assert raw['history'].shape == (total, 24, 12) and raw['snapshot'].shape == (total, 104)
    times = pd.DatetimeIndex(pd.concat([training.timestamp, predictions.timestamp]))
    np.testing.assert_array_equal(raw['timestamps'], times.asi8)
    np.testing.assert_allclose(raw['history'][:, -1],
        raw['snapshot'][:, [names.index(n) for n in HISTORY]], rtol=0, atol=0, equal_nan=True)
    baseline = ROOT / f'artifacts/task01_recovery_head/reusable_gefs_crossyear_v1/{year}.joblib'
    native = joblib.load(baseline).predict(pd.DataFrame(raw['snapshot'][len(training):],
                                                      columns=names))
    np.testing.assert_array_equal(native, predictions.native)
    print(json.dumps({'year': year, 'preflight': 'passed', 'population': len(predictions)}),
          flush=True)


def run_year(year, names):
    source, target = OLD / str(year), OUT / str(year)
    target.mkdir()
    for name in ('training.parquet', 'selection_training.parquet', 'selection_rows.parquet',
                 'raw_inputs.npz'):
        shutil.copy2(source / name, target / name)
    training = pd.read_parquet(target / 'training.parquet')
    cohort = pd.read_parquet(source / 'scored_predictions.parquet').drop(
        columns=['snapshot', 'history'])
    predictions = pd.read_parquet(source / 'all_predictions.parquet')[['timestamp', 'native']]
    raw = np.load(target / 'raw_inputs.npz', allow_pickle=False)
    history, snapshot = raw['history'], raw['snapshot']
    total, ntrain = len(training) + len(predictions), len(training)
    assert history.shape == (total, 24, 12) and snapshot.shape == (total, 104)
    times = pd.DatetimeIndex(pd.concat([training.timestamp, predictions.timestamp]))
    np.testing.assert_array_equal(raw['timestamps'], times.asi8)
    assert training.timestamp.is_unique and predictions.timestamp.is_unique
    assert set(cohort.timestamp) == set(predictions.timestamp)
    np.testing.assert_allclose(history[:, -1], snapshot[:, [names.index(n) for n in HISTORY]],
                               rtol=0, atol=0, equal_nan=True)
    assert not set(training.continuity_block) & set(cohort.continuity_block)
    fitting, selection = split_members(training, year)
    expected = {2024: (6770, 396, 7166, 335), 2025: (7166, 335, 7624, 266)}[year]
    assert (int(fitting.sum()), int(selection.sum()), ntrain, len(predictions)) == expected
    for mask, filename in ((fitting, 'selection_training.parquet'),
                           (selection, 'selection_rows.parquet')):
        pd.testing.assert_frame_equal(training.loc[mask].reset_index(drop=True),
                                      pd.read_parquet(target / filename).reset_index(drop=True))
    native = score_predictions(cohort, cohort.native.to_numpy())
    np.testing.assert_array_equal(predictions.native,
                                  cohort.set_index('timestamp').loc[predictions.timestamp].native)
    normalizations = {}
    for variant, repeat in (('history', False), ('repeat', True)):
        for phase, train_mask in (('selection_', fitting), ('', np.ones(ntrain, dtype=bool))):
            mask = np.concatenate([train_mask, np.zeros(len(predictions), dtype=bool)])
            h, x, scaler = normalize_arrays(history, snapshot, mask, repeat)
            np.savez_compressed(target / f'{phase}{variant}_inputs.npz',
                                history=h, snapshot=x, **scaler)
            normalizations[phase + variant] = (h, x)
    runs = {}
    for seed in SEEDS:
        for arm in ARMS:
            prefix = f'{arm}_seed{seed}'
            variant = 'repeat' if arm == 'lstm_repeat' else 'history'
            sh, sx = normalizations['selection_' + variant]
            print(json.dumps({'year': year, 'run': prefix, 'phase': 'selection'}), flush=True)
            model, trace, epochs = fit(arm, seed, sh[:ntrain][fitting], sx[:ntrain][fitting],
                training.lower_bound_hours.to_numpy()[fitting],
                training.upper_bound_hours.to_numpy()[fitting], 60,
                (sh[:ntrain][selection], sx[:ntrain][selection], training.loc[selection]))
            torch.save(model.state_dict(), target / f'{prefix}_selection.pt')
            development_score = score_predictions(training.loc[selection],
                predict(model, sh[:ntrain][selection], sx[:ntrain][selection]))['exact_block_mae']
            np.testing.assert_allclose(development_score,
                trace[epochs - 1]['selection_block_mae'], rtol=1e-7, atol=1e-7)
            sh, sx = normalizations[variant]
            print(json.dumps({'year': year, 'run': prefix, 'phase': 'refit',
                              'epochs': epochs}), flush=True)
            model, refit_trace, _ = fit(arm, seed, sh[:ntrain], sx[:ntrain],
                training.lower_bound_hours.to_numpy(), training.upper_bound_hours.to_numpy(), epochs)
            prediction = predict(model, sh[ntrain:], sx[ntrain:])
            torch.save(model.state_dict(), target / f'{prefix}.pt')
            restored = make_model(arm, sh.shape[-1], sx.shape[-1])
            restored.load_state_dict(torch.load(target / f'{prefix}.pt',
                                     map_location='cuda', weights_only=True))
            np.testing.assert_array_equal(prediction, predict(restored, sh[ntrain:], sx[ntrain:]))
            predictions[prefix] = prediction
            cohort[prefix] = pd.Series(prediction, index=predictions.timestamp).loc[
                cohort.timestamp].to_numpy()
            run = {'year': year, 'arm': arm, 'seed': seed, 'selected_epoch': epochs,
                   'selection_trace': trace, 'refit_trace': refit_trace,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'development_replay_mae': development_score,
                   'scores': score_predictions(cohort, cohort[prefix].to_numpy()),
                   'paired_native': paired_error_summary(cohort, cohort.native, cohort[prefix])}
            write_json(target / f'{prefix}_run.json', run)
            runs[prefix] = run
            print(json.dumps({'year': year, 'run': prefix, 'scores': run['scores']}), flush=True)
            del model, restored
            torch.cuda.empty_cache()
    predictions.to_parquet(target / 'all_predictions.parquet', index=False)
    cohort.to_parquet(target / 'scored_predictions.parquet', index=False)
    summaries = summarize(runs)
    comparisons = {str(seed): paired_error_summary(cohort, cohort[f'lstm_repeat_seed{seed}'],
                                                    cohort[f'lstm_history_seed{seed}'])
                   for seed in SEEDS}
    result = {'year': year, 'native': native, 'runs': runs, 'summaries': summaries,
              'lstm_history_vs_repeat': comparisons,
              'exact_advantage': exact_gate(native, runs, summaries),
              'deployed': False}
    write_json(target / 'report.json', result)
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if not torch.cuda.is_available():
        raise RuntimeError('This frozen run requires CUDA')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    frozen = json.loads((OLD / 'protocol.json').read_bytes())
    hashes = {str(OLD / 'protocol.json'): digest(OLD / 'protocol.json')}
    for year in YEARS:
        path = ROOT / f'artifacts/task01_recovery_head/reusable_gefs_crossyear_v1/{year}.joblib'
        expected = frozen['input_sha256'][str(path)]
        assert digest(path) == expected, path
        hashes[str(path)] = expected
    for year in YEARS:
        report_path = OLD / str(year) / 'report.json'
        hashes[str(report_path)] = digest(report_path)
        report = json.loads(report_path.read_bytes())
        for name, expected in report['output_sha256'].items():
            path = OLD / str(year) / name
            assert digest(path) == expected, path
            hashes[str(path)] = expected
        preflight_year(year, frozen['features'])
    for relative in ('scripts/benchmark_lstm_direct_crossyear.py',
                     'src/haze_forecast/direct_lstm_aft.py',
                     'src/haze_forecast/direct_history_aft.py',
                     'src/haze_forecast/torch_interval_aft.py',
                     'src/haze_forecast/cohort_scorecard.py',
                     'src/haze_forecast/paired_interval_score.py',
                     'docs/superpowers/specs/2026-10-12-lstm-direct-duration-design.md'):
        hashes[str(ROOT / relative)] = digest(ROOT / relative)
    deployment = {str(path): digest(path) for path in
                  (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir() if path.is_file()}
    deployment[str(ROOT / 'src/haze_forecast/showcase_server.py')] = digest(
        ROOT / 'src/haze_forecast/showcase_server.py')
    OUT.mkdir()
    write_json(OUT / 'protocol.json', {'input_sha256': hashes, 'deployment_sha256': deployment,
        'old_archive': str(OLD), 'features': frozen['features'], 'history_features': HISTORY,
        'years': YEARS, 'seeds': SEEDS, 'arms': ARMS, 'batch_size': 128, 'scale': 1.2,
        'max_epochs': 60, 'patience': 8, 'device': 'cuda', 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'device_name': torch.cuda.get_device_name(0),
        'approval': 'User approved design and frozen 601-origin scope on 2026-10-12',
        'target': 'full remaining hours, not capped', 'deployed': False})
    reports = {str(year): run_year(year, frozen['features']) for year in YEARS}
    for path, expected in {**hashes, **deployment}.items():
        assert digest(Path(path)) == expected, path
    for report in reports.values():
        report['all_input_hashes_unchanged'] = True
        write_json(OUT / str(report['year']) / 'report.json', report)
    result = {'years': reports, 'both_years_exact_advantage': all(
        report['exact_advantage'] for report in reports.values()), 'deployed': False,
        'all_input_hashes_unchanged': True,
        'output_sha256': {str(path.relative_to(OUT)): digest(path)
                          for path in OUT.rglob('*') if path.is_file()}}
    write_json(OUT / 'report.json', result)
    print(json.dumps({'both_years_exact_advantage': result['both_years_exact_advantage'],
                      'deployed': False}), flush=True)


if __name__ == '__main__':
    main()
