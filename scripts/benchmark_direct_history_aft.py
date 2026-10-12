"""Select direct-duration training epochs in 2022, refit, then evaluate 2023."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.direct_history_aft import DirectHistoryAFT, fit_scale, transform
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.torch_interval_aft import interval_aft_nll

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ['aqi', 'pm2_5', 'pm10', 'so2', 'no2', 'co', 'o3', 'temperature_c',
           'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms', 'boundary_layer_height_m']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fit(h, x, lower, upper, epochs, validation=None):
    torch.manual_seed(42)
    np.random.seed(42)
    model = DirectHistoryAFT(h.shape[-1], x.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    tensors = [torch.as_tensor(a, device='cuda') for a in (h, x, lower, upper)]
    generator = torch.Generator().manual_seed(42)
    trace, best, best_epoch = [], float('inf'), 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(h), generator=generator).cuda()
        losses = []
        for ids in permutation.split(128):
            optimizer.zero_grad(set_to_none=True)
            mu = model(tensors[0][ids], tensors[1][ids])
            loss = interval_aft_nll(mu, tensors[2][ids], tensors[3][ids]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append((float(loss.detach()), len(ids)))
        entry = {'epoch': epoch, 'training_nll': sum(v*n for v, n in losses)/len(h)}
        if validation is not None:
            hv, xv, frame = validation
            pred = predict(model, hv, xv)
            score = score_predictions(frame, pred)['exact_block_mae']
            entry['selection_block_mae'] = score
            if score < best:
                best, best_epoch = score, epoch
        trace.append(entry)
        if epoch % 10 == 0:
            print(json.dumps(entry), flush=True)
        if validation is not None and epoch - best_epoch >= 8:
            break
    return model, trace, best_epoch


def predict(model, h, x):
    model.eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(h), 128):
            mu = model(torch.from_numpy(h[start:start+128]).cuda(), torch.from_numpy(x[start:start+128]).cuda())
            pieces.append(torch.exp(mu.double()).cpu().numpy())
    result = np.concatenate(pieces)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError('Invalid duration')
    return result


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'direct_history_aft_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    frozen = json.loads((base / 'grouped_forecast_aft_2023_20260916/protocol.json').read_bytes())
    paths = [base / 'gefs_2021_extension_v1/f024_training_rows.parquet',
             base / 'gefs_2021_extension_v1/inference_rows.parquet',
             base / 'grouped_forecast_aft_2023_20260916/expanded.parquet',
             base / 'observed_business_dataset_v1/features.parquet',
             Path(__file__).resolve(), ROOT / 'direct_history_aft_plan_20260918.md',
             ROOT / 'src/haze_forecast/direct_history_aft.py', ROOT / 'src/haze_forecast/torch_interval_aft.py']
    hashes = dict(frozen['input_sha256'])
    hashes.update({str(p): digest(p) for p in paths})
    for p, expected in hashes.items():
        assert digest(Path(p)) == expected, p
    train, valid, cohort, hourly = [pd.read_parquet(p) for p in paths[:4]]
    names = frozen['features']
    combined = pd.concat([train, valid], ignore_index=True)
    times = pd.DatetimeIndex(hourly.timestamp)
    assert times.is_unique and times.is_monotonic_increasing
    positions = times.get_indexer(combined.timestamp)
    assert (positions >= 23).all()
    for pos in positions:
        assert np.all(np.diff(times[pos-23:pos+1].asi8) == 3600000000000)
    raw = hourly[HISTORY].to_numpy(float)
    h = np.stack([raw[pos-23:pos+1] for pos in positions])
    np.testing.assert_allclose(h[:, -1], combined[HISTORY], rtol=0, atol=0, equal_nan=True)
    x = combined[names].to_numpy(float)
    cut = pd.Timestamp('2022-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    select = train.timestamp.ge(cut).to_numpy() & train.label_available_at.lt(end).to_numpy()
    fit_rows = (train.timestamp.lt(cut) & train.label_available_at.lt(cut)
                & ~train.continuity_block.isin(train.loc[select, 'continuity_block'])).to_numpy()
    assert train.label_available_at.lt(end).all() and train.timestamp.lt(end).all()
    assert not set(train.continuity_block) & set(cohort.continuity_block)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    out.mkdir()
    protocol = {'input_sha256': hashes, 'features': names, 'history_features': HISTORY,
                'selection_train_rows': int(fit_rows.sum()), 'selection_rows': int(select.sum()),
                'refit_rows': len(train), 'inference_rows': len(valid), 'seed': 42,
                'scale': 1.2, 'max_epochs': 60, 'patience': 8, 'batch_size': 128,
                'target': 'full remaining hours', 'deployed': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train.to_parquet(out / 'training.parquet', index=False)
    train.loc[fit_rows, ['timestamp', 'continuity_block', 'label_available_at']].to_parquet(out / 'selection_training_members.parquet', index=False)
    train.loc[select].to_parquet(out / 'selection_rows.parquet', index=False)
    np.savez_compressed(out / 'raw_inputs.npz', history=h, snapshot=x,
                        timestamps=pd.DatetimeIndex(combined.timestamp).asi8)
    scores = {'native': score_predictions(cohort, cohort.forecast_aft.to_numpy())}
    all_predictions = valid[['timestamp']].copy()
    traces = {}
    for arm in ('snapshot', 'history'):
        arm_h = h if arm == 'history' else np.repeat(h[:, -1:], 24, axis=1)
        hm, hs = fit_scale(arm_h[:len(train)][fit_rows].reshape(-1, len(HISTORY)))
        xm, xs = fit_scale(x[:len(train)][fit_rows])
        hsupport = np.isfinite(arm_h[:len(train)][fit_rows]).any(axis=(0, 1))
        xsupport = np.isfinite(x[:len(train)][fit_rows]).any(axis=0)
        sh, sx = transform(arm_h, hm, hs, hsupport), transform(x, xm, xs, xsupport)
        print(json.dumps({'arm': arm, 'phase': 'selection'}), flush=True)
        _, selection_trace, epochs = fit(sh[:len(train)][fit_rows], sx[:len(train)][fit_rows],
            train.lower_bound_hours.to_numpy()[fit_rows], train.upper_bound_hours.to_numpy()[fit_rows], 60,
            (sh[:len(train)][select], sx[:len(train)][select], train.loc[select]))
        assert epochs > 0
        # Epoch count is frozen before refitting or using any 2023 target.
        (out / f'{arm}_selection.json').write_text(json.dumps({'selected_epoch': epochs, 'trace': selection_trace}, indent=2), encoding='utf-8')
        hm, hs = fit_scale(arm_h[:len(train)].reshape(-1, len(HISTORY)))
        xm, xs = fit_scale(x[:len(train)])
        hsupport = np.isfinite(arm_h[:len(train)]).any(axis=(0, 1))
        xsupport = np.isfinite(x[:len(train)]).any(axis=0)
        sh, sx = transform(arm_h, hm, hs, hsupport), transform(x, xm, xs, xsupport)
        np.savez_compressed(out / f'{arm}_inputs.npz', history=sh, snapshot=sx,
                            history_mean=hm, history_scale=hs, snapshot_mean=xm, snapshot_scale=xs,
                            history_supported=hsupport, snapshot_supported=xsupport)
        print(json.dumps({'arm': arm, 'phase': 'refit', 'epochs': epochs}), flush=True)
        model, trace, _ = fit(sh[:len(train)], sx[:len(train)], train.lower_bound_hours.to_numpy(),
                              train.upper_bound_hours.to_numpy(), epochs)
        torch.save(model.cpu().state_dict(), out / f'{arm}.pt')
        model.cuda()
        prediction = predict(model, sh[len(train):], sx[len(train):])
        restored = DirectHistoryAFT(24, 208).cuda()
        restored.load_state_dict(torch.load(out / f'{arm}.pt', map_location='cuda', weights_only=True))
        np.testing.assert_array_equal(prediction, predict(restored, sh[len(train):], sx[len(train):]))
        all_predictions[arm] = prediction
        cohort[arm] = pd.Series(prediction, index=valid.timestamp).loc[cohort.timestamp].to_numpy()
        scores[arm] = score_predictions(cohort, cohort[arm].to_numpy())
        traces[arm] = {'selected_epochs': epochs, 'refit': trace,
                       'parameters': sum(p.numel() for p in model.parameters())}
        print(json.dumps({'arm': arm, 'scores': scores[arm]}), flush=True)
    paired = {name: paired_error_summary(cohort, cohort['forecast_aft' if name == 'native' else name], cohort.history)
              for name in ('native', 'snapshot')}
    gates = {name: all(scores['history'][k] < scores[name][k] for k in ('exact_row_mae', 'exact_block_mae'))
             and scores['history']['tail24_block_mae'] <= scores[name]['tail24_block_mae']
             and paired[name]['all_row_delta_lower'] <= 0 and paired[name]['all_block_delta_lower'] <= 0
             for name in ('native', 'snapshot')}
    cohort.to_parquet(out / 'scored_predictions.parquet', index=False)
    all_predictions.to_parquet(out / 'all_predictions.parquet', index=False)
    for p, expected in hashes.items():
        assert digest(Path(p)) == expected, p
    report = {'scores': scores, 'paired': paired, 'gates': gates, 'passed_development': all(gates.values()),
              'training': traces, 'deployed': False,
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'gates': gates, 'paired': paired}), flush=True)


if __name__ == '__main__':
    main()
