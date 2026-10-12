"""User-approved fixed annual comparison; no deployment or outcome-based tuning."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.direct_history_aft import DirectHistoryAFT, fit_scale, transform
from haze_forecast.paired_interval_score import paired_error_summary
from scripts.benchmark_direct_history_aft import HISTORY, ROOT, digest, fit, predict


def run_year(year, out, names, hourly, canonical):
    base = ROOT / 'artifacts/task01_recovery_head'
    train = pd.read_parquet(base / f'original_gefs_without_pbl_crossyear_v1/{year}_training_rows.parquet')
    valid = pd.read_parquet(base / f'reusable_gefs_crossyear_v1/{year}_features.parquet')
    old = pd.read_parquet(base / f'reusable_gefs_crossyear_v1/{year}_predictions.parquet')
    valid['timestamp'] = old.timestamp.to_numpy()
    cohort = canonical.loc[canonical.timestamp.dt.year.eq(year)].copy()
    assert set(cohort.timestamp) == set(valid.timestamp) and valid.timestamp.is_unique
    cut = pd.Timestamp(f'{year-1}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(end).all() and train.label_available_at.lt(end).all()
    assert not set(train.continuity_block) & set(cohort.continuity_block)
    labels = canonical.set_index('timestamp').loc[train.timestamp]
    for col in ('lower_bound_hours', 'upper_bound_hours', 'exact_hours'):
        np.testing.assert_array_equal(train[col], labels[col])
    assert cohort.label_available_at.lt(end + pd.DateOffset(years=1)).all()
    native = joblib.load(base / f'reusable_gefs_crossyear_v1/{year}.joblib').predict(valid[names])
    np.testing.assert_array_equal(native, old.forecast_aft)
    cohort['native'] = pd.Series(native, index=valid.timestamp).loc[cohort.timestamp].to_numpy()
    combined = pd.concat([train, valid], ignore_index=True)
    times = pd.DatetimeIndex(hourly.timestamp)
    positions = times.get_indexer(combined.timestamp)
    assert (positions >= 23).all() and times.is_unique
    for pos in positions:
        assert np.all(np.diff(times[pos-23:pos+1].asi8) == 3600000000000)
    raw = hourly[HISTORY].to_numpy(float)
    h = np.stack([raw[pos-23:pos+1] for pos in positions])
    np.testing.assert_allclose(h[:, -1], combined[HISTORY].to_numpy(float), rtol=0, atol=0, equal_nan=True)
    x = combined[names].to_numpy(float)
    select = (train.timestamp.ge(cut) & train.label_available_at.lt(end)).to_numpy()
    fitting = (train.timestamp.lt(cut) & train.label_available_at.lt(cut)
               & ~train.continuity_block.isin(train.loc[select, 'continuity_block'])).to_numpy()
    expected = {2024: (6770, 396, 7166, 335), 2025: (7166, 335, 7624, 266)}[year]
    assert (int(fitting.sum()), int(select.sum()), len(train), len(valid)) == expected
    out.mkdir()
    train.to_parquet(out / 'training.parquet', index=False)
    train.loc[fitting].to_parquet(out / 'selection_training.parquet', index=False)
    train.loc[select].to_parquet(out / 'selection_rows.parquet', index=False)
    np.savez_compressed(out / 'raw_inputs.npz', history=h, snapshot=x, timestamps=pd.DatetimeIndex(combined.timestamp).asi8)
    predictions = valid[['timestamp']].copy()
    predictions['native'] = native
    scores = {'native': score_predictions(cohort, cohort.native.to_numpy())}
    traces = {}
    for arm in ('snapshot', 'history'):
        ah = h if arm == 'history' else np.repeat(h[:, -1:], 24, axis=1)

        def normalize(rows, ah=ah):
            hm, hs = fit_scale(ah[:len(train)][rows].reshape(-1, 12))
            xm, xs = fit_scale(x[:len(train)][rows])
            hk = np.isfinite(ah[:len(train)][rows]).any(axis=(0, 1))
            xk = np.isfinite(x[:len(train)][rows]).any(axis=0)
            return transform(ah, hm, hs, hk), transform(x, xm, xs, xk), {
                'history_mean': hm, 'history_scale': hs, 'history_supported': hk,
                'snapshot_mean': xm, 'snapshot_scale': xs, 'snapshot_supported': xk}

        sh, sx, _ = normalize(fitting)
        print(json.dumps({'year': year, 'arm': arm, 'phase': 'selection'}), flush=True)
        _, selection_trace, epochs = fit(sh[:len(train)][fitting], sx[:len(train)][fitting],
            train.lower_bound_hours.to_numpy()[fitting], train.upper_bound_hours.to_numpy()[fitting], 60,
            (sh[:len(train)][select], sx[:len(train)][select], train.loc[select]))
        (out / f'{arm}_selection.json').write_text(json.dumps({'selected_epoch': epochs, 'trace': selection_trace}, indent=2), encoding='utf-8')
        sh, sx, scaler = normalize(np.ones(len(train), dtype=bool))
        np.savez_compressed(out / f'{arm}_inputs.npz', history=sh, snapshot=sx, **scaler)
        print(json.dumps({'year': year, 'arm': arm, 'phase': 'refit', 'epochs': epochs}), flush=True)
        model, trace, _ = fit(sh[:len(train)], sx[:len(train)], train.lower_bound_hours.to_numpy(),
                              train.upper_bound_hours.to_numpy(), epochs)
        torch.save(model.cpu().state_dict(), out / f'{arm}.pt')
        model.cuda()
        p = predict(model, sh[len(train):], sx[len(train):])
        restored = DirectHistoryAFT(24, 208).cuda()
        restored.load_state_dict(torch.load(out / f'{arm}.pt', weights_only=True, map_location='cuda'))
        np.testing.assert_array_equal(p, predict(restored, sh[len(train):], sx[len(train):]))
        predictions[arm] = p
        cohort[arm] = pd.Series(p, index=valid.timestamp).loc[cohort.timestamp].to_numpy()
        scores[arm] = score_predictions(cohort, cohort[arm].to_numpy())
        traces[arm] = {'epochs': epochs, 'trace': trace}
        print(json.dumps({'year': year, 'arm': arm, 'scores': scores[arm]}), flush=True)
    paired = {a: paired_error_summary(cohort, cohort.native, cohort[a]) for a in ('snapshot', 'history')}
    paired['history_vs_snapshot'] = paired_error_summary(cohort, cohort.snapshot, cohort.history)
    gates = {a: all(scores[a][k] < scores['native'][k] for k in ('exact_row_mae', 'exact_block_mae'))
             and scores[a]['tail24_block_mae'] <= scores['native']['tail24_block_mae']
             and paired[a]['all_row_delta_lower'] <= 0 and paired[a]['all_block_delta_lower'] <= 0
             for a in ('snapshot', 'history')}
    cohort.to_parquet(out / 'scored_predictions.parquet', index=False)
    predictions.to_parquet(out / 'all_predictions.parquet', index=False)
    report = {'year': year, 'scores': scores, 'paired': paired, 'gates': gates, 'training': traces,
              'deployed': False, 'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'direct_history_crossyear_20260918'
    if out.exists():
        raise FileExistsError(out)
    old = json.loads((base / 'direct_history_aft_2023_20260918/protocol.json').read_bytes())
    for p, value in old['input_sha256'].items():
        assert digest(Path(p)) == value, p
    names = old['features']
    paths = [Path(__file__).resolve(), ROOT / 'direct_history_crossyear_draft_20260918.md',
             base / 'interval_business_labels_v1/origins.parquet',
             base / 'ghcnh_weather_repair_v1/quality_filtered_weather.parquet',
             base / 'ghcnh_weather_repair_v1/report.json']
    for year in (2024, 2025):
        paths += [base / f'original_gefs_without_pbl_crossyear_v1/{year}_training_rows.parquet']
        paths += [base / f'reusable_gefs_crossyear_v1/{year}{suffix}' for suffix in
                  ('_features.parquet', '_predictions.parquet', '.joblib')]
    hashes = dict(old['input_sha256'])
    hashes.update({str(p): digest(p) for p in paths})
    repair_report = json.loads(paths[4].read_bytes())
    assert digest(paths[3]) == repair_report['output_sha256'][paths[3].name]
    hourly = pd.read_parquet(base / 'observed_business_dataset_v1/features.parquet')
    weather = pd.read_parquet(paths[3]).set_index('timestamp').reindex(pd.DatetimeIndex(hourly.timestamp)).reset_index(drop=True)
    patched = hourly.copy()
    counts = {}
    for col in ('temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms'):
        mask = hourly.timestamp.gt(pd.Timestamp(repair_report['patch_after'])) & hourly[col].isna() & weather[col].notna()
        patched.loc[mask, col] = weather.loc[mask, col]
        counts[col] = int(mask.sum())
    assert counts == repair_report['patched_grid_cells']
    canonical = pd.read_parquet(paths[2])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'features': names,
        'history_features': HISTORY, 'years': [2024, 2025], 'approval': 'User selected option 1 on 2026-09-18',
        'policy': 'Fixed architecture and prior-year epoch selection, no outcome-based tuning',
        'patched_cells': counts, 'deployment_authorized': False}, indent=2), encoding='utf-8')
    patched[['timestamp', *HISTORY]].to_parquet(out / 'hourly_inputs.parquet', index=False)
    reports = {str(y): run_year(y, out / str(y), names, patched if y == 2025 else hourly, canonical) for y in (2024, 2025)}
    for p, value in hashes.items():
        assert digest(Path(p)) == value, p
    result = {'year_gates': {y: r['gates'] for y, r in reports.items()},
              'both_years_pass': {a: all(r['gates'][a] for r in reports.values()) for a in ('snapshot', 'history')},
              'deployed': False}
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
