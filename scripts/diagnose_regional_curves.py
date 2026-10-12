"""Read-only diagnosis of frozen rolling curves, not a model-selection experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def curve_statistics(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 72:
        raise ValueError('Expected matching N by 72 arrays')
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite prediction')
    valid = np.isfinite(truth)
    observed = valid.sum(axis=1)
    error = np.where(valid, abs(prediction - truth), 0).sum(axis=1)
    mae = np.divide(error, observed, out=np.full(len(truth), np.nan), where=observed > 0)

    def recovery(values):
        windows = np.lib.stride_tricks.sliding_window_view(values, 6, axis=1)
        clean = (windows <= 100).all(axis=2)
        return np.where(clean.any(axis=1), clean.argmax(axis=1) + 1, 68).astype(float)

    predicted = recovery(prediction)
    actual = recovery(truth)
    actual[observed != 72] = np.nan
    false_window = np.full(len(truth), np.nan)
    for row, hour in enumerate(predicted):
        if hour < 68:
            window = truth[row, int(hour) - 1:int(hour) + 5]
            if np.isfinite(window).all():
                false_window[row] = float((window > 100).any())
    return {'observed_hours': observed, 'aqi_mae': mae,
            'predicted_recovery': predicted, 'truth_recovery': actual,
            'false_clean_hours': (valid & (prediction <= 100) & (truth > 100)).sum(axis=1),
            'false_dirty_hours': (valid & (prediction > 100) & (truth <= 100)).sum(axis=1),
            'near_threshold_hours': (valid & (abs(truth - 100) <= 10)).sum(axis=1),
            'false_predicted_window': false_window}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'regional_curve_diagnosis_20260916'
    if out.exists():
        raise FileExistsError(out)
    city_path = ROOT / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
    city = pd.read_parquet(city_path).set_index('timestamp').sort_index()
    if not city.index.is_unique:
        raise ValueError('Duplicate city timestamps')
    hashes = {str(city_path): digest(city_path), str(Path(__file__).resolve()): digest(Path(__file__).resolve())}
    reports, details = {}, {}
    for year in (2022, 2023):
        folder = base / f'regional_rolling_{year}_20260916'
        protocol = json.loads((folder / 'protocol.json').read_bytes())
        frozen = json.loads((folder / 'report.json').read_bytes())
        for path, expected in protocol['input_sha256'].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'Changed frozen input: {path}')
            hashes[path] = expected
        for name, expected in frozen['output_sha256'].items():
            if digest(folder / name) != expected:
                raise ValueError(f'Changed frozen output: {name}')
            hashes[str(folder / name)] = expected
        hashes[str(folder / 'report.json')] = digest(folder / 'report.json')
        frame = pd.read_parquet(folder / 'predictions.parquet')
        with np.load(folder / 'city_curves.npz') as saved:
            np.testing.assert_array_equal(saved['origin_ns'], pd.DatetimeIndex(frame.timestamp).asi8)
            curves = saved['prediction'].copy()
        current = city.reindex(pd.DatetimeIndex(frame.timestamp))
        np.testing.assert_allclose(current.aqi, frame.aqi, equal_nan=True)
        future = pd.DatetimeIndex(np.concatenate([
            pd.date_range(t + pd.Timedelta(hours=1), periods=72, freq='h') for t in frame.timestamp]))
        raw = city.reindex(future)[['pm2_5', 'aqi']].to_numpy().reshape(-1, 72, 2)
        stats = curve_statistics(curves[:, :, 1], raw[:, :, 1])
        np.testing.assert_array_equal(stats['predicted_recovery'], frame.airformer)
        data = frame[['timestamp', 'continuity_block', 'score_mature', 'exact_hours',
                      'airformer', 'matched_aft']].copy()
        for key, value in stats.items():
            data[key] = value
        # Compare the two concentration curves on exactly the same observed cells.
        for index, name in enumerate(('pm25', 'aqi')):
            observed = np.isfinite(raw[:, :, index])
            count = observed.sum(axis=1)
            for model, values in (('airformer', curves[:, :, index]),
                                  ('persistence', np.broadcast_to(current[['pm2_5', 'aqi']].to_numpy()[:, index, None], count.shape + (72,)))):
                error = np.where(observed, abs(values - raw[:, :, index]), 0).sum(axis=1)
                data[f'{model}_{name}_mae'] = np.divide(error, count, out=np.full(len(data), np.nan), where=count > 0)
        data['duration_error'] = abs(data.airformer - data.exact_hours)
        data['aft_duration_error'] = abs(data.matched_aft - data.exact_hours)
        mature = data.loc[data.score_mature]
        complete = mature.loc[mature.observed_hours.eq(72)]
        exact = complete.loc[complete.exact_hours.notna()]
        mismatch = exact.truth_recovery.ne(exact.exact_hours)
        concentrations = {}
        for name in ('pm25', 'aqi'):
            for model in ('airformer', 'persistence'):
                column = f'{model}_{name}_mae'
                concentrations[column] = {
                    'origin_mean': float(complete[column].mean()),
                    'block_mean': float(complete.groupby('continuity_block')[column].mean().mean())}
        reports[str(year)] = {
            'predicted_origins': len(data), 'mature_origins': len(mature),
            'complete_aqi_72h_origins': len(complete),
            'incomplete_aqi_72h_origins': int(mature.observed_hours.ne(72).sum()),
            'complete_exact_origins': len(exact),
            'city_vs_canonical_recovery_mismatches': int(mismatch.sum()),
            'concentration_complete_aqi_subset': concentrations,
            'observed_aqi_cells_with_repeated_timestamps': int(mature.observed_hours.sum()),
            'false_clean_cells': int(mature.false_clean_hours.sum()),
            'false_dirty_cells': int(mature.false_dirty_hours.sum()),
            'near_threshold_cells': int(mature.near_threshold_hours.sum()),
            'predicted_recovery_windows': int(mature.airformer.lt(68).sum()),
            'predicted_windows_with_complete_truth': int(mature.false_predicted_window.notna().sum()),
            'false_predicted_windows': int(mature.false_predicted_window.eq(1).sum()),
            'complete_true_recovery_missed': int((complete.airformer.eq(68) & complete.truth_recovery.lt(68)).sum()),
            'complete_true_recovery_origins': int(complete.truth_recovery.lt(68).sum()),
            'aqi_mae_duration_error_spearman_complete_exact': float(exact[['aqi_mae', 'duration_error']].corr(method='spearman').iloc[0, 1]),
        }
        details[year] = data
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Input changed during audit: {path}')
    out.mkdir()
    for year, data in details.items():
        data.to_parquet(out / f'origins_{year}.parquet', index=False)
        selected = data.loc[data.score_mature & data.exact_hours.notna()]
        selected.groupby('continuity_block').agg(
            origins=('timestamp', 'size'), aqi_mae=('aqi_mae', 'mean'),
            airformer_duration_mae=('duration_error', 'mean'),
            aft_duration_mae=('aft_duration_error', 'mean')).to_csv(out / f'blocks_{year}.csv')
    report = {'years': reports, 'input_sha256': hashes,
              'policy': 'Diagnostic only; no threshold, checkpoint, mixing or deployment changes',
              'limitations': ['Repeated future hours are not independent samples',
                              'City median observed station support differs from origin-fixed prediction support',
                              'Concentration complete-window subset is not the full business scoring cohort',
                              'Persistence is a concentration reference, not the AFT model',
                              'Correlation and near-threshold counts are descriptive, not causal evidence'],
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
