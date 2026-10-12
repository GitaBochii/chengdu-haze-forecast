"""Common-origin, explicitly capped, interval-aware scoring of frozen regional models."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def curve_recovery(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (72,) or not np.isfinite(values).all():
        raise ValueError('Expected complete 72-hour forecast')
    for start in range(67):
        if np.all(values[start:start + 6] <= 100):
            return start + 1
    return 68


def cap_labels(frame):
    result = frame.copy()
    result['lower_bound_hours'] = np.minimum(frame.lower_bound_hours, 68.)
    result['upper_bound_hours'] = np.minimum(frame.upper_bound_hours, 68.)
    result['exact_hours'] = result.lower_bound_hours.where(
        result.lower_bound_hours.eq(result.upper_bound_hours))
    return result


def main():
    base = ROOT / 'artifacts'
    regional = base / 'regional_air_models'
    source = base / 'task01_recovery_head/unified_scorecard_crossyear_v1'
    out = base / 'task01_recovery_head/regional_common_intervals_2025_20260916'
    if out.exists():
        raise FileExistsError(out)
    folders = {model: regional / 'full_history_2014_2023_v1'
               for model in ('matched_aft', 'pm25_gnn', 'airformer')}
    folders['pm25_gnn_history'] = regional / 'history_structure_seed42_v1'
    cohort_path = source / '2025_original_common.parquet'
    cohort = pd.read_parquet(cohort_path)
    records = {model: pd.read_parquet(folder / f'{model}_2025_recovery.parquet')
               for model, folder in folders.items()}
    paths = [cohort_path, source / 'report.json', Path(__file__).resolve(),
             ROOT / 'src/haze_forecast/cohort_scorecard.py',
             ROOT / 'src/haze_forecast/paired_interval_score.py']
    for model, folder in folders.items():
        paths += [folder / f'{model}_2025_recovery.parquet', folder / 'report.json']
        if model != 'matched_aft':
            paths.append(folder / f'{model}_2025_curves.npz')
    paths += list((base / 'cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    expected = json.loads((source / 'report.json').read_bytes())['output_sha256'][cohort_path.name]
    if hashes[str(cohort_path)] != expected:
        raise ValueError('Frozen canonical cohort changed')
    common = set(cohort.timestamp)
    for record in records.values():
        if not record.timestamp.is_unique:
            raise ValueError('Duplicate saved origin')
        common &= set(record.timestamp)
    uncapped = cohort.loc[cohort.timestamp.isin(common)].copy().reset_index(drop=True)
    if len(uncapped) != 42:
        raise ValueError('Previously inspected common cohort changed')
    capped = cap_labels(uncapped)
    audits, scores, paired = {}, {}, {}
    for model, folder in folders.items():
        saved = records[model].set_index('timestamp').loc[uncapped.timestamp]
        actual = saved.actual_hours.to_numpy()
        known = np.isfinite(actual)
        incompatible = known & ((actual < uncapped.lower_bound_hours.to_numpy())
                                | (actual > uncapped.upper_bound_hours.to_numpy()))
        if incompatible.any():
            raise ValueError('Saved finite recovery incompatible with canonical labels')
        if model == 'matched_aft':
            prediction = saved.predicted_hours.to_numpy()
            if not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise ValueError('Invalid matched AFT output')
            prediction = np.minimum(prediction, 68.)
        else:
            with np.load(folder / f'{model}_2025_curves.npz', allow_pickle=False) as curves:
                times = pd.to_datetime(curves['origin_ns'], utc=True).tz_convert('Asia/Shanghai')
                if not times.is_unique:
                    raise ValueError('Duplicate curve origin')
                positions = times.get_indexer(uncapped.timestamp)
                if (positions < 0).any():
                    raise ValueError('Missing curve origin')
                prediction = np.array([curve_recovery(curve) for curve in
                                       curves['city_prediction'][positions, :, 1]], dtype=float)
            old_prediction = saved.predicted_hours.fillna(np.inf).to_numpy()
            np.testing.assert_array_equal(prediction, np.minimum(old_prediction, 68.))
        capped[model] = prediction
        scores[model] = score_predictions(capped, prediction)
        audits[model] = {'old_verified_rows': int(saved.label_verified_in_scoring_window.sum()),
                         'finite_saved_truth_rows': int(known.sum()),
                         'no_within_window_recovery_predictions': int((prediction == 68).sum())}
    for model in folders:
        if model != 'matched_aft':
            paired[model] = paired_error_summary(capped, capped.matched_aft, capped[model])
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input changed: {path}')
    out.mkdir()
    uncapped.to_parquet(out / 'uncapped_labels.parquet', index=False)
    capped.to_parquet(out / 'capped_predictions.parquet', index=False)
    result = {'input_sha256': hashes, 'scores': scores, 'paired_vs_matched_aft': paired,
              'audits': audits, 'common_origins': len(capped), 'requested_origins': len(cohort),
              'cohort_coverage': len(capped) / len(cohort),
              'uncapped_interval_rows': int(uncapped.exact_hours.isna().sum()),
              'certainly_beyond_67_rows': int(uncapped.lower_bound_hours.gt(67).sum()),
              'target': 'min(full recovery start time, 68 hours), not full duration',
              'training': 'Existing 2014-2023 models, checkpoints selected using 2024',
              'limitations': ['Reused 2025 research subset, not an independent test or full-year coverage',
                              'Subset fixed by existing six-hour prediction schedule, not performance',
                              'Canonical city-hour labels used; original exclusion mask retained only as audit',
                              'Finite old labels compatible is not proof of every raw station value',
                              'Unconfirmed curve recovery is capped, never a claim that true duration is 68',
                              'Saved forecasts replayed, neural checkpoints not rerun',
                              'Existing regional model reports do not furnish a complete immutable output manifest'],
              'deployment_authorized': False,
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
