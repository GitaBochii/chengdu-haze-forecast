"""Read-only, fixed-cohort signed-error audit; never fits or selects a model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_gefs_72h_aft import verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def number(value):
    return float(value) if np.isfinite(value) else None


def summarize(frame, column):
    prediction = frame[column].astype(float)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError('Predictions must be finite and positive')
    lower, upper = frame.lower_bound_hours, frame.upper_bound_hours
    exact = frame.exact_hours.notna()
    if (not np.isfinite(lower).all() or (lower <= 0).any()
            or upper.isna().any() or (upper < lower).any()
            or not frame.loc[exact, 'exact_hours'].eq(lower[exact]).all()
            or not lower[exact].eq(upper[exact]).all()
            or frame.continuity_block.isna().any()):
        raise ValueError('Invalid bounds or block labels')
    result = {}
    for name, mask in [('exact', exact), ('short', exact & frame.exact_hours.lt(24)),
                       ('tail', exact & frame.exact_hours.ge(24))]:
        error = prediction[mask] - frame.loc[mask, 'exact_hours']
        blocks = frame.loc[mask, 'continuity_block']
        result[name] = {
            'rows': int(mask.sum()), 'blocks': int(blocks.nunique()),
            'row_mae': number(error.abs().mean()),
            'block_mae': number(error.abs().groupby(blocks).mean().mean()),
            'row_bias': number(error.mean()),
            'block_bias': number(error.groupby(blocks).mean().mean()),
            'underprediction_fraction': number((error < 0).mean()),
        }
    mask = ~exact
    distance = (lower[mask] - prediction[mask]).clip(lower=0) + (
        prediction[mask] - upper[mask]).clip(lower=0)
    result['interval'] = {
        'rows': int(mask.sum()),
        'blocks': int(frame.loc[mask, 'continuity_block'].nunique()),
        'row_distance': number(distance.mean()),
        'block_distance': number(distance.groupby(frame.loc[mask, 'continuity_block']).mean().mean()),
        'below_lower_fraction': number((prediction[mask] < lower[mask]).mean()),
    }
    direction = {}
    for name, indicator in {
        'definitely_under': lower > prediction,
        'possibly_under': upper > prediction,
        'definitely_over': upper < prediction,
        'possibly_over': lower < prediction,
        'exact_tie': exact & prediction.eq(lower),
    }.items():
        direction[f'{name}_fraction'] = number(indicator.mean())
        direction[f'block_{name}_fraction'] = number(
            indicator.groupby(frame.continuity_block).mean().mean())
    result['direction_bounds'] = direction
    return result


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'original_gefs_without_pbl_v1'
    cross = base / 'original_gefs_without_pbl_crossyear_v1'
    out = base / 'aft_signed_error_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [Path(__file__), ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    inputs = {str(p): sha(p) for p in paths}
    for folder in (dev, cross):
        report = json.loads((folder / 'report.json').read_bytes())
        inputs.update({str(folder / name): value for name, value in report['output_sha256'].items()})
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
    verify_hashes(inputs)
    cohorts = {}
    blocks = []
    for year in (2023, 2024, 2025):
        path = dev / 'expanded_predictions.parquet' if year == 2023 else cross / f'{year}_common.parquet'
        frame = pd.read_parquet(path)
        if year == 2023:
            frame = frame.loc[frame.score_allowed].copy()
        if frame.timestamp.duplicated().any():
            raise ValueError('Duplicate scoring timestamps')
        columns = ['forecast_aft', 'no_history_pbl']
        if year != 2023:
            columns.append('strict_prediction_hours')
        cohorts[str(year)] = {col: summarize(frame, col) for col in columns}
        exact = frame.loc[frame.exact_hours.notna()].copy()
        for col in columns:
            exact['signed_error'] = exact[col] - exact.exact_hours
            exact['absolute_error'] = exact.signed_error.abs()
            grouped = exact.groupby('continuity_block').agg(
                rows=('timestamp', 'size'), bias=('signed_error', 'mean'),
                mae=('absolute_error', 'mean'), truth_mean=('exact_hours', 'mean'))
            grouped['year'], grouped['model'] = year, col
            blocks.append(grouped.reset_index())
    verify_hashes(inputs)
    out.mkdir()
    pd.concat(blocks, ignore_index=True).to_csv(out / 'block_errors.csv', index=False)
    result = {
        'input_sha256': inputs, 'cohorts': cohorts,
        'sign_convention': 'prediction minus exact truth; negative means too early',
        'selection': 'none; frozen models and cohorts, descriptive only',
        'limitations': [
            'Grouping by future true duration is diagnostic only, never a router feature.',
            'Tail underprediction alone does not prove conditional median miscalibration.',
            'Direction bounds are descriptive, not confidence intervals or conditional calibration tests.',
            'Intervals use distance to bounds, not MAE against unknown exact truth.',
            '2023 development and 2024/2025 common expanded cohorts differ and are reused.',
        ],
        'deployment_authorized': False,
        'output_sha256': {'block_errors.csv': sha(out / 'block_errors.csv')},
    }
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(cohorts, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
