"""Fit one nonnegative LAD affine map on 2022, freeze before reading 2023."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from scripts.benchmark_gefs_72h_aft import compare_cohort, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def predict(params, values):
    return np.maximum(1e-6, params[0] + params[1] * np.asarray(values, dtype=float))


def fit_calibrator(values, truth, weights):
    x, y, w = [np.asarray(v, dtype=float) for v in (values, truth, weights)]
    if (x.ndim != 1 or not len(x) or y.shape != x.shape or w.shape != x.shape
            or not all(np.isfinite(v).all() for v in (x, y, w))
            or (x <= 0).any() or (y <= 0).any() or (w < 0).any() or w.sum() <= 0):
        raise ValueError('Finite aligned positive observations and valid weights required')
    n = len(x)
    design = np.column_stack([np.ones(n), x])
    # Residual auxiliaries encode |a + b*x - y|; nonnegative b preserves order.
    constraints = np.vstack([np.column_stack([design, -np.eye(n)]),
                             np.column_stack([-design, -np.eye(n)])])
    result = linprog(np.r_[0., 0., w / w.sum()], A_ub=constraints,
                     b_ub=np.r_[y, -y], bounds=(0, None), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    params = result.x[:2]
    loss = float(np.average(np.abs(predict(params, x) - y), weights=w))
    if loss > float(np.average(np.abs(x - y), weights=w)) + 1e-6:
        raise RuntimeError('Calibration worse than feasible identity on fitting data')
    return params


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    past = base / 'gefs_temporal_calibration_data_v1'
    later = base / 'gefs_2021_extension_v1'
    out = base / 'gefs_temporal_affine_calibration_v1'
    if out.exists():
        raise FileExistsError(out)
    inputs = {str(Path(__file__)): sha(Path(__file__))}
    for folder in (past, later):
        report = json.loads((folder / 'report.json').read_bytes())
        inputs.update({str(folder / name): value for name, value in report['output_sha256'].items()})
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
        inputs.update(json.loads((folder / 'protocol.json').read_bytes())['input_sha256'])
    verify_hashes(inputs)
    calibration = pd.read_parquet(past / 'predictions.parquet')
    calibration = calibration.loc[calibration.score_allowed].copy()
    end = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    if not (calibration.timestamp.lt(end).all() and calibration.label_available_at.lt(end).all()
            and calibration.exact_hours.notna().all()):
        raise ValueError('Calibration must contain only mature exact pre-2023 labels')
    weights = 1. / calibration.continuity_block.map(calibration.continuity_block.value_counts())
    params = fit_calibrator(calibration.forecast_aft, calibration.exact_hours, weights)
    out.mkdir()
    protocol = {
        'input_sha256': inputs, 'intercept': float(params[0]), 'slope': float(params[1]),
        'rule': 'nonnegative intercept/slope, block-balanced LAD fitted on 2022 only; prediction floor 1e-6',
        'selection': 'one fixed map, no parameter sweep; freeze before reading 2023 prediction values',
        'gate': 'both 2023 cohorts improve row/block MAE and do not worsen tail versus original f024 and uncalibrated extension',
        'calibration_rows': len(calibration), 'deployment_authorized': False,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    calibration['calibrated'] = predict(params, calibration.forecast_aft)
    calibration.to_parquet(out / 'calibration_predictions.parquet', index=False)
    cohorts = {}
    for name, filename in [('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]:
        frame = pd.read_parquet(later / filename)
        if set(calibration.continuity_block) & set(frame.continuity_block):
            raise ValueError('Calibration and development blocks overlap')
        prediction = predict(params, frame.extended_2021_aft)
        primary = compare_cohort(frame, prediction)
        paired = frame.copy()
        paired['forecast_aft'] = paired.extended_2021_aft
        secondary = compare_cohort(paired, prediction)
        cohorts[name] = {'against_original': primary, 'against_extension': secondary,
                         'passed': primary['passed'] and secondary['passed']}
        frame['calibrated'] = prediction
        frame.to_parquet(out / f'{name}_predictions.parquet', index=False)
    verify_hashes(inputs)
    result = {'cohorts': cohorts, 'passed': all(v['passed'] for v in cohorts.values()),
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'parameters': params.tolist(), **result}, indent=2))


if __name__ == '__main__':
    main()
