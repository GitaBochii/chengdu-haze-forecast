"""Descriptive support and paired-error audit; no model or threshold selection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_gefs_72h_aft import ROOT, sha, verify_hashes


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_dual_horizon_v1'
    out = base / 'dual_horizon_error_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((source / 'report.json').read_bytes())
    inputs = {str(source / name): value for name, value in report['output_sha256'].items()}
    inputs[str(source / 'report.json')] = sha(source / 'report.json')
    inputs[str(Path(__file__))] = sha(Path(__file__))
    verify_hashes(inputs)
    train = pd.read_parquet(source / 'training_rows.parquet')
    inference = pd.read_parquet(source / 'inference_rows.parquet')
    support = []
    for name, data in [('train', train), ('development', inference)]:
        for horizon in ['f024', 'f072']:
            available = data[f'gefs_mean_temperature_c_{horizon}'].notna()
            selected = data.loc[available]
            support.append({'population': name, 'horizon': horizon, 'total_rows': len(data),
                            'forecast_rows': len(selected),
                            'forecast_blocks': selected.continuity_block.nunique(),
                            'years': sorted(selected.timestamp.dt.year.unique().tolist())})
    grouped, blocks, cohorts = [], [], {}
    for cohort in ['original', 'expanded']:
        data = pd.read_parquet(source / f'{cohort}_predictions.parquet')
        pd.testing.assert_series_equal(data.timestamp, inference.timestamp)
        data['coverage'] = np.select([
            inference.gefs_mean_temperature_c_f024.notna() & inference.gefs_mean_temperature_c_f072.notna(),
            inference.gefs_mean_temperature_c_f024.notna(),
            inference.gefs_mean_temperature_c_f072.notna()], ['both', 'f024_only', 'f072_only'], default='neither')
        scored = data.loc[data.score_allowed & data.exact_hours.notna()].copy()
        assert np.isfinite(scored[['forecast_aft', 'dual_horizon_aft', 'exact_hours']].to_numpy()).all()
        scored['f024_error'] = (scored.forecast_aft - scored.exact_hours).abs()
        scored['dual_error'] = (scored.dual_horizon_aft - scored.exact_hours).abs()
        scored['error_delta'] = scored.dual_error - scored.f024_error
        scored['prediction_delta'] = scored.dual_horizon_aft - scored.forecast_aft
        for coverage, frame in scored.groupby('coverage'):
            grouped.append({'cohort': cohort, 'coverage': coverage, 'rows': len(frame),
                            'blocks': frame.continuity_block.nunique(),
                            'f024_mae': frame.f024_error.mean(), 'dual_mae': frame.dual_error.mean(),
                            'error_delta_sum': frame.error_delta.sum(),
                            'mean_abs_prediction_change': float(frame.prediction_delta.abs().mean())})
        for block, frame in scored.groupby('continuity_block'):
            blocks.append({'cohort': cohort, 'block': int(block), 'rows': len(frame),
                           'start': frame.timestamp.min().isoformat(),
                           'truth_mean': frame.exact_hours.mean(),
                           'f024_mae': frame.f024_error.mean(), 'dual_mae': frame.dual_error.mean(),
                           'error_delta': frame.error_delta.mean(),
                           'forecast_rows': int(frame.coverage.ne('neither').sum())})
        cohorts[cohort] = {'rows': len(scored), 'mae_delta': scored.error_delta.mean(),
                           'total_error_delta': scored.error_delta.sum()}
    verify_hashes(inputs)
    out.mkdir()
    pd.DataFrame(grouped).to_csv(out / 'coverage_errors.csv', index=False)
    pd.DataFrame(blocks).to_csv(out / 'block_errors.csv', index=False)
    result = {'support': support, 'cohorts': cohorts, 'coverage_errors': grouped,
              'input_sha256': inputs, 'limitations': [
                  'Descriptive development-set audit, not independent validation.',
                  'Paired prediction changes include joint tree retraining and feature subsampling.',
                  'Coverage groups are not deployment routing rules; do not remove difficult rows.'],
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'support': support, 'coverage_errors': grouped, 'cohorts': cohorts}, indent=2))


if __name__ == '__main__':
    main()
