"""Paired page comparison, block uncertainty, and partially identified MAE."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'interval_candidate_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    labels_path = base / 'interval_business_labels_v1/origins.parquet'
    page_path = ROOT / 'artifacts/strict_history_backtest/predictions.parquet'
    candidate_paths = {year: base / f'interval_crossyear_v1/{year}/predictions.parquet' for year in (2024, 2025)}
    paths = [labels_path, page_path, *candidate_paths.values(), Path(__file__), ROOT / 'src/haze_forecast/app.py']
    before = {str(p): sha(p) for p in paths}
    labels = pd.read_parquet(labels_path)[['timestamp', 'lower_bound_hours', 'upper_bound_hours']]
    labels = labels.rename(columns={'lower_bound_hours': 'interval_lower', 'upper_bound_hours': 'interval_upper'})
    page = pd.read_parquet(page_path)[['timestamp', 'strict_prediction_hours']]
    out.mkdir(parents=True)
    results = {}
    rng = np.random.default_rng(42)
    for year, path in candidate_paths.items():
        original = pd.read_parquet(path)
        frame = original.merge(page, on='timestamp', validate='one_to_one').merge(labels, on='timestamp', validate='one_to_one')
        assert np.isfinite(frame[['interval_lower', 'interval_upper']].to_numpy()).all()
        arms = ['interval_bounds', 'strict_prediction_hours']
        exact = frame.loc[frame.exact_hours.notna()].copy()
        for arm in arms:
            exact[arm + '_error'] = (exact[arm] - exact.exact_hours).abs()
            # Bounds on absolute error require neither imputation nor an interval midpoint.
            frame[arm + '_error_lower'] = np.maximum.reduce([
                frame.interval_lower - frame[arm], frame[arm] - frame.interval_upper, np.zeros(len(frame))])
            frame[arm + '_error_upper'] = np.maximum(
                (frame[arm] - frame.interval_lower).abs(), (frame[arm] - frame.interval_upper).abs())
        block_rows = []
        for block, rows in exact.groupby('continuity_block'):
            tail = rows.loc[rows.exact_hours.ge(24)]
            block_rows.append({'block': int(block), 'start': str(rows.timestamp.min()), 'n': len(rows),
                'tail_n': len(tail), 'candidate_mae': float(rows.interval_bounds_error.mean()),
                'page_mae': float(rows.strict_prediction_hours_error.mean()),
                'candidate_tail_mae': float(tail.interval_bounds_error.mean()) if len(tail) else None,
                'page_tail_mae': float(tail.strict_prediction_hours_error.mean()) if len(tail) else None,
                'candidate_tail_bias': float((tail.interval_bounds-tail.exact_hours).mean()) if len(tail) else None})
        blocks = pd.DataFrame(block_rows)
        uncertainty = {}
        for name, sample, candidate, baseline in [
            ('block_mae', blocks, 'candidate_mae', 'page_mae'),
            ('tail_block_mae', blocks.loc[blocks.tail_n.gt(0)], 'candidate_tail_mae', 'page_tail_mae')]:
            differences = (sample[candidate] - sample[baseline]).to_numpy(float)
            draws = rng.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
            uncertainty[name] = {'blocks': len(differences), 'candidate_minus_page': float(differences.mean()),
                'paired_block_bootstrap_95_interval': np.quantile(draws, [.025, .975]).tolist(),
                'leave_one_block_out_differences': [float(np.delete(differences, i).mean()) for i in range(len(differences))]}
        partial = {}
        for arm in arms:
            partial[arm] = {kind: [float(frame[arm + '_error_' + bound].mean()) if kind == 'row_mae' else
                float(frame.groupby('continuity_block')[arm + '_error_' + bound].mean().mean())
                for bound in ('lower', 'upper')] for kind in ('row_mae', 'block_mae')}
        results[str(year)] = {'candidate_origins': len(original), 'common_origins': len(frame),
            'common_exact': len(exact), 'unmatched_origins': len(original)-len(frame),
            'exact_row_mae': {arm: float(exact[arm + '_error'].mean()) for arm in arms},
            'uncertainty': uncertainty, 'common_population_compatible_mae_ranges': partial}
        blocks.to_csv(out / f'{year}_blocks.csv', index=False)
        frame.to_parquet(out / f'{year}_common.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'input_sha256': before, 'years': results, 'inputs_unchanged': True,
        'limitations': ['same-origin diagnostic; annual candidate versus event-updated webpage',
            '2025 candidate uses repaired weather but webpage retains original inputs',
            'bootstrap is descriptive with few blocks, not proof of generalization',
            'compatible MAE ranges are not confidence intervals; extrema need not be jointly attainable',
            'no model selection or deployment'],
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['years'], indent=2))


if __name__ == '__main__':
    main()
