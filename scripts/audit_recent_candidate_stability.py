"""Paired process bootstrap of frozen candidates; descriptive, not selection corrected."""

import json

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'recent_candidate_stability_v1'
    if out.exists():
        raise FileExistsError(out)
    arms = {
        'ozone': ('ozone_contract_2020_v1/predictions.parquet', 'ozone_corrected_prediction'),
        'analog': ('process_analog_2020_v2/predictions.parquet', 'process_analog_prediction'),
        'leaf_analog': ('leaf_process_analog_2020_v1/predictions.parquet', 'leaf_analog_prediction'),
        'recent3': ('recent_three_years_v1/2020_predictions.parquet', 'recent_prediction'),
    }
    reference = pd.read_parquet(base / 'transition_earlier_recovery_v1/2020_predictions.parquet')
    keys = ['timestamp', 'continuity_block', 'lower_bound_hours', 'upper_bound_hours', 'exact_hours', 'paired_aft']
    hashes, report, tables = {}, {}, []
    for name, (relative, prediction) in arms.items():
        path = base / relative
        hashes[relative] = sha(path)
        frame = pd.read_parquet(path)
        pd.testing.assert_frame_equal(reference[keys], frame[keys])
        assert np.isfinite(frame[prediction]).all()
        exact = frame.loc[frame.exact_hours.notna()].copy()
        exact['delta'] = (exact[prediction] - exact.exact_hours).abs() - (exact.paired_aft - exact.exact_hours).abs()
        blocks = exact.groupby('continuity_block').delta.mean()
        rng = np.random.default_rng(20260912)
        indices = rng.integers(0, len(blocks), size=(10000, len(blocks)))
        values = blocks.to_numpy()
        draws = values[indices].mean(axis=1)
        jackknife = (values.sum() - values) / (len(values) - 1)
        report[name] = {'all_rows': len(frame), 'exact_rows': len(exact),
                        'finite_interval_rows_not_in_exact_mae': int(frame.exact_hours.isna().sum()),
                        'exact_processes': len(blocks), 'candidate_minus_aft_process_mae': float(values.mean()),
                        'descriptive_bootstrap_95_range': np.quantile(draws, [.025, .975]).tolist(),
                        'bootstrap_fraction_delta_below_zero': float((draws < 0).mean()),
                        'leave_one_process_out_delta_range': [float(jackknife.min()), float(jackknife.max())],
                        'processes_improved': int((values < 0).sum()),
                        'processes_worsened': int((values > 0).sum())}
        tables.append(blocks.rename('delta').reset_index().assign(candidate=name))
    assert hashes == {p: sha(base / p) for p in hashes}
    out.mkdir()
    pd.concat(tables).to_csv(out / 'process_deltas.csv', index=False)
    document = {'input_sha256': hashes, 'seed': 20260912, 'draws': 10000, 'results': report,
                'limitations': ['2020 repeatedly used for development; intervals are not selection corrected',
                                'Bootstrap assumes processes exchangeable; temporal dependence may remain',
                                'Exact-process metric only; 32 interval-label rows retained in original scores',
                                'No new model, significance-based gate relaxation, or deployment']}
    (out / 'report.json').write_text(json.dumps(document, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
