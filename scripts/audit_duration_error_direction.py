"""Descriptive frozen-cohort error attribution; no calibration or model fitting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'duration_error_direction_20260918'
    if out.exists():
        raise FileExistsError(out)
    records, hashes = {}, {}
    for year in (2023, 2024, 2025):
        path = base / f'hourly_support_20260918/{year}_predictions.parquet'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        d = pd.read_parquet(path)
        assert d.timestamp.is_unique and d.native.notna().all()
        lo, hi, p = d.lower_bound_hours, d.upper_bound_hours, d.native
        d['under'] = (lo - p).clip(lower=0)
        d['over'] = (p - hi).clip(lower=0)
        d['distance'] = d.under + d.over
        np.testing.assert_allclose(d.distance, np.maximum.reduce([
            lo - p, p - hi, np.zeros(len(d))]))
        blocks = d.groupby('continuity_block').agg(
            rows=('native', 'size'), distance_sum=('distance', 'sum'),
            under_sum=('under', 'sum'), over_sum=('over', 'sum'))
        blocks['mean_distance'] = blocks.distance_sum / blocks.rows
        exact = d.loc[d.exact_hours.notna()].copy()
        exact['signed_error'] = exact.native - exact.exact_hours
        exact['absolute_error'] = exact.signed_error.abs()
        exact['duration_group'] = pd.cut(exact.exact_hours, [0, 6, 24, 48, np.inf],
                                        right=False, labels=['<6', '6-24', '24-48', '>=48'])
        groups = exact.groupby('duration_group', observed=True).agg(
            rows=('native', 'size'), signed_error=('signed_error', 'mean'),
            mae=('absolute_error', 'mean'), true_mean=('exact_hours', 'mean'),
            prediction_mean=('native', 'mean'))
        np.testing.assert_allclose(
            (groups.rows * groups.mae).sum(), exact.absolute_error.sum())
        np.testing.assert_allclose(blocks.distance_sum.sum(), d.distance.sum())
        records[str(year)] = {
            'rows': len(d), 'blocks': len(blocks),
            'guaranteed_under_rows': int((p < lo).sum()),
            'guaranteed_over_rows': int((p > hi).sum()),
            'under_share_of_minimum_error': float(d.under.sum() / d.distance.sum()),
            'top_two_blocks_share_of_minimum_error': float(
                blocks.distance_sum.nlargest(2).sum() / d.distance.sum()),
            'exact_duration_groups': groups.reset_index().to_dict('records'),
            'blocks_detail': blocks.reset_index().to_dict('records'),
        }
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    out.mkdir()
    report = {'input_sha256': hashes, 'years': records,
              'scope': 'Descriptive only. True-duration bins are hindsight, not deployable gates. '
                       'Interval distances are lower bounds, not exact absolute errors. '
                       'No refit, calibration, deployment, or new holdout claim.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(records, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
