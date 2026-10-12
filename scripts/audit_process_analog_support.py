"""Diagnose frozen analogue support without selecting a new predictor."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    source = ROOT / 'artifacts/task01_recovery_head/process_analog_2020_v2'
    out = ROOT / 'artifacts/task01_recovery_head/process_analog_support_2020_v1'
    if out.exists():
        raise FileExistsError(out)
    files = [source / 'predictions.parquet', source / 'neighbors.parquet',
             source / 'report.json', Path(__file__)]
    before = {str(p): sha(p) for p in files}
    rows = pd.read_parquet(files[0])
    neighbors = pd.read_parquet(files[1])
    assert rows.timestamp.is_unique and len(neighbors) == 15 * len(rows)
    assert neighbors.groupby('query_timestamp').block.nunique().eq(15).all()
    records = []
    for stamp, group in neighbors.groupby('query_timestamp', sort=True):
        group = group.sort_values('rank')
        # A positive gap proves at least two donor outcomes cannot coincide.
        gap = max(0., float(group.lower.max() - group.upper.min()))
        records.append({'timestamp': stamp, 'nearest_distance': float(group.distance.iloc[0]),
                        'farthest_distance': float(group.distance.iloc[-1]),
                        'donor_incompatible_gap': gap,
                        'donor_definitely_long': int(group.lower.ge(24).sum()),
                        'donor_definitely_short': int(group.upper.lt(24).sum()),
                        'nearest_lower': float(group.lower.iloc[0]),
                        'nearest_upper': float(group.upper.iloc[0])})
    diagnostics = rows.merge(pd.DataFrame(records), on='timestamp', validate='one_to_one')
    assert len(diagnostics) == len(rows)
    exact = diagnostics.exact_hours.notna()
    diagnostics['analog_signed_error'] = diagnostics.process_analog_prediction - diagnostics.exact_hours
    diagnostics['aft_signed_error'] = diagnostics.paired_aft - diagnostics.exact_hours
    diagnostics['analog_abs_error'] = diagnostics.analog_signed_error.abs()
    diagnostics['aft_abs_error'] = diagnostics.aft_signed_error.abs()
    diagnostics['nearest_outcome_gap'] = np.maximum.reduce([
        (diagnostics.nearest_lower - diagnostics.upper_bound_hours).to_numpy(),
        (diagnostics.lower_bound_hours - diagnostics.nearest_upper).to_numpy(),
        np.zeros(len(diagnostics))])
    known = diagnostics.loc[exact]
    summary = {}
    for name, frame in [('all_exact', known), ('short_exact', known.loc[known.exact_hours.lt(24)]),
                        ('long_exact', known.loc[known.exact_hours.ge(24)])]:
        summary[name] = {'rows': len(frame), 'blocks': int(frame.continuity_block.nunique()),
                         'analog_signed_error_mean': float(frame.analog_signed_error.mean()),
                         'aft_signed_error_mean': float(frame.aft_signed_error.mean()),
                         'nearest_distance_median': float(frame.nearest_distance.median()),
                         'donor_gap_median': float(frame.donor_incompatible_gap.median()),
                         'donor_long_fraction': float(frame.donor_definitely_long.mean() / 15),
                         'nearest_outcome_gap_median': float(frame.nearest_outcome_gap.median())}
    blocks = known.groupby('continuity_block').agg(
        rows=('timestamp', 'size'), analog_mae=('analog_abs_error', 'mean'),
        aft_mae=('aft_abs_error', 'mean'), donor_gap=('donor_incompatible_gap', 'median'),
        nearest_distance=('nearest_distance', 'median')).reset_index()
    summary['processes_improved'] = int(blocks.analog_mae.lt(blocks.aft_mae).sum())
    summary['processes_worsened'] = int(blocks.analog_mae.gt(blocks.aft_mae).sum())
    summary['distance_error_spearman'] = float(known.nearest_distance.corr(known.analog_abs_error, method='spearman'))
    summary['gap_error_spearman'] = float(known.donor_incompatible_gap.corr(known.analog_abs_error, method='spearman'))
    assert before == {str(p): sha(p) for p in files}
    out.mkdir()
    diagnostics.to_parquet(out / 'origins.parquet', index=False)
    blocks.to_csv(out / 'blocks.csv', index=False)
    report = {'input_sha256': before, 'summary': summary, 'model_changed': False,
              'limitations': ['Outcome groups and correlations are retrospective diagnostics only',
                              'Donor disagreement is not an irreducible-error bound',
                              'Distances depend on the fixed feature scaling and retrieval definition']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
