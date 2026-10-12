"""Bound-certain duration ordering across different processes, diagnostic only."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def audit(frame, names):
    rows = []
    for ga, gb in itertools.combinations(sorted(frame.continuity_block.unique()), 2):
        a = frame.loc[frame.continuity_block.eq(ga)]
        b = frame.loc[frame.continuity_block.eq(gb)]
        a_short = a.upper_bound_hours.to_numpy()[:, None] < b.lower_bound_hours.to_numpy()[None, :]
        b_short = b.upper_bound_hours.to_numpy()[None, :] < a.lower_bound_hours.to_numpy()[:, None]
        known = a_short | b_short
        count = int(known.sum())
        result = {'block_a': int(ga), 'block_b': int(gb), 'possible_pairs': len(a)*len(b),
                  'certain_pairs': count}
        for name in names:
            difference = a[name].to_numpy()[:, None] - b[name].to_numpy()[None, :]
            correct = ((difference < 0) & a_short) | ((difference > 0) & b_short)
            credit = correct.astype(float) + 0.5*(difference == 0)
            result[name] = float(credit[known].mean()) if count else None
            # Independent scalar pass verifies strict interval ordering and tie credit.
            scalar = []
            for x in a.to_dict('records'):
                for y in b.to_dict('records'):
                    sign = -1 if x['upper_bound_hours'] < y['lower_bound_hours'] else (
                        1 if y['upper_bound_hours'] < x['lower_bound_hours'] else 0)
                    if sign:
                        scalar.append(0.5 if x[name] == y[name] else float((x[name]-y[name])*sign > 0))
            assert len(scalar) == count
            if count:
                np.testing.assert_allclose(result[name], sum(scalar)/count, atol=1e-12)
        rows.append(result)
    pairs = pd.DataFrame(rows)
    valid = pairs.certain_pairs.gt(0)
    return pairs, {
        'origins': len(frame), 'processes': int(frame.continuity_block.nunique()),
        'possible_crossprocess_pairs': int(pairs.possible_pairs.sum()),
        'certain_crossprocess_pairs': int(pairs.certain_pairs.sum()),
        'eligible_process_pairs': int(valid.sum()),
        'process_pair_equal_concordance': {n:float(pairs.loc[valid,n].mean()) for n in names},
        'origin_pair_equal_concordance': {n:float(np.average(pairs.loc[valid,n],weights=pairs.loc[valid,'certain_pairs'])) for n in names},
    }


def main():
    b = ROOT/'artifacts/task01_recovery_head'
    out = b/'crossprocess_ordering_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = {'2023': b/'gefs_2020_extension_2023_20260918/scored.parquet',
             '2022': b/'staged_forecast_matched_control_20260918/predictions.parquet'}
    names = {'2023':['native','extended_2021','candidate'], '2022':['native','joint','weather']}
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths.values(),Path(__file__).resolve()]}
    reports, tables = {}, {}
    for year,path in paths.items():
        frame = pd.read_parquet(path)
        assert frame.timestamp.is_unique
        assert np.isfinite(frame[names[year]].to_numpy()).all()
        assert frame.lower_bound_hours.le(frame.upper_bound_hours).all()
        tables[year], reports[year] = audit(frame,names[year])
    out.mkdir()
    for year,table in tables.items():
        table.to_csv(out/f'{year}_process_pairs.csv',index=False)
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    report={'input_sha256':hashes,'years':reports,
            'definition':'Only pairs from different continuity blocks with nonoverlapping label bounds; prediction ties receive half credit.',
            'limitations':['Overlapping bounds and true ties do not establish a direction and are excluded from ordering only, not MAE.',
                           'Pairs share origins and blocks; pair counts are not independent sample sizes.',
                           'Diagnostic only; no new selection, fitting, threshold or deployment.']}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()
