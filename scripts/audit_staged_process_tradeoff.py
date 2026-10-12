"""Describe paired process errors without changing the candidate or evaluation cohort."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'staged_forecast_matched_control_20260918/predictions.parquet'
    out = base / 'staged_process_tradeoff_20260918'
    if out.exists():
        raise FileExistsError(out)
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, Path(__file__).resolve())}
    d = pd.read_parquet(source)
    assert d.timestamp.is_unique and d.exact_hours.notna().all()
    assert (d.exact_hours == d.lower_bound_hours).all() and (d.exact_hours == d.upper_bound_hours).all()
    rows = []
    for block, group in d.groupby('continuity_block', sort=True):
        candidate = (group.weather - group.exact_hours).abs()
        control = (group.joint - group.exact_hours).abs()
        rows.append({'block': int(block), 'rows': len(group),
                     'first_origin': group.timestamp.min().isoformat(),
                     'last_origin': group.timestamp.max().isoformat(),
                     'candidate_mae': float(candidate.mean()), 'control_mae': float(control.mean()),
                     'delta_hours': float((candidate - control).mean())})
    blocks = pd.DataFrame(rows)
    delta = blocks.delta_hours.to_numpy()
    row_delta = float(np.average(delta, weights=blocks.rows))
    scalar = [abs(float(p) - float(y)) - abs(float(q) - float(y))
              for p, q, y in zip(d.weather, d.joint, d.exact_hours)]
    np.testing.assert_allclose(row_delta, np.mean(scalar), atol=1e-12)
    independent = []
    for block in blocks.block:
        group = d.loc[d.continuity_block.eq(block)]
        independent.append(sum(abs(float(p)-float(y))-abs(float(q)-float(y))
                               for p,q,y in zip(group.weather,group.joint,group.exact_hours))/len(group))
    np.testing.assert_allclose(delta, independent, atol=1e-12)
    leave_one_out = (delta.sum() - delta) / (len(delta) - 1)
    # This sensitivity calculation does not remove any observation from official scores.
    report = {'input_sha256': inputs, 'rows': len(d), 'blocks': len(blocks),
              'improved_blocks': int((delta < 0).sum()), 'worse_blocks': int((delta > 0).sum()),
              'tied_blocks': int((delta == 0).sum()), 'row_delta_hours': row_delta,
              'block_delta_hours': float(delta.mean()),
              'median_block_delta_hours': float(np.median(delta)),
              'leave_one_block_out_negative_count': int((leave_one_out < 0).sum()),
              'leave_one_block_out_positive_count': int((leave_one_out > 0).sum()),
              'leave_one_block_out_range_hours': [float(leave_one_out.min()), float(leave_one_out.max())],
              'interpretation': 'Descriptive sensitivity only, not a confidence interval or a changed gate.'}
    for p,h in inputs.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    out.mkdir()
    blocks.to_csv(out / 'blocks.csv', index=False)
    report['blocks_sha256'] = hashlib.sha256((out / 'blocks.csv').read_bytes()).hexdigest()
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(blocks.sort_values('delta_hours').to_string(index=False))


if __name__ == '__main__':
    main()
