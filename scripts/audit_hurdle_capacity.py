"""Hindsight representational bounds, never an inference or tuning rule."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'artifacts/task01_recovery_head/supervised_hurdle_2023_20260918'
    out = ROOT / 'artifacts/task01_recovery_head/hurdle_capacity_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / 'report.json', source / 'scored.parquet', Path(__file__).resolve()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = json.loads(paths[0].read_bytes())
    d = pd.read_parquet(paths[1])
    mu = report['fits']['input_gate']['mu']
    distributions = [truncnorm(-mu[0]/1.2, (np.log(24)-mu[0])/1.2, loc=mu[0], scale=1.2),
                     truncnorm((np.log(24)-mu[1])/1.2, np.inf, loc=mu[1], scale=1.2)]
    low, high = [float(np.exp(dist.ppf(.5))) for dist in distributions]
    assert d.input_gate.between(low, high).all()
    exact = d.loc[d.exact_hours.notna()].copy()
    exact['oracle_prediction'] = exact.exact_hours.clip(low, high)
    exact['oracle_error'] = (exact.oracle_prediction-exact.exact_hours).abs()
    exact['actual_error'] = (exact.input_gate-exact.exact_hours).abs()
    assert (exact.actual_error >= exact.oracle_error-1e-10).all()
    # A monotone family of medians spans the two conditional medians.
    xs = np.linspace(low, high, 501)
    f0, f1 = [dist.cdf(np.log(xs)) for dist in distributions]
    oracle_p = (.5-f0)/(f1-f0)
    assert (oracle_p >= -1e-12).all() and (oracle_p <= 1+1e-12).all()
    np.testing.assert_allclose((1-oracle_p)*f0+oracle_p*f1, .5, atol=1e-12)
    target = np.where(d.lower_bound_hours >= 24, 1.,
                      np.where(d.upper_bound_hours < 24, 0., np.nan))
    known = np.isfinite(target)
    p, y = d.input_gate_probability.to_numpy()[known], target[known]
    result = {
        'input_sha256': hashes, 'median_envelope_hours': [low, high],
        'actual_prediction_range': [float(d.input_gate.min()), float(d.input_gate.max())],
        'exact_rows': len(exact),
        'outside_envelope': {'below': int((exact.exact_hours < low).sum()),
                             'above': int((exact.exact_hours > high).sum())},
        'oracle_exact_row_mae': float(exact.oracle_error.mean()),
        'actual_exact_row_mae': float(exact.actual_error.mean()),
        'oracle_exact_block_mae': float(exact.groupby('continuity_block').oracle_error.mean().mean()),
        'oracle_hard_class_row_mae': float(np.abs(np.where(exact.exact_hours < 24, low, high)-exact.exact_hours).mean()),
        'known_class_rows': int(known.sum()), 'auc': float(roc_auc_score(y, p)),
        'brier': float(np.mean((p-y)**2)),
        'long_recall_at_half': float(np.mean(p[y == 1] >= .5)),
        'mean_probability_long': float(p[y == 1].mean()),
        'mean_probability_short': float(p[y == 0].mean()),
        'scope': 'Oracle uses true duration; not deployable, not measured accuracy gain. '
                 'Finite median envelope is not a hard cap on the distribution support.'}
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    out.mkdir()
    exact.to_parquet(out / 'exact_oracle_diagnostic.parquet', index=False)
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
