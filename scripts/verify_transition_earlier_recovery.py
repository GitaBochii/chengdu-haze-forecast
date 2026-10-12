"""Saved-model replay, temporal checks and independent metric formula checks."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'transition_earlier_recovery_v1'
    source = base / 'transition_earlier_years_v1'
    p = json.loads((out / 'protocol.json').read_bytes())
    r = json.loads((out / 'report.json').read_bytes())
    for path, digest in p['input_sha256'].items():
        assert sha(Path(path)) == digest
    for name, digest in r['output_sha256'].items():
        assert sha(out / name) == digest
    counts = {}
    for year in p['years']:
        tr = pd.read_parquet(out / f'{year}_training.parquet')
        va = pd.read_parquet(out / f'{year}_predictions.parquet')
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        assert tr.timestamp.lt(cut).all() and tr.label_available_at.lt(cut).all()
        assert va.timestamp.ge(cut).all() and va.timestamp.lt(end).all() and va.label_available_at.lt(end).all()
        assert not set(tr.continuity_block) & set(va.continuity_block)
        np.testing.assert_array_equal(joblib.load(out / f'{year}_aft.joblib').predict(va[p['features']]), va.paired_aft)
        horizon = int(tr.upper_bound_hours.max()) + 5
        model = lgb.Booster(model_str=(source / f'{year}_model.txt').read_text(encoding='utf-8'))
        saved = np.load(out / f'{year}_distribution.npy')
        for start in range(0, len(va), 16):
            x = va[p['features']].iloc[start:start + 16].to_numpy(dtype=float)
            augmented = np.column_stack((np.repeat(x, horizon * 6, axis=0),
                                         np.tile(np.repeat(np.arange(1, horizon + 1), 6), len(x)),
                                         np.tile(np.arange(6), len(x) * horizon)))
            q = model.predict(augmented, num_threads=1).reshape(len(x), horizon, 6)
            mass = np.zeros((len(x), 7))
            mass[:, 0] = 1
            events = []
            for hour in range(horizon):
                matrix = np.zeros((len(x), 7, 7))
                for state in range(6):
                    matrix[:, state, 0] = 1 - q[:, hour, state]
                    matrix[:, state, state+1] = q[:, hour, state]
                matrix[:, 6, 6] = 1
                updated = np.einsum('bi,bij->bj', mass, matrix)
                if hour >= 5:
                    events.append(updated[:, 6] - mass[:, 6])
                mass = updated
            decoded = np.column_stack((*events, mass[:, :6].sum(axis=1)))
            np.testing.assert_allclose(decoded, saved[start:start+len(x)], atol=1e-12, rtol=1e-10)
        cdf = saved[:, :-1].cumsum(axis=1)
        median = np.where(cdf[:, -1] >= .5, (cdf >= .5).argmax(axis=1)+1., np.nan)
        np.testing.assert_array_equal(median, va.transition)
        for name, column in [('control', 'paired_aft'), ('candidate', 'transition')]:
            score = r['scores'][str(year)][name]
            if score is None:
                continue
            error = (va[column] - va.exact_hours).abs()
            exact = va.exact_hours.notna()
            tail = va.exact_hours.ge(24)
            np.testing.assert_allclose(error[exact].mean(), score['exact_row_mae'])
            np.testing.assert_allclose(error[exact].groupby(va.loc[exact, 'continuity_block']).mean().mean(), score['exact_block_mae'])
            np.testing.assert_allclose(error[tail].groupby(va.loc[tail, 'continuity_block']).mean().mean(), score['tail24_block_mae'])
            low = np.maximum(np.maximum(va.lower_bound_hours-va[column], va[column]-va.upper_bound_hours), 0)
            high = np.maximum(abs(va.lower_bound_hours-va[column]), abs(va.upper_bound_hours-va[column]))
            np.testing.assert_allclose([low.mean(), high.mean()], [score['all_row_error_lower'], score['all_row_error_upper']])
        counts[str(year)] = len(va)
    result = {'status': 'PASS', 'verified_origins': counts, 'verifier_sha256': sha(Path(__file__))}
    (out / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
