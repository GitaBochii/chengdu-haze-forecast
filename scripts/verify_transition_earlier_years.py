"""Independently check temporal memberships, observed paths and model replay."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'transition_earlier_years_v1'
    p = json.loads((out / 'protocol.json').read_bytes())
    r = json.loads((out / 'report.json').read_bytes())
    for path, digest in p['input_sha256'].items():
        assert sha(Path(path)) == digest
    for name, digest in r['output_sha256'].items():
        assert sha(out / name) == digest
    rows = pd.read_parquet(base / 'gefs_temporal_calibration_data_v1/training_rows.parquet')
    grid = pd.read_parquet(base / 'observed_business_dataset_v1/features.parquet')
    ranges = grid.groupby('continuity_block').timestamp.agg(['min', 'max'])
    indexed = grid.set_index('timestamp')
    checks = {}
    for year in p['years']:
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year + 1}-01-01', tz='Asia/Shanghai')
        cross = ranges.index[(ranges['min'] < cut) & (ranges['max'] >= cut)]
        last = ranges.index[(ranges['min'] < end) & (ranges['max'] >= end)]
        allowed = rows.index[(rows.timestamp >= cut) & (rows.timestamp < end)
                             & (rows.label_available_at < end)
                             & ~rows.continuity_block.isin(cross.union(last))]
        expected = []
        for owner in allowed:
            row = rows.loc[owner]
            begin = indexed.index.get_loc(row.timestamp) + 1
            future = indexed.aqi.iloc[begin:begin + int(row.upper_bound_hours) + 5].to_numpy()
            count = 0
            for hour, value in enumerate(future, start=1):
                if np.isnan(value):
                    break
                expected.append((owner, hour, count, int(value <= 100)))
                count = count + 1 if value <= 100 else 0
                if count == 6:
                    break
        table = pd.read_parquet(out / f'{year}_probabilities.parquet')
        np.testing.assert_array_equal(table[['owner', 'horizon_hours', 'clean_run', 'target']], expected)
        x = np.column_stack((rows[p['features']].to_numpy(dtype=float)[table.owner.to_numpy()],
                             table.horizon_hours, table.clean_run))
        model = lgb.Booster(model_str=(out / f'{year}_model.txt').read_text(encoding='utf-8'))
        np.testing.assert_array_equal(model.predict(x, num_threads=1), table.probability)
        for score in r['years'][str(year)]['states']:
            part = table[table.clean_run.eq(score['state'])]
            np.testing.assert_allclose(part.target.mean(), score['observed_clean_rate'])
            np.testing.assert_allclose(part.probability.mean(), score['predicted_clean_rate'])
            assert part.loc[part.target.eq(0), 'valid_time'].nunique() == score['failure_physical_times']
        checks[str(year)] = len(table)
    result = {'status': 'PASS', 'verified_transition_predictions': checks,
              'verifier_sha256': sha(Path(__file__))}
    (out / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
