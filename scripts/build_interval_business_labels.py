"""Retain post-gap evidence as intervals on the unchanged business population."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.interval_recovery_labels import interval_recovery_bounds
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    data = ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1'
    out = ROOT / 'artifacts/task01_recovery_head/interval_business_labels_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [data / 'features.parquet', data / 'origins.parquet', data / 'report.json', Path(__file__),
        ROOT / 'src/haze_forecast/interval_recovery_labels.py', ROOT / 'src/haze_forecast/observed_recovery_labels.py',
        ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    full = pd.read_parquet(data / 'features.parquet')
    old = pd.read_parquet(data / 'origins.parquet')
    bounds = interval_recovery_bounds(full[['timestamp', 'aqi']], old.timestamp)
    np.testing.assert_array_equal(bounds.timestamp, old.timestamp)
    assert bounds.lower_bound_hours.ge(old.lower_bound_hours).all()
    assert bounds.upper_bound_hours.le(old.upper_bound_hours).all()
    exact = old.exact_hours.notna()
    np.testing.assert_array_equal(bounds.loc[exact, 'exact_hours'], old.loc[exact, 'exact_hours'])
    result = old.drop(columns=[c for c in bounds if c != 'timestamp']).merge(bounds, on='timestamp', validate='one_to_one')
    result = result.rename(columns={'observed_prefix_hours': 'first_gap_observed_prefix_hours'})
    yearly = {}
    for year, rows in result.groupby(result.timestamp.dt.year):
        yearly[str(year)] = {'origins': len(rows), 'exact': int(rows.exact_hours.notna().sum()),
            'interval': int(rows.reason.eq('interval').sum()), 'right_censored': int(rows.reason.eq('right_censored').sum()),
            'uninformative': int((~rows.informative).sum())}
    assert before == {str(p): sha(p) for p in paths}
    out.mkdir(parents=True)
    result.to_parquet(out / 'origins.parquet', index=False)
    report = {'status': 'completed_interval_labels_no_training', 'input_sha256': before,
        'yearly': yearly, 'origins': len(result), 'old_exact_preserved': int(exact.sum()),
        'old_censored_now_exact': int((~exact & result.exact_hours.notna()).sum()),
        'finite_intervals': int(result.reason.eq('interval').sum()),
        'strictly_tighter_bounds': int((bounds.lower_bound_hours.gt(old.lower_bound_hours) | bounds.upper_bound_hours.lt(old.upper_bound_hours)).sum()),
        'output_sha256': sha(out / 'origins.parquet'), 'inputs_unchanged': True,
        'limitations': ['missing values not imputed; bounds describe compatible observations not probability',
            'label availability may be later than first-gap censor availability; enforce at each training cutoff',
            'all later observations used for labels only, never features or eligibility',
            'no accuracy claim and no change to original exact scoring cohort']}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
