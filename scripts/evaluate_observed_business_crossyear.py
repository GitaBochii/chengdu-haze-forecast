"""Frozen-model cross-year diagnostics, with paired continuity-block uncertainty."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/task01_recovery_head/observed_business_aft_v1'
DATA = ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1'
OUT = ROOT / 'artifacts/task01_recovery_head/observed_business_crossyear_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol = json.loads((BASE / 'protocol.json').read_bytes())
    inputs = [BASE / f'{arm}.joblib' for arm in protocol['arms']]
    inputs += [BASE / 'protocol.json', DATA / 'origins.parquet', Path(__file__),
               ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/observed_aft.py',
               ROOT / 'src/haze_forecast/app.py']
    inputs += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in inputs}
    origins = pd.read_parquet(DATA / 'origins.parquet')
    rows = origins.loc[origins.timestamp.dt.year.isin([2024, 2025])].copy().reset_index(drop=True)
    for arm in protocol['arms']:
        model = joblib.load(BASE / f'{arm}.joblib')
        rows[arm] = model.predict(rows[protocol['features']])
        assert np.isfinite(rows[arm]).all()
    scores = {}
    for year, group in rows.groupby(rows.timestamp.dt.year):
        exact = group.loc[group.exact_hours.notna()]
        censor = group.loc[group.exact_hours.isna() & group.informative]
        per_arm = {}
        for arm in protocol['arms']:
            error = (exact[arm] - exact.exact_hours).abs()
            tail = exact.exact_hours.ge(24)
            per_arm[arm] = {
                'row_mae': float(error.mean()),
                'block_mae': float(error.groupby(exact.continuity_block).mean().mean()),
                'tail_24h_block_mae': float(error[tail].groupby(exact.loc[tail, 'continuity_block']).mean().mean()),
                'below_censor_bound_fraction': float(censor[arm].lt(censor.lower_bound_hours).mean()),
            }
        delta = (exact.observed_bounds - exact.exact_hours).abs() - (exact.legacy_training - exact.exact_hours).abs()
        block_delta = delta.groupby(exact.continuity_block).mean().to_numpy()
        rng = np.random.default_rng(42)
        boot = rng.choice(block_delta, size=(10000, len(block_delta)), replace=True).mean(axis=1)
        scores[str(year)] = {'all_origins': len(group), 'exact_rows': len(exact),
            'exact_blocks': int(exact.continuity_block.nunique()),
            'informative_censored_rows': len(censor), 'scores': per_arm,
            'bounds_minus_legacy_block_mae_bootstrap_95_ci': np.quantile(boot, [.025, .975]).tolist()}
    assert hashes == {str(p): sha(p) for p in inputs}
    OUT.mkdir(parents=True)
    rows.to_parquet(OUT / 'predictions.parquet', index=False)
    result = {'status': 'completed_frozen_crossyear_diagnostic', 'input_sha256': hashes,
        'scores': scores, 'prediction_sha256': sha(OUT / 'predictions.parquet'),
        'limitations': ['2024_2025_previously_inspected_not_pristine_holdout',
            'all_arms_same_grid_features_not_actual_website_replay',
            'exact_subset_excludes_unknown_outcomes', 'bootstrap_not_selection_corrected',
            '2024_includes_development_crossyear_block_1329', 'no_refitting_or_deployment']}
    (OUT / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
