"""Observed-prefix hazard development gate with no fabricated tail medians."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_hazard import ObservedHazardRegressor, expand_bounds
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha

BASE = ROOT / 'artifacts/task01_recovery_head/observed_annual_update_v1'
DEV = ROOT / 'artifacts/task01_recovery_head/observed_business_aft_v1'
OUT = ROOT / 'artifacts/task01_recovery_head/observed_hazard_development_v1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    q = json.loads((DEV / 'protocol.json').read_bytes())
    names = q['features']
    paths = [BASE / '2023/training_rows.parquet', DEV / 'predictions.parquet', DEV / 'protocol.json',
        Path(__file__), ROOT / 'src/haze_forecast/observed_hazard.py',
        ROOT / 'src/haze_forecast/discrete_survival.py', ROOT / 'scripts/benchmark_observed_business_aft.py',
        ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    train = pd.read_parquet(paths[0])
    val = pd.read_parquet(paths[1])
    OUT.mkdir(parents=True)
    protocol = {'input_sha256': hashes, 'features': names, 'rounds': 300, 'seed': 42,
        'arms': ['exact_only', 'observed_bounds'], 'selection': 'no crossyear unless complete median coverage and all three development metrics improve legacy',
        'horizon': 'maximum observed training risk hour; no validation-derived horizon',
        'weighting': 'unweighted person-period binary likelihood; no class reweighting'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scores = {}
    baseline = metrics(val.loc[val.score_allowed], val.loc[val.score_allowed, 'legacy_training'].to_numpy())
    for arm in protocol['arms']:
        selected = train if arm == 'observed_bounds' else train.loc[train.exact_hours.notna()]
        model = ObservedHazardRegressor(seed=42).fit_bounds(selected[names], selected.lower_bound_hours, selected.upper_bound_hours)
        joblib.dump(model, OUT / f'{arm}.joblib')
        pred = model.predict(val[names])
        val[arm + '_hazard'] = pred
        expansion, target, _ = expand_bounds(selected[names], selected.lower_bound_hours, selected.upper_bound_hours)
        ok = val.score_allowed & np.isfinite(pred)
        result = metrics(val.loc[ok], pred[ok])
        result.update({'all_origin_median_coverage': float(np.isfinite(pred).mean()),
            'unresolved_origins': int(np.isnan(pred).sum()), 'horizon': model.max_horizon_hours,
            'training_rows': len(selected), 'expanded_rows': len(expansion), 'positive_rows': int(target.sum())})
        result['passes_gate'] = bool(np.isfinite(pred).all() and result['row_mae'] < baseline['row_mae']
            and result['block_mae'] < baseline['block_mae'] and result['tail_24h_block_mae'] <= baseline['tail_24h_block_mae'])
        scores[arm] = result
    val.to_parquet(OUT / 'predictions.parquet', index=False)
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_development', 'baseline': baseline, 'scores': scores,
        'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
