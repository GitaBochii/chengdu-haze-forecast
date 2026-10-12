"""Single prespecified 50/50 blend, with development gate and paired block audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    data = ROOT / 'artifacts/task01_recovery_head'
    trajectory = data / 'hazard_trajectory_v1'
    repair = data / 'ghcnh_weather_repair_v1'
    out = data / 'fixed_aft_hazard_blend_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [trajectory / f'{year}_predictions.parquet' for year in (2023, 2024, 2025)]
    paths += [trajectory / 'report.json', repair / 'predictions.parquet', repair / 'report.json',
        Path(__file__), ROOT / 'scripts/benchmark_observed_business_aft.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    out.mkdir(parents=True)
    protocol = {'input_sha256': hashes, 'weight_aft': .5, 'weight_hazard': .5,
        'components': 'annual legacy-training AFT + 98-feature observed-prefix hazard',
        'selection': 'one fixed blend; require development both main errors lower than AFT and tail not worse',
        'weather': '2025 both components GHCNh repaired; 2023/2024 original weather',
        'limitations': ['repeated historical diagnostics not pristine holdout',
            'arithmetic mean of point estimates not a survival-distribution median',
            'legacy training control not actual webpage replay', 'bootstrap not selection corrected']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    proceed = True
    for year in (2023, 2024, 2025):
        if not proceed:
            break
        frame = pd.read_parquet(trajectory / f'{year}_predictions.parquet')
        aft = frame.legacy_training.to_numpy()
        if year == 2025:
            repaired = pd.read_parquet(repair / 'predictions.parquet')
            np.testing.assert_array_equal(frame.timestamp, repaired.timestamp)
            aft = repaired.legacy_training_after.to_numpy()
        hazard = frame.trajectory_hazard.to_numpy()
        assert np.isfinite(aft).all() and np.isfinite(hazard).all()
        blend = (aft + hazard) / 2
        frame['aft_control'], frame['fixed_blend'] = aft, blend
        allowed = frame.score_allowed if year == 2023 else np.ones(len(frame), dtype=bool)
        selected = frame.loc[allowed]
        baseline, candidate = metrics(selected, aft[allowed]), metrics(selected, blend[allowed])
        passed = bool(candidate['row_mae'] < baseline['row_mae'] and
            candidate['block_mae'] < baseline['block_mae'] and
            candidate['tail_24h_block_mae'] <= baseline['tail_24h_block_mae'])
        exact = selected.loc[selected.exact_hours.notna()]
        a_error = (exact.aft_control - exact.exact_hours).abs()
        b_error = (exact.fixed_blend - exact.exact_hours).abs()
        block_delta = (b_error - a_error).groupby(exact.continuity_block).mean().to_numpy()
        rng = np.random.default_rng(42)
        boot = rng.choice(block_delta, size=(10000, len(block_delta)), replace=True).mean(axis=1)
        results[str(year)] = {'baseline': baseline, 'blend': candidate, 'passes_gate': passed,
            'paired_block_delta_95_ci': np.quantile(boot, [.025, .975]).tolist()}
        frame.to_parquet(out / f'{year}_predictions.parquet', index=False)
        if year == 2023:
            proceed = passed
            (out / 'development_decision.json').write_text(json.dumps(results['2023'], indent=2), encoding='utf-8')
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_fixed_blend_diagnostic', 'years': results,
        'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
