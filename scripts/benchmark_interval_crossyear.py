"""Fixed annual interval AFT diagnostic; preserve the original exact scoring cohort."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import DATA, ROOT, metrics, sha


def main():
    started = time.monotonic()
    base = ROOT / 'artifacts/task01_recovery_head'
    labels = base / 'interval_business_labels_v1/origins.parquet'
    repair = base / 'ghcnh_weather_repair_v1'
    out = base / 'interval_crossyear_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [labels, DATA / 'features.parquet', DATA / 'origins.parquet', DATA / 'report.json',
             repair / 'repaired_features.parquet', repair / 'predictions.parquet', Path(__file__),
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/app.py']
    paths += [base / f'observed_annual_update_v1/{y}/predictions.parquet' for y in (2024, 2025)]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((DATA / 'report.json').read_bytes())['feature_names']
    full = pd.read_parquet(DATA / 'features.parquet')
    origins = pd.read_parquet(labels)
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'seed': 42, 'rounds': 500,
                'years': [2024, 2025], 'training': '2014 onward; mature interval labels before January 1; purge crossing blocks',
                'scoring': 'original exact population unchanged; 2025 corrected weather for all arms',
                'selection': 'none; historical diagnostic, not pristine holdout'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    for year in protocol['years']:
        cutoff = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        cross = set(full.loc[full.timestamp.lt(cutoff), 'continuity_block']) & set(
            full.loc[full.timestamp.ge(cutoff), 'continuity_block'])
        train = origins.loc[origins.timestamp.lt(cutoff) & origins.label_available_at.lt(cutoff)
                            & origins.informative & ~origins.continuity_block.isin(cross)].copy()
        validation = pd.read_parquet(base / f'observed_annual_update_v1/{year}/predictions.parquet')
        x = validation[names].copy()
        if year == 2025:
            repaired = pd.read_parquet(repair / 'predictions.parquet')
            np.testing.assert_array_equal(validation.timestamp, repaired.timestamp)
            x = pd.read_parquet(repair / 'repaired_features.parquet')[names]
            assert len(x) == len(validation)
            for arm in ('legacy_training', 'exact_only', 'observed_bounds'):
                validation[arm] = repaired[f'{arm}_after'].to_numpy()
        assert not set(train.continuity_block) & set(validation.continuity_block)
        dest = out / str(year)
        dest.mkdir()
        train.to_parquet(dest / 'training_rows.parquet', index=False)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(
            train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, dest / 'interval_bounds.joblib')
        pred = model.predict(x)
        assert np.isfinite(pred).all()
        np.testing.assert_array_equal(joblib.load(dest / 'interval_bounds.joblib').predict(x), pred)
        validation['interval_bounds'] = pred
        validation.to_parquet(dest / 'predictions.parquet', index=False)
        scores = {arm: metrics(validation, validation[arm]) for arm in
                  ('legacy_training', 'exact_only', 'observed_bounds', 'interval_bounds')}
        candidate, baseline = scores['interval_bounds'], scores['legacy_training']
        results[str(year)] = {'scores': scores, 'train_rows': len(train),
                             'train_intervals': int(train.reason.eq('interval').sum()),
                             'crossing_blocks': sorted(cross),
                             'passes_gate': bool(candidate['row_mae'] < baseline['row_mae'] and
                                 candidate['block_mae'] < baseline['block_mae'] and
                                 candidate['tail_24h_block_mae'] <= baseline['tail_24h_block_mae'])}
    assert before == {str(p): sha(p) for p in paths}
    report = {'years': results, 'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - started,
              'output_sha256': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
