"""Annual expanding training comparison without hyperparameter selection."""

import json

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.training import AFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import DATA, ROOT, metrics, sha
from scripts.build_strict_history_backtest import SOURCES, prepare

OUT = ROOT / 'artifacts/task01_recovery_head/observed_annual_update_v1'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    names = json.loads((DATA / 'report.json').read_bytes())['feature_names']
    full = pd.read_parquet(DATA / 'features.parquet')
    origins = pd.read_parquet(DATA / 'origins.parquet')
    old = pd.concat([prepare(pd.read_parquet(p), k, names) for k, p in SOURCES.items()], ignore_index=True)
    assert old.timestamp.is_unique
    old = old.merge(full[['timestamp', 'continuity_block']], on='timestamp', validate='one_to_one')
    old['legacy_confirmed_at'] = old.timestamp + pd.to_timedelta(old.remaining_hours + 5, unit='h')
    old['legacy_event_available_at'] = old.groupby('event_key').legacy_confirmed_at.transform('max')
    paths = list(SOURCES.values()) + [DATA / n for n in ('features.parquet', 'origins.parquet', 'report.json')]
    paths += [ROOT / p for p in ('scripts/benchmark_observed_annual_update.py',
        'scripts/benchmark_observed_business_aft.py', 'scripts/build_strict_history_backtest.py',
        'src/haze_forecast/training.py', 'src/haze_forecast/observed_aft.py',
        'src/haze_forecast/features.py', 'src/haze_forecast/app.py')]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    protocol = {'input_sha256': before, 'features': names, 'seed': 42, 'rounds': 500,
        'years': [2023, 2024, 2025], 'arms': ['legacy_training', 'exact_only', 'observed_bounds'],
        'training': 'expanding from 2014; January 1 cutoff; mature labels only; purge crossing blocks',
        'evaluation': 'all origins within year; exact outcomes when eventually observed; no end-year score purge',
        'selection': 'none; repeated historical diagnostic; not pristine holdout',
        'gate': 'both main MAEs improve each year; >=24h block MAE not worse each year',
        'limitations': ['legacy control retains old feature clock and label errors',
                       'all arms infer same new hourly features, not actual webpage replay',
                       'exact scores exclude censored outcomes', 'O3 source semantics differ']}
    OUT.mkdir(parents=True)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    reports = {}
    for year in protocol['years']:
        cutoff = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        cross = set(full.loc[full.timestamp < cutoff, 'continuity_block']) & set(full.loc[full.timestamp >= cutoff, 'continuity_block'])
        train = origins.loc[origins.timestamp.lt(cutoff) & origins.label_available_at.lt(cutoff)
                            & origins.informative & ~origins.continuity_block.isin(cross)].copy()
        legacy = old.loc[old.timestamp.lt(cutoff) & old.legacy_event_available_at.lt(cutoff)
                         & ~old.continuity_block.isin(cross)].copy()
        validation = origins.loc[origins.timestamp.dt.year.eq(year)].copy().reset_index(drop=True)
        assert not set(train.continuity_block) & set(validation.continuity_block)
        assert not set(legacy.continuity_block) & set(validation.continuity_block)
        dest = OUT / str(year)
        dest.mkdir()
        train.to_parquet(dest / 'training_rows.parquet', index=False)
        legacy.to_parquet(dest / 'legacy_training_rows.parquet', index=False)
        scores = {}
        for arm in protocol['arms']:
            if arm == 'legacy_training':
                selected = legacy
                model = AFTSurvivalRegressor(seed=42, nthread=1).fit(selected[names], selected.remaining_hours)
            else:
                selected = train if arm == 'observed_bounds' else train.loc[train.exact_hours.notna()]
                model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(
                    selected[names], selected.lower_bound_hours, selected.upper_bound_hours)
            joblib.dump(model, dest / f'{arm}.joblib')
            prediction = model.predict(validation[names])
            assert np.isfinite(prediction).all()
            validation[arm] = prediction
            scores[arm] = metrics(validation, prediction)
            scores[arm]['training_rows'] = len(selected)
        validation.to_parquet(dest / 'predictions.parquet', index=False)
        reports[str(year)] = {'scores': scores, 'crossing_blocks': sorted(cross),
                              'origins': len(validation)}
    assert before == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_annual_update_diagnostic', 'years': reports,
              'input_unchanged': True,
              'output_sha256': {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
