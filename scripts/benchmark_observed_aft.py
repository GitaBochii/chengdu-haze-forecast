"""Fixed-year diagnostic separating label changes from training-row removal."""

import json

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.training import AFTSurvivalRegressor
from scripts.build_strict_history_backtest import ROOT, prepare
from scripts.train_recovery_weather_formal import sha


def main():
    output = ROOT / 'artifacts/task01_recovery_head/observed_aft_development_v1'
    assert not output.exists()
    labels_dir = ROOT / 'artifacts/task01_recovery_head/observed_recovery_label_audit_v1'
    inputs = {'2014': 'cnemc_chengdu_2014_training_features.parquet',
              '2015_2021': 'training_features.parquet', '2022': 'cnemc_chengdu_2022_training_features.parquet',
              '2023': 'cnemc_chengdu_2023_training_features.parquet'}
    metadata = ROOT / 'artifacts/cnemc_2025_deployment_candidate/model_metadata.json'
    names = json.loads(metadata.read_bytes())['feature_names']
    paths = [metadata, labels_dir / 'report.json', ROOT / 'scripts/benchmark_observed_aft.py',
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py']
    paths += [ROOT / 'data/processed' / p for p in inputs.values()]
    paths += [labels_dir / f'{s}.parquet' for s in inputs]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    frames = {}
    for source, filename in inputs.items():
        features = prepare(pd.read_parquet(ROOT / 'data/processed' / filename), source, names)
        bounds = pd.read_parquet(labels_dir / f'{source}.parquet')
        np.testing.assert_array_equal(features.timestamp, bounds.timestamp)
        for column in bounds.columns.difference(['timestamp', 'old_event_id', 'old_remaining_hours']):
            features[column] = bounds[column].to_numpy()
        frames[source] = features
    train = pd.concat([frames[s] for s in ('2014', '2015_2021', '2022')], ignore_index=True)
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert (train.timestamp < cutoff).all()
    assert (train.label_available_at < cutoff).all()
    assert ((train.timestamp + pd.to_timedelta(train.remaining_hours + 5, unit='h')) < cutoff).all()
    validation = frames['2023'].copy()
    assert validation.timestamp.dt.year.eq(2023).all()
    exact = validation.exact_hours.notna()
    assert exact.sum() == 266
    arms = {'legacy_full': np.ones(len(train), dtype=bool),
            'legacy_informative': train.informative.to_numpy(),
            'exact_only': train.exact_hours.notna().to_numpy(),
            'observed_bounds': train.informative.to_numpy()}
    protocol = {'input_sha256': hashes, 'features': names,
        'train_years': '2014-2022', 'validation_year': 2023, 'seed': 42, 'rounds': 500,
        'arms': list(arms), 'selection': 'none_diagnostic_only',
        'primary_metrics': 'full_duration_row_and_event_equal_mae_on_same_exact_rows',
        'limitations': ['historical_feature_windows_and_eligibility_not_rebuilt',
                        'censoring_may_be_informative', 'exact_subset_not_full_population',
                        'not_comparable_to_expanding_event_refit_scores', 'no_deployment']}
    output.mkdir(parents=True)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train.to_parquet(output / 'training_rows.parquet', index=False)
    scores = {}
    for arm, selected in arms.items():
        subset = train.loc[selected]
        if arm == 'observed_bounds':
            model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(
                subset[names], subset.lower_bound_hours, subset.upper_bound_hours)
        else:
            target = subset.exact_hours if arm == 'exact_only' else subset.remaining_hours
            model = AFTSurvivalRegressor(seed=42, nthread=1).fit(subset[names], target)
        joblib.dump(model, output / f'{arm}.joblib')
        pred = model.predict(validation[names])
        np.testing.assert_array_equal(pred, joblib.load(output / f'{arm}.joblib').predict(validation[names]))
        assert np.isfinite(pred).all()
        validation[arm] = pred
        error = abs(validation.loc[exact, 'exact_hours'] - pred[exact])
        censored = validation.exact_hours.isna() & validation.informative
        scores[arm] = {'train_rows': len(subset), 'exact_score_rows': int(exact.sum()),
            'exact_score_events': int(validation.loc[exact, 'event_key'].nunique()),
            'row_mae': float(error.mean()),
            'event_equal_mae': float(error.groupby(validation.loc[exact, 'event_key']).mean().mean()),
            'censored_informative_rows': int(censored.sum()),
            'below_censor_bound_fraction': float((pred[censored] < validation.loc[censored, 'lower_bound_hours']).mean()),
            'checkpoint_sha256': sha(output / f'{arm}.joblib')}
    validation.to_parquet(output / 'validation.parquet', index=False)
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_fixed_year_label_diagnostic', 'scores': scores,
        'protocol_sha256': sha(output / 'protocol.json'),
        'validation_sha256': sha(output / 'validation.parquet'),
        'training_sha256': sha(output / 'training_rows.parquet'), 'inputs_unchanged': True}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
