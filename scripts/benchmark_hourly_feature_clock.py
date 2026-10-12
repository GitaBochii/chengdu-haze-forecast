"""Compare row-clock and hour-clock inputs on fixed common eligible origins."""

import json

import joblib
import numpy as np
import pandas as pd

from haze_forecast.features import MEASUREMENT_COLUMNS
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.observed_hourly_features import build_observed_hourly_features
from scripts.audit_observed_recovery_labels import SOURCES
from scripts.build_strict_history_backtest import ROOT, prepare
from scripts.train_recovery_weather_formal import sha


def main():
    output = ROOT / 'artifacts/task01_recovery_head/hourly_feature_clock_v1'
    assert not output.exists()
    data = ROOT / 'data/processed'
    label_dir = ROOT / 'artifacts/task01_recovery_head/observed_recovery_label_audit_v1'
    metadata = ROOT / 'artifacts/cnemc_2025_deployment_candidate/model_metadata.json'
    names = json.loads(metadata.read_bytes())['feature_names']
    paths = [metadata, ROOT / 'scripts/benchmark_hourly_feature_clock.py',
             ROOT / 'src/haze_forecast/observed_hourly_features.py',
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/features.py',
             ROOT / 'src/haze_forecast/labels.py']
    paths += [data / name for pair in SOURCES.values() for name in pair]
    paths += [label_dir / f'{s}.parquet' for s in SOURCES]
    enriched_sources = {s: ROOT / f'data/interim/cnemc_chengdu_{s}_weather.parquet' for s in ('2014', '2022')}
    paths += list(enriched_sources.values())
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    old_parts, grid_parts, audits = {}, {}, {}
    for source, (raw_name, feature_name) in SOURCES.items():
        old = prepare(pd.read_parquet(data / feature_name), source, names)
        bounds = pd.read_parquet(label_dir / f'{source}.parquet')
        np.testing.assert_array_equal(old.timestamp, bounds.timestamp)
        raw = pd.read_parquet(enriched_sources.get(source, data / raw_name))
        observed = raw.set_index('timestamp').reindex(pd.DatetimeIndex(old.timestamp))
        for column in MEASUREMENT_COLUMNS:
            if column in old:
                np.testing.assert_allclose(observed[column].to_numpy(float), old[column].to_numpy(float), equal_nan=True)
        hourly = build_observed_hourly_features(raw)
        aligned = hourly.set_index('timestamp').reindex(pd.DatetimeIndex(old.timestamp)).reset_index()
        aligned['event_id'] = old.event_id.to_numpy()
        grid = prepare(aligned, source, names)
        eligible = grid.aqi.gt(100) & grid.particle_primary.fillna(False)
        for target in (old, grid):
            for key in bounds.columns.difference(['timestamp', 'old_event_id', 'old_remaining_hours']):
                target[key] = bounds[key].to_numpy()
        audits[source] = {'old_business_rows': len(old), 'common_eligible_rows': int(eligible.sum()),
            'excluded_under_hourly_eligibility': int((~eligible).sum()),
            'changed_feature_cells': int((~np.isclose(old[names].to_numpy(float), grid[names].to_numpy(float), equal_nan=True)).sum())}
        old_parts[source], grid_parts[source] = old.loc[eligible].copy(), grid.loc[eligible].copy()
    training_sources = ('2014', '2015_2021', '2022')
    train = {kind: pd.concat([parts[s] for s in training_sources], ignore_index=True)
             for kind, parts in [('old', old_parts), ('grid', grid_parts)]}
    val = {'old': old_parts['2023'].reset_index(drop=True), 'grid': grid_parts['2023'].reset_index(drop=True)}
    np.testing.assert_array_equal(train['old'].timestamp, train['grid'].timestamp)
    np.testing.assert_array_equal(val['old'].timestamp, val['grid'].timestamp)
    assert (train['grid'].label_available_at < pd.Timestamp('2023-01-01', tz='Asia/Shanghai')).all()
    output.mkdir(parents=True)
    protocol = {'input_sha256': hashes, 'features': names, 'audits': audits,
        'train_years': '2014-2022', 'validation_year': 2023, 'arms': ['old_exact', 'grid_exact', 'old_bounds', 'grid_bounds'],
        'seed': 42, 'rounds': 500, 'selection': 'none_diagnostic',
        'limitations': ['common_old_and_new_eligibility_only', 'not_full_new_business_population',
                       'exact_validation_subset', 'no_deployment']}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scores = {}
    result = val['old'][['timestamp', 'event_key', 'exact_hours', 'lower_bound_hours', 'upper_bound_hours', 'informative']].copy()
    for kind in ('old', 'grid'):
        train[kind].to_parquet(output / f'{kind}_training.parquet', index=False)
        val[kind].to_parquet(output / f'{kind}_validation.parquet', index=False)
        for label in ('exact', 'bounds'):
            arm = f'{kind}_{label}'
            selected = train[kind].exact_hours.notna() if label == 'exact' else train[kind].informative
            fit = train[kind].loc[selected]
            model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(fit[names], fit.lower_bound_hours, fit.upper_bound_hours)
            joblib.dump(model, output / f'{arm}.joblib')
            prediction = model.predict(val[kind][names])
            np.testing.assert_array_equal(prediction, joblib.load(output / f'{arm}.joblib').predict(val[kind][names]))
            result[arm] = prediction
            exact = result.exact_hours.notna()
            error = abs(prediction[exact] - result.loc[exact, 'exact_hours'])
            scores[arm] = {'train_rows': len(fit), 'score_rows': int(exact.sum()),
                'score_events': int(result.loc[exact, 'event_key'].nunique()),
                'row_mae': float(error.mean()),
                'event_equal_mae': float(error.groupby(result.loc[exact, 'event_key']).mean().mean()),
                'checkpoint_sha256': sha(output / f'{arm}.joblib')}
    result.to_parquet(output / 'predictions.parquet', index=False)
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_common_eligibility_clock_diagnostic', 'scores': scores, 'audits': audits,
        'protocol_sha256': sha(output / 'protocol.json'), 'predictions_sha256': sha(output / 'predictions.parquet')}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
