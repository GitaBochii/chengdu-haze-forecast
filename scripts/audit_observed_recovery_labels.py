"""Materialize new observed-prefix targets without overwriting old training data."""

import json

import numpy as np
import pandas as pd

from haze_forecast.observed_recovery_labels import observed_recovery_bounds
from scripts.build_strict_history_backtest import ROOT
from scripts.train_recovery_weather_formal import sha


SOURCES = {
    '2014': ('cnemc_chengdu_2014_labeled.parquet', 'cnemc_chengdu_2014_training_features.parquet'),
    '2015_2021': ('city_labeled.parquet', 'training_features.parquet'),
    '2022': ('cnemc_chengdu_2022_labeled.parquet', 'cnemc_chengdu_2022_training_features.parquet'),
    '2023': ('cnemc_chengdu_2023_labeled.parquet', 'cnemc_chengdu_2023_training_features.parquet'),
}


def main():
    output = ROOT / 'artifacts/task01_recovery_head/observed_recovery_label_audit_v1'
    assert not output.exists()
    data = ROOT / 'data/processed'
    paths = [data / name for pair in SOURCES.values() for name in pair]
    paths += [ROOT / 'src/haze_forecast/labels.py', ROOT / 'src/haze_forecast/observed_recovery_labels.py',
              ROOT / 'scripts/audit_observed_recovery_labels.py']
    hashes = {str(path): sha(path) for path in paths}
    results, frames = {}, {}
    for source, (label_name, feature_name) in SOURCES.items():
        raw = pd.read_parquet(data / label_name)
        old = pd.read_parquet(data / feature_name)
        result = observed_recovery_bounds(raw[['timestamp', 'aqi']], old.timestamp)
        np.testing.assert_array_equal(result.timestamp, old.timestamp)
        result['old_remaining_hours'] = old.remaining_hours.to_numpy()
        result['old_event_id'] = old.event_id.to_numpy()
        exact = result.exact_hours.notna()
        changed = exact & ~np.isclose(result.exact_hours, result.old_remaining_hours)
        results[source] = {'rows': len(result), 'exact_rows': int(exact.sum()),
            'censored_rows': int((~exact).sum()), 'uninformative_rows': int((~result.informative).sum()),
            'changed_exact_rows': int(changed.sum()),
            'maximum_exact_label_change_hours': float(abs(result.loc[exact, 'exact_hours'] - result.loc[exact, 'old_remaining_hours']).max()),
            'reasons': {str(k): int(v) for k, v in result.reason.value_counts().items()}}
        assert (result.lower_bound_hours > 0).all()
        assert (result.label_available_at >= result.timestamp).all()
        frames[source] = result
    assert hashes == {str(path): sha(path) for path in paths}
    output.mkdir(parents=True)
    for source, frame in frames.items():
        frame.to_parquet(output / f'{source}.parquet', index=False)
    report = {'status': 'completed_separate_label_audit_not_model_training', 'sources': results,
        'input_sha256': hashes,
        'limitations': ['old_business_origins_only', 'does_not_rebuild_feature_windows_or_eligibility',
                       '2024_2025_not_materialized_from_an_unverified_source',
                       'conservative_first_gap_censoring', 'not_accuracy_improvement']}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
