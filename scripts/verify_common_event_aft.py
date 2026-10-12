"""Independently replay every reproduced AFT model and its training boundaries."""

import json

import joblib
import numpy as np
import pandas as pd

from scripts.build_strict_history_backtest import ROOT, SOURCES, prepare
from scripts.train_recovery_weather_formal import sha


def main():
    run = ROOT / 'artifacts/task01_recovery_head/aft_common_event_reproduction_v1'
    report = json.loads((run / 'report.json').read_bytes())
    protocol = json.loads((run / 'protocol.json').read_bytes())
    assert report['status'] == 'completed_reproduction_only'
    for name, value in protocol['input_sha256'].items():
        assert sha(ROOT / name) == value
    names = protocol['features']
    rows = pd.concat([prepare(pd.read_parquet(p), s, names) for s, p in SOURCES.items()], ignore_index=True)
    rows = rows.sort_values('timestamp').reset_index(drop=True)
    events = rows.groupby('event_key').timestamp.min()
    results = []
    for item in report['events']:
        key = item['event_key']
        folder = run / key.replace(':', '__')
        for name, field in [('model.joblib', 'model_sha256'), ('training_rows.parquet', 'training_rows_sha256'),
                            ('predictions.parquet', 'predictions_sha256')]:
            assert sha(folder / name) == item[field]
        training = pd.read_parquet(folder / 'training_rows.parquet')
        test_start = pd.Timestamp(item['event_start'])
        expected = rows.loc[rows.event_key.isin(events.index[events < test_start])]
        np.testing.assert_array_equal(training.ordered_row_index, expected.index)
        np.testing.assert_array_equal(training.event_key, expected.event_key)
        np.testing.assert_array_equal(training.remaining_hours, expected.remaining_hours)
        known = expected.timestamp + pd.to_timedelta(expected.remaining_hours + 5, unit='h')
        assert (known < test_start).all()
        np.testing.assert_array_equal(pd.DatetimeIndex(known).asi8,
                                      pd.DatetimeIndex(training.label_available_at).asi8)
        model = joblib.load(folder / 'model.joblib')
        test = rows.loc[rows.event_key.eq(key)]
        predicted = model.predict(test[names])
        saved = pd.read_parquet(folder / 'predictions.parquet')
        np.testing.assert_array_equal(test.timestamp, saved.timestamp)
        np.testing.assert_array_equal(predicted, saved.reproduced_prediction_hours)
        np.testing.assert_array_equal(predicted, saved.strict_prediction_hours)
        results.append({'event_key': key, 'rows': len(test), 'maximum_replay_difference': 0.0})
    assert len(results) == report['event_count'] == 22
    assert sum(item['rows'] for item in results) == report['all_event_rows'] == 579
    for name, value in protocol['input_sha256'].items():
        assert sha(ROOT / name) == value
    result = {'verification': 'PASS', 'event_count': len(results), 'all_event_rows': 579,
        'model_replay_maximum_difference': 0., 'training_label_maturity_verified': True,
        'source_report_sha256': sha(run / 'report.json'), 'verifier_sha256': sha(ROOT / 'scripts/verify_common_event_aft.py'),
        'limitations': ['reproduction_not_accuracy_improvement', 'shared_source_feature_preparation']}
    (run / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
