"""Replay AFT weighting checkpoints and independently compute development scores."""

import json

import joblib
import numpy as np
import pandas as pd

from scripts.build_strict_history_backtest import ROOT, SOURCES, prepare
from scripts.train_recovery_weather_formal import sha


def main():
    run = ROOT / 'artifacts/task01_recovery_head/aft_event_weight_development_v1'
    report = json.loads((run / 'report.json').read_bytes())
    protocol = json.loads((run / 'protocol.json').read_bytes())
    assert report['status'] == 'completed_2023_development_only'
    assert sha(run / 'protocol.json') == report['protocol_sha256']
    assert sha(run / 'predictions.parquet') == report['predictions_sha256']
    for name, digest in protocol['input_sha256'].items():
        assert sha(ROOT / name) == digest
    names = protocol['features']
    rows = pd.concat([prepare(pd.read_parquet(p), s, names) for s, p in SOURCES.items()], ignore_index=True)
    rows = rows.sort_values('timestamp').reset_index(drop=True).sort_values('timestamp').reset_index(drop=True)
    starts = rows.groupby('event_key').timestamp.min()
    for item in report['events']:
        key = item['event_key']
        folder = run / key.replace(':', '__')
        train = pd.read_parquet(folder / 'training_rows.parquet')
        expected = rows.loc[rows.event_key.isin(starts.index[starts < pd.Timestamp(item['event_start'])])]
        np.testing.assert_array_equal(train.ordered_row_index, expected.index)
        np.testing.assert_array_equal(train.event_key, expected.event_key)
        known = expected.timestamp + pd.to_timedelta(expected.remaining_hours + 5, unit='h')
        assert (known < pd.Timestamp(item['event_start'])).all()
        assert sha(folder / 'training_rows.parquet') == item['training_rows_sha256']
        assert sha(folder / 'predictions.parquet') == item['predictions_sha256']
        saved = pd.read_parquet(folder / 'predictions.parquet')
        test = rows.loc[rows.event_key.eq(key)]
        np.testing.assert_array_equal(saved.timestamp, test.timestamp)
        np.testing.assert_array_equal(saved.remaining_hours, test.remaining_hours)
        counts = expected.event_key.map(expected.event_key.value_counts()).to_numpy(float)
        for arm, power in protocol['arms'].items():
            weight = counts ** -power
            weight /= weight.mean()
            np.testing.assert_allclose(weight, train[arm + '_weight'], rtol=0, atol=1e-12)
            assert sha(folder / (arm + '.joblib')) == item['models'][arm]
            model = joblib.load(folder / (arm + '.joblib'))
            np.testing.assert_array_equal(model.predict(test[names]), saved[arm])
        np.testing.assert_array_equal(saved.row_control, saved.strict_prediction_hours)
    predictions = pd.read_parquet(run / 'predictions.parquet')
    assert predictions.timestamp.dt.year.eq(2023).all()
    assert not predictions.timestamp.duplicated().any()
    output = {}
    for arm in protocol['arms']:
        metrics = {}
        for prefix, cap in [('full', np.inf), ('cap68', 68)]:
            errors = abs(np.minimum(predictions.remaining_hours, cap) - np.minimum(predictions[arm], cap))
            metrics[prefix + '_row_mae'] = float(errors.mean())
            metrics[prefix + '_event_equal_mae'] = float(errors.groupby(predictions.event_key).mean().mean())
            if prefix == 'cap68':
                tail = predictions.remaining_hours.ge(68)
                metrics['cap68_tail_event_equal_mae'] = float(errors[tail].groupby(predictions.loc[tail, 'event_key']).mean().mean())
        for key, value in metrics.items():
            np.testing.assert_allclose(value, report['scores'][arm][key], rtol=0, atol=1e-12)
        output[arm] = metrics
    base = output['row_control']
    passed = [arm for arm, m in output.items() if arm != 'row_control'
              and m['full_row_mae'] < base['full_row_mae']
              and m['full_event_equal_mae'] < base['full_event_equal_mae']
              and m['cap68_tail_event_equal_mae'] <= base['cap68_tail_event_equal_mae']]
    passed.sort(key=lambda arm: (output[arm]['full_event_equal_mae'], output[arm]['full_row_mae'], protocol['arms'][arm]))
    assert passed == report['passed_arms']
    assert report['selected_arm'] == (passed[0] if passed else None)
    for name, digest in protocol['input_sha256'].items():
        assert sha(ROOT / name) == digest
    result = {'verification': 'PASS', 'scores': output, 'passed_arms': passed,
        'selected_arm': report['selected_arm'], 'verifier_sha256': sha(ROOT / 'scripts/verify_aft_event_weight.py'),
        'report_sha256': sha(run / 'report.json'), 'limitations': ['development_only', 'shared_feature_preparation']}
    (run / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
