import json

import numpy as np
import pandas as pd
import pytest

from audit_full_training_results import audit, metrics, sha


def sample():
    return pd.DataFrame(dict(lower_bound_hours=[10., 20., 80.],
        upper_bound_hours=[10., 40., np.inf], reason=['exact', 'interval', 'right'],
        continuity_block=['a', 'a', 'b']))


def test_independent_interval_and_block_means():
    result = metrics(sample(), np.array([15., 30., 60.]))
    assert result['exact_row_mae_capped_hours'] == 5
    assert result['row_error_lower_hours'] == pytest.approx(13/3)
    assert result['row_error_upper_hours'] == pytest.approx(23/3)
    assert result['block_error_lower_hours'] == 5.25
    assert result['block_error_upper_hours'] == 7.75


@pytest.mark.parametrize('pred', [[0, 30, 60], [15, np.nan, 60], [15, 30, 69]])
def test_invalid_predictions_rejected(pred):
    with pytest.raises(ValueError):
        metrics(sample(), np.asarray(pred))


def test_no_exact_and_invalid_bounds():
    frame = sample()
    frame.reason = 'interval'
    assert metrics(frame, np.array([15., 30., 60.]))['exact_row_mae_capped_hours'] is None
    frame.loc[0, 'upper_bound_hours'] = 9
    with pytest.raises(ValueError):
        metrics(frame, np.array([15., 30., 60.]))


def make_run(path):
    frame = pd.concat([sample().assign(partition=p, timestamp=pd.date_range(
        f'{2010+i}-01-01', periods=3, freq='h', tz='UTC'),
        continuity_block=[f'{p}_a', f'{p}_a', f'{p}_b'])
        for i, p in enumerate(('train', 'dev', 'retrospective_test', 'recent_audit'))], ignore_index=True)
    frame.to_parquet(path/'selected.parquet')
    frame.iloc[:0].to_parquet(path/'excluded.parquet')
    runs = []
    for seed in (24, 42, 2026):
        for arm in ('baseline', 'capacity_control', 'spatial_fusion'):
            checkpoint = path/f'{seed}_{arm}.pt'
            checkpoint.write_bytes(f'{seed}_{arm}'.encode())
            preds = frame.assign(predicted_capped_median_hours=np.tile([15., 30., 60.], 4))
            preds.to_parquet(path/f'{seed}_{arm}_predictions.parquet')
            runs.append(dict(seed=seed, arm=arm, checkpoint_sha256=sha(checkpoint),
                summary={p: metrics(g, g.predicted_capped_median_hours.to_numpy())
                    for p,g in preds.groupby('partition')}))
    (path/'report.json').write_text(json.dumps(dict(status='completed_offline_fixed_protocol',
        runs=runs, counts=frame.groupby('partition').size().to_dict(), excluded_count=0)))


def test_full_artifact_audit_and_changed_checkpoint(tmp_path):
    make_run(tmp_path)
    result = audit(tmp_path)
    assert result['status'] == 'pass'
    assert result['summary']['train']['baseline']['row_error_lower_hours']['std'] == 0
    (tmp_path/'24_baseline.pt').write_bytes(b'changed')
    assert audit(tmp_path)['status'] == 'fail'


def test_missing_final_report_is_not_complete(tmp_path):
    assert audit(tmp_path)['status'] == 'fail'


def test_changed_labels_and_missing_arm_rejected(tmp_path):
    make_run(tmp_path)
    path = tmp_path/'24_baseline_predictions.parquet'
    frame = pd.read_parquet(path)
    frame.loc[0, 'lower_bound_hours'] = 11
    frame.to_parquet(path)
    (tmp_path/'42_capacity_control_predictions.parquet').unlink()
    result = audit(tmp_path)
    assert result['status'] == 'fail'
    assert len(result['errors']) >= 2
