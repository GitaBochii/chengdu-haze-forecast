import pandas as pd
import pytest

from scripts.compare_regional_air_history import aligned_duration_scores, validate_reports


def test_comparison_uses_only_shared_verified_keys_and_rejects_changed_truth():
    times = pd.date_range('2025-01-01', periods=3, freq='h', tz='Asia/Shanghai')
    full = pd.DataFrame({'timestamp': times, 'event_id': [1, 1, 1], 'actual_hours': [10., 9., 8.],
                         'predicted_hours': [10., 10., 10.],
                         'label_verified_in_scoring_window': [True, True, False]})
    recent = full.iloc[1:].copy()
    recent['predicted_hours'] = 12.
    scores = aligned_duration_scores({'full': full, 'recent': recent})
    assert scores['common_event_rows'] == 1
    assert scores['models']['full']['restricted_time_mae'] == 1.
    assert scores['models']['recent']['restricted_time_mae'] == 3.
    recent.loc[1, 'actual_hours'] = 90.
    with pytest.raises(ValueError, match='truth'):
        aligned_duration_scores({'full': full, 'recent': recent})


def test_empty_intersection_rejected():
    frame = pd.DataFrame({'timestamp': [pd.Timestamp('2025-01-01')], 'event_id': [1],
                          'actual_hours': [1.], 'predicted_hours': [1.],
                          'label_verified_in_scoring_window': [False]})
    with pytest.raises(ValueError, match='no common'):
        aligned_duration_scores({'full': frame, 'recent': frame})


def test_protocol_rejects_wrong_years_and_budget():
    from copy import deepcopy
    shared = {'history': 24, 'horizon': 72, 'ozone_column': 'o3_8h', 'validation_stride': 6,
              'eval_stride': 6, 'seed': 42, 'epochs': 24, 'patience': 5, 'learning_rate': .001,
              'batch_size': 256, 'train_stride': 6, 'max_train': 0, 'max_eval': 0}
    reports = {name: {'deployment_artifact_changed': False, 'input_data_sha256': {'a': 'b'},
                      'configuration': dict(shared, train_start_year=start, train_end_year=2023)}
               for name, start in [('full', 2014), ('recent', 2023)]}
    validate_reports(reports)
    bad = deepcopy(reports)
    bad['full']['configuration']['train_start_year'] = 2023
    with pytest.raises(ValueError, match='training years'):
        validate_reports(bad)
    reports['recent']['configuration']['epochs'] = 1
    with pytest.raises(ValueError, match='epochs'):
        validate_reports(reports)
