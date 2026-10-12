from copy import deepcopy

import pytest

from scripts.benchmark_recovery_auxiliary import compare_fold


def test_control_matching_allows_only_auxiliary_change():
    keys = ('nodes', 'feature_names', 'coordinates_lon_lat', 'scaler', 'partitions',
            'training_windows_by_year', 'training_label_audit', 'validation_label_audit')
    config_keys = ('data', 'epochs', 'batch_size', 'learning_rate', 'seed', 'device', 'history', 'horizon')
    control = {key: {'sentinel': 1} for key in keys}
    control['configuration'] = dict.fromkeys(config_keys, 1)
    control['configuration']['auxiliary_weight'] = .1
    candidate = deepcopy(control)
    candidate['configuration']['auxiliary_weight'] = 0.
    compare_fold(control, candidate)
    candidate['configuration']['seed'] = 43
    with pytest.raises(ValueError, match='configuration mismatch: seed'):
        compare_fold(control, candidate)
    candidate = deepcopy(control)
    candidate['scaler'] = {'sentinel': 2}
    with pytest.raises(ValueError, match='control mismatch: scaler'):
        compare_fold(control, candidate)


def test_saved_year_keys_match_in_memory_year_keys():
    keys = ('nodes', 'feature_names', 'coordinates_lon_lat', 'scaler', 'partitions',
            'training_windows_by_year', 'training_label_audit', 'validation_label_audit')
    config_keys = ('data', 'epochs', 'batch_size', 'learning_rate', 'seed', 'device', 'history', 'horizon')
    control = dict.fromkeys(keys, None)
    control['configuration'] = dict.fromkeys(config_keys, 1)
    control['training_windows_by_year'] = {'2014': 878}
    candidate = deepcopy(control)
    candidate['training_windows_by_year'] = {2014: 878}
    compare_fold(control, candidate)
