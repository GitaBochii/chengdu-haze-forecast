import pytest

from scripts.benchmark_grouped_forecast_aft import audit_paths, feature_groups


def test_groups_cover_schema_without_overlap():
    names = [f'h{i}' for i in range(92)] + [f'gefs_{i}' for i in range(9)] + [
        'forecast_known_age_hours', 'forecast_run_age_hours', 'forecast_remaining_lead_hours']
    groups = feature_groups(names)
    assert groups == [names[:92], names[92:]]
    with pytest.raises(ValueError):
        feature_groups(names[:-1])


def test_audit_rejects_mixed_path():
    tree = {'split': 'h', 'children': [
        {'leaf': 0}, {'split': 'w', 'children': [{'leaf': 1}, {'leaf': 2}]}]}
    with pytest.raises(ValueError, match='Mixed'):
        audit_paths([tree], [['h'], ['w']])


def test_audit_accepts_separate_trees_and_rejects_unknown():
    trees = [{'split': n, 'children': [{'leaf': 0}, {'leaf': 1}]} for n in ['h', 'w']]
    assert audit_paths(trees, [['h'], ['w']]) == {'trees': 2, 'leaf_paths': 4, 'mixed_paths': 0}
    with pytest.raises(ValueError, match='Unknown'):
        audit_paths(trees, [['h'], ['z']])
