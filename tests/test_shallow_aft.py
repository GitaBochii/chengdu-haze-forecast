import pytest

from scripts.benchmark_shallow_aft import shallow_params, tree_depth


def test_only_depth_changes_and_input_unchanged():
    original = {'max_depth': 4, 'eta': .05, 'seed': 42}
    candidate = shallow_params(original)
    assert candidate == {'max_depth': 2, 'eta': .05, 'seed': 42}
    assert original['max_depth'] == 4
    with pytest.raises(ValueError):
        shallow_params({'max_depth': 3})
    with pytest.raises(ValueError):
        shallow_params({'max_depth': 4, 'interaction_constraints': [['a']]})


def test_actual_tree_depth():
    leaf = {'leaf': .5}
    assert tree_depth(leaf) == 0
    assert tree_depth({'split': 'a', 'children': [leaf, leaf]}) == 1
    assert tree_depth({'split': 'a', 'children': [leaf, {
        'split': 'b', 'children': [leaf, leaf]}]}) == 2
