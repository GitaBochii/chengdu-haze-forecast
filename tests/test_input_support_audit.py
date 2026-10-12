import numpy as np

from scripts.audit_frozen_input_support import process_distances


def test_same_process_cannot_supply_its_own_support():
    distance, blocks = process_distances(
        np.array([[0.], [.1], [2.]]), np.array([[0.]]),
        np.array([1, 1, 2]), np.array([1]),
    )
    assert blocks.tolist() == [1, 2]
    assert np.isinf(distance[0, 0])
    assert distance[0, 1] == 4.


def test_repeated_hours_count_as_one_process():
    distance, blocks = process_distances(
        np.array([[0.], [0.], [2.], [4.]]), np.array([[1.], [3.]]),
        np.array([1, 1, 2, 3]), np.array([9, 9]),
    )
    np.testing.assert_array_equal(distance, [[1., 1., 9.], [9., 1., 1.]])
    assert len(blocks) == 3
