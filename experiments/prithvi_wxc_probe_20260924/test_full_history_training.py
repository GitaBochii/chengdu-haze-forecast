import numpy as np
import pandas as pd
import pytest
import torch

from experiments.prithvi_wxc_probe_20260924.run_full_history_training import (
    select_population, sample_blocks, unique_train_pair_mask, check_groups, merge_hashes,
)


def sample():
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-05-01', '2024-05-01', '2025-05-01'], utc=True),
        'partition': ['train', 'dev', 'retrospective_test'],
        'cross_partition_block': [False]*3,
        'label_mature_within_partition': [True]*3,
        'history_within_partition': [True]*3,
        'year_start_boundary_uncertain': [False]*3,
        'episode_id': [1, 2, 3], 'continuity_block': [1, 2, 3],
    })


def test_population_excludes_each_failure_and_preserves_input():
    p = sample()
    p.loc[0, 'year_start_boundary_uncertain'] = True
    p.loc[1, 'cross_partition_block'] = True
    kept, excluded = select_population(p)
    assert kept.index.tolist() == [2]
    assert excluded.index.tolist() == [0, 1]
    assert len(p) == 3


def test_groups_reject_cross_partition_episode():
    p = sample()
    p.loc[1, 'episode_id'] = 1
    with pytest.raises(ValueError, match='episode_id'):
        check_groups(p)


def test_pca_population_uses_unique_training_pairs_only():
    mask = unique_train_pair_mask(['train', 'train', 'dev', 'train'], ['a', 'a', 'b', 'c'])
    assert mask.tolist() == [True, False, False, True]


def test_block_sampler_balanced_reproducible():
    blocks = [np.array([0]), np.arange(1, 101)]
    a = sample_blocks(blocks, np.random.default_rng(24), 10000)
    b = sample_blocks(blocks, np.random.default_rng(24), 10000)
    assert np.array_equal(a, b)
    assert .47 < (a == 0).mean() < .53
    assert a.min() == 0 and a.max() <= 100


def test_provenance_cannot_be_silently_overwritten():
    with pytest.raises(ValueError, match='Conflicting provenance'):
        merge_hashes({'source':'old'}, {'source':'new'})
