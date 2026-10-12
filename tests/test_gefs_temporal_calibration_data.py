import pandas as pd

from scripts.build_gefs_temporal_calibration_data import temporal_split


def test_split_purges_crossing_block_and_immature_labels():
    times = pd.to_datetime(['2021-12-30', '2021-12-31', '2022-01-01', '2022-02-01', '2022-12-31', '2023-01-01']).tz_localize('Asia/Shanghai')
    frame = pd.DataFrame({'timestamp': times, 'label_available_at': times,
                          'continuity_block': [1, 2, 2, 3, 4, 4]})
    train, val = temporal_split(frame, frame)
    assert train.continuity_block.tolist() == [1]
    assert val.continuity_block.tolist() == [2, 3, 4]
    assert val.score_allowed.tolist() == [False, True, False]


def test_training_label_must_mature_before_cutoff():
    frame = pd.DataFrame({
        'timestamp': pd.to_datetime(['2021-12-01', '2021-12-02', '2022-02-01']).tz_localize('Asia/Shanghai'),
        'label_available_at': pd.to_datetime(['2021-12-02', '2022-01-01', '2022-02-02']).tz_localize('Asia/Shanghai'),
        'continuity_block': [1, 2, 3]})
    train, _ = temporal_split(frame, frame)
    assert train.continuity_block.tolist() == [1]
