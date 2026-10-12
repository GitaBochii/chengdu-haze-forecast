import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_gefs_2020_extension import extend_training


def fixture():
    times = pd.to_datetime(['2020-10-24', '2021-01-02']).tz_localize('Asia/Shanghai')
    train = pd.DataFrame({'timestamp': times, 'aqi': [150., 160.], 'gefs_x': [np.nan, 2.]})
    features = pd.DataFrame({'timestamp': times[:1], 'gefs_x': [7.]})
    return train, features


def test_only_authorized_cells_change():
    train, features = fixture()
    before = train.copy(deep=True)
    result = extend_training(train, features, ['gefs_x'], expected_rows=1)
    assert result.gefs_x.tolist() == [7., 2.]
    pd.testing.assert_frame_equal(train, before)
    pd.testing.assert_series_equal(result.aqi, before.aqi)


def test_existing_values_not_overwritten():
    train, features = fixture()
    train.loc[0, 'gefs_x'] = 3.
    with pytest.raises(ValueError):
        extend_training(train, features, ['gefs_x'], expected_rows=1)


def test_wrong_year_rejected():
    train, features = fixture()
    features.timestamp = train.timestamp.iloc[1]
    with pytest.raises(ValueError):
        extend_training(train, features, ['gefs_x'], expected_rows=1)


def test_duplicate_and_missing_origins_rejected():
    train, features = fixture()
    with pytest.raises(ValueError):
        extend_training(train, pd.concat([features, features]), ['gefs_x'], expected_rows=2)
    features.timestamp = pd.Timestamp('2020-11-01', tz='Asia/Shanghai')
    with pytest.raises(ValueError):
        extend_training(train, features, ['gefs_x'], expected_rows=1)
