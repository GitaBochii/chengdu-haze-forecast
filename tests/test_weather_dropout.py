import numpy as np
import pandas as pd

from scripts.benchmark_weather_dropout import augmented_training


def test_masking_preserves_other_fields_labels_and_weight_per_origin():
    frame = pd.DataFrame({'aqi': [130., 140.], 'temperature_c': [10., 12.],
                          'wind_speed_ms': [2., np.nan], 'lower_bound_hours': [10., 20.],
                          'upper_bound_hours': [10., 25.]})
    names = ['aqi', 'temperature_c', 'wind_speed_ms']
    masked, weights = augmented_training(frame, names, masked=True)
    duplicate, control_weights = augmented_training(frame, names, masked=False)
    pd.testing.assert_frame_equal(masked.iloc[:2], frame)
    assert masked.iloc[2:][names[1:]].isna().all().all()
    pd.testing.assert_frame_equal(masked[['aqi', 'lower_bound_hours', 'upper_bound_hours']],
                                  duplicate[['aqi', 'lower_bound_hours', 'upper_bound_hours']])
    pd.testing.assert_frame_equal(duplicate.iloc[2:].reset_index(drop=True), frame)
    np.testing.assert_array_equal(weights, control_weights)
    np.testing.assert_array_equal(weights[:2] + weights[2:], [1., 1.])
