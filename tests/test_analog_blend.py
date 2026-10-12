import numpy as np
import pandas as pd

from haze_forecast.analog_blend import MissingRegimeAnalogBlend


def test_analog_blend_only_adjusts_rows_in_the_missing_input_regime():
    model = MissingRegimeAnalogBlend(feature_names=["signal"], neighbors=1, blend_weight=0.5)
    model.fit(pd.DataFrame({"signal": [0.0, 10.0]}), pd.Series([20.0, 60.0]))

    result = model.predict(
        base_prediction=np.array([10.0, 10.0]),
        features=pd.DataFrame({"signal": [0.0, 10.0]}),
        apply_blend=np.array([True, False]),
    )

    assert result.tolist() == [15.0, 10.0]


def test_analog_blend_rejects_a_regime_mask_with_the_wrong_length():
    model = MissingRegimeAnalogBlend(feature_names=["signal"], neighbors=1, blend_weight=0.1)
    model.fit(pd.DataFrame({"signal": [0.0, 1.0]}), pd.Series([2.0, 3.0]))

    with np.testing.assert_raises_regex(ValueError, "apply_blend"):
        model.predict(
            base_prediction=np.array([1.0, 2.0]),
            features=pd.DataFrame({"signal": [0.0, 1.0]}),
            apply_blend=np.array([True]),
        )
