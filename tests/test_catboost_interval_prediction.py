import numpy as np
import pandas as pd

from scripts.benchmark_catboost_interval import make_model, median_hours


def test_normal_aft_output_is_log_time():
    x = pd.DataFrame({'x': np.arange(80, dtype=float)})
    target = np.full((80, 2), 20.)
    model = make_model(iterations=300).set_params(allow_const_label=True).fit(x, target)
    prediction = median_hours(model, x)
    # Stochastic tree fitting need not interpolate every identical target exactly.
    np.testing.assert_allclose(np.median(prediction), 20., rtol=.05)
    np.testing.assert_allclose(np.log(prediction), model.predict(x, prediction_type='RawFormulaVal'))
