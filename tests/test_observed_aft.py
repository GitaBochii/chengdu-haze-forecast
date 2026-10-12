import numpy as np
import pandas as pd
import pytest

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.training import AFTSurvivalRegressor


def data():
    x = pd.DataFrame({"x": np.arange(80, dtype=float), "z": np.arange(80) % 7})
    x.loc[2, "x"] = np.nan
    y = pd.Series(np.arange(80) % 24 + 1.0)
    return x, y


def test_defaults_and_inherited_fit_are_unchanged():
    model = ObservedAFTSurvivalRegressor()
    assert (model.num_boost_round, model.seed, model.nthread) == (500, 42, None)
    assert ObservedAFTSurvivalRegressor.fit is AFTSurvivalRegressor.fit
    assert ObservedAFTSurvivalRegressor.predict is AFTSurvivalRegressor.predict


@pytest.mark.parametrize("weighted", [False, True])
def test_exact_fit_matches_original_elementwise(weighted):
    x, y = data()
    weights = np.linspace(0, 2, len(x)) if weighted else None
    old = AFTSurvivalRegressor(num_boost_round=8, nthread=1).fit(x, y, weights)
    new = ObservedAFTSurvivalRegressor(num_boost_round=8, nthread=1)
    assert new.fit_bounds(x, y, y, weights) is new
    np.testing.assert_array_equal(new.predict(x), old.predict(x))


def test_censoring_and_interval_bounds_reach_real_xgboost(monkeypatch):
    import haze_forecast.observed_aft as module

    x, lower = data()
    upper = lower.copy()
    upper.iloc[::3] = np.inf
    upper.iloc[1::3] += 6
    original_train = module.xgb.train
    seen = []

    def capture(params, matrix, **kwargs):
        np.testing.assert_array_equal(matrix.get_float_info("label_lower_bound"), lower)
        np.testing.assert_array_equal(matrix.get_float_info("label_upper_bound"), upper)
        seen.append(True)
        return original_train(params, matrix, **kwargs)

    monkeypatch.setattr(module.xgb, "train", capture)
    model = ObservedAFTSurvivalRegressor(num_boost_round=8, nthread=1)
    model.fit_bounds(x, lower, upper)
    assert seen == [True]
    assert np.isfinite(model.predict(x)).all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("lower", [0, 1]), ("lower", [-1, 1]), ("lower", [np.nan, 1]),
        ("lower", [np.inf, 1]), ("lower", [[1], [1]]), ("lower", [1]),
        ("upper", [np.nan, 1]), ("upper", [-np.inf, 1]),
        ("upper", [0.5, 1]), ("upper", [[1], [1]]), ("upper", [1]),
        ("weights", [0, 0]), ("weights", [-1, 1]),
        ("weights", [np.inf, 1]), ("weights", [np.nan, 1]),
        ("weights", [[1], [1]]), ("weights", [1]),
        ("features", pd.DataFrame({"x": [np.inf, 1]})),
        ("features", pd.DataFrame({"x": [-np.inf, 1]})),
        ("features", pd.DataFrame({"x": ["bad", 1]})),
        ("features", pd.DataFrame(index=[0, 1])),
        ("features", pd.DataFrame({"x": []})),
    ],
)
def test_invalid_inputs_rejected_before_training(monkeypatch, field, value):
    import haze_forecast.observed_aft as module

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid inputs reached XGBoost training")

    monkeypatch.setattr(module.xgb, "train", forbidden)
    args = {"features": pd.DataFrame({"x": [1, 2]}), "lower": [1, 1],
            "upper": [1, 1], "weights": None}
    args[field] = value
    with pytest.raises(ValueError):
        ObservedAFTSurvivalRegressor().fit_bounds(
            args["features"], args["lower"], args["upper"], args["weights"]
        )


@pytest.mark.parametrize("field", ["lower", "upper", "weight"])
def test_series_index_misalignment_is_rejected(field):
    x = pd.DataFrame({"x": [1, 2]}, index=[3, 4])
    args = {"lower": [1, 1], "upper": [1, 1], "weight": [1, 1]}
    args[field] = pd.Series([1, 1])
    with pytest.raises(ValueError, match="index"):
        ObservedAFTSurvivalRegressor().fit_bounds(
            x, args["lower"], args["upper"], args["weight"]
        )


@pytest.mark.parametrize("lower,upper", [(1e-100, 1), (1, 1e100), (1e100, np.inf)])
def test_bounds_must_survive_xgboost_float32_storage(lower, upper):
    with pytest.raises(ValueError, match="float32"):
        ObservedAFTSurvivalRegressor(nthread=1).fit_bounds(
            pd.DataFrame({"x": [1]}), [lower], [upper]
        )
