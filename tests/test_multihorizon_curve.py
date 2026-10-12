import numpy as np
import pandas as pd

from haze_forecast.multihorizon_curve import (
    DirectCurveRegressor,
    add_future_targets,
    curve_recovery_or_fallback,
    recovery_hour_from_curve,
)


def test_future_targets_drop_rows_when_a_horizon_crosses_a_timestamp_gap():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"], utc=True
            ),
            "aqi": [160.0, 140.0, 90.0],
            "pm2_5": [100.0, 80.0, 30.0],
        }
    )

    result = add_future_targets(frame, horizons=(1,))

    assert result.loc[0, "aqi_target_h001"] == 140.0
    assert result.loc[1:, "aqi_target_h001"].isna().all()


def test_recovery_hour_requires_six_consecutive_predicted_hours_below_threshold():
    curve = np.array([130, 99, 98, 97, 96, 95, 94, 110, 90, 89, 88, 87, 86, 85], dtype=float)

    assert recovery_hour_from_curve(curve, threshold=100.0, stable_hours=6) == 2.0


def test_recovery_hour_returns_first_forecast_hour_when_six_hours_are_clean():
    assert recovery_hour_from_curve(np.full(6, 100.0)) == 1.0


def test_recovery_hour_does_not_bridge_missing_or_incomplete_clean_windows():
    assert recovery_hour_from_curve(np.array([90, 90, np.nan, 90, 90, 90])) is None
    assert recovery_hour_from_curve(np.full(5, 90.0)) is None


def test_direct_curve_fit_purges_targets_at_or_after_cutoff_per_horizon(monkeypatch):
    from lightgbm import LGBMRegressor

    timestamps = pd.date_range("2024-12-31 19:00", periods=9, freq="h", tz="UTC")
    frame = add_future_targets(
        pd.DataFrame({"timestamp": timestamps, "aqi": np.arange(9.0), "pm2_5": np.arange(9.0)}),
        horizons=(1, 3),
    )
    cutoff = pd.Timestamp("2025-01-01", tz="UTC")
    training_calls = []
    original_fit = LGBMRegressor.fit

    def recording_fit(self, features, target, **kwargs):
        training_calls.append((target.name, features.index.tolist(), target.tolist()))
        return original_fit(self, features, target, **kwargs)

    monkeypatch.setattr(LGBMRegressor, "fit", recording_fit)
    DirectCurveRegressor(horizons=(1, 3)).fit(frame, ["aqi"], training_cutoff=cutoff)

    assert training_calls == [
        ("aqi_target_h001", [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0]),
        ("aqi_target_h003", [0, 1], [3.0, 4.0]),
        ("pm2_5_target_h001", [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0]),
        ("pm2_5_target_h003", [0, 1], [3.0, 4.0]),
    ]


def test_direct_curve_regressor_returns_each_horizon_and_hybrid_keeps_long_event_fallback():
    train = pd.DataFrame(
        {
            "aqi": [150.0, 160.0, 170.0],
            "pm2_5": [90.0, 100.0, 110.0],
            "aqi_target_h001": [140.0, 150.0, 160.0],
            "pm2_5_target_h001": [80.0, 90.0, 100.0],
        }
    )
    model = DirectCurveRegressor(horizons=(1,)).fit(train, ["aqi", "pm2_5"])

    predicted = model.predict(train.iloc[[0]])

    assert set(predicted) == {"aqi_h001", "pm2_5_h001"}
    assert curve_recovery_or_fallback(np.full(24, 150.0), 40.0) == 40.0
