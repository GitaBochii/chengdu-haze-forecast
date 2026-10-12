"""Compare direct short-horizon pollutant curves with the deployed AFT protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators, build_features
from haze_forecast.labels import label_city_data
from haze_forecast.multihorizon_curve import (
    HORIZONS,
    DirectCurveRegressor,
    add_future_targets,
    curve_recovery_or_fallback,
    select_hybrid,
)
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts" / "multihorizon_pollutant_curve_corrected"
PBL_COLUMNS = [
    "boundary_layer_height_m", "boundary_layer_height_m_lag_1", "boundary_layer_height_m_lag_3",
    "boundary_layer_height_m_lag_24", "boundary_layer_height_m_roll_3",
    "boundary_layer_height_m_roll_24", "boundary_layer_height_m_change_3",
]
CURVE_FEATURES = [
    "aqi", "pm2_5", "pm10", "so2", "no2", "co", "o3", "temperature_c",
    "relative_humidity_pct", "wind_u_ms", "wind_v_ms", "boundary_layer_height_m",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "wind_speed_ms",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_24", "aqi_roll_3", "aqi_roll_24",
    "pm2_5_lag_1", "pm2_5_lag_3", "pm2_5_lag_24", "pm2_5_roll_3", "pm2_5_roll_24",
]


def _continuous_frame(path: Path, *, relabel: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    labelled = label_city_data(frame) if relabel else frame
    return add_future_targets(build_features(labelled))


def load_continuous() -> pd.DataFrame:
    frames = [
        _continuous_frame(DATA / "processed" / "city_labeled.parquet", relabel=False),
        _continuous_frame(DATA / "interim" / "cnemc_chengdu_2022_weather.parquet", relabel=True),
        _continuous_frame(DATA / "interim" / "cnemc_chengdu_2023_weather.parquet", relabel=True),
        _continuous_frame(DATA / "interim" / "cnemc_chengdu_2024_2025_weather.parquet", relabel=True),
    ]
    result = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    missing = set(CURVE_FEATURES) - set(result.columns)
    if missing:
        raise ValueError(f"continuous curve features missing columns: {sorted(missing)}")
    return result


def _prepare_events(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["event_key"] = source + ":" + result["event_id"].astype(str)
    result = add_missingness_indicators(result, PBL_COLUMNS)
    result["boundary_layer_height_m_change_3_is_missing"] = (
        result["boundary_layer_height_m"].isna() | result["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return result


def load_event_rows() -> pd.DataFrame:
    processed = DATA / "processed"
    sources = {
        "prefix_2014": processed / "cnemc_chengdu_2014_training_features.parquet",
        "base": processed / "training_features.parquet",
        "external_2022": processed / "cnemc_chengdu_2022_training_features.parquet",
        "external_2023": processed / "cnemc_chengdu_2023_training_features.parquet",
        "external_2024_2025": processed / "cnemc_chengdu_2024_2025_training_features.parquet",
    }
    return pd.concat(
        [_prepare_events(pd.read_parquet(path), source) for source, path in sources.items()],
        ignore_index=True,
    ).sort_values("timestamp").reset_index(drop=True)


def _curve_hours(predictions: dict[str, np.ndarray], row: int, current_aqi: float) -> np.ndarray:
    points = np.array([0, *HORIZONS], dtype=float)
    values = np.array([current_aqi, *[predictions[f"aqi_h{hour:03d}"][row] for hour in HORIZONS]])
    return np.interp(np.arange(1, 25, dtype=float), points, values)


def evaluate_year(
    year: int, continuous: pd.DataFrame, events: pd.DataFrame, feature_names: list[str]
) -> dict[str, object]:
    event_train, event_test = event_isolated_year_split(events, year, event_column="event_key")
    aft = AFTSurvivalRegressor(seed=42, nthread=1).fit(
        event_train[feature_names], event_train["remaining_hours"]
    )
    aft_prediction = np.asarray(aft.predict(event_test[feature_names]), dtype=float)
    test_features = continuous.set_index("timestamp").reindex(event_test["timestamp"]).reset_index()
    available = test_features["aqi"].notna().to_numpy()
    test_features = test_features.loc[available].reset_index(drop=True)
    event_test = event_test.loc[available].reset_index(drop=True)
    aft_prediction = aft_prediction[available]
    cutoff = pd.Timestamp(f"{year}-01-01", tz=continuous["timestamp"].dt.tz)
    curve_train = continuous[continuous["timestamp"].lt(cutoff)]
    curve = DirectCurveRegressor().fit(curve_train, CURVE_FEATURES, training_cutoff=cutoff)
    direct = curve.predict(test_features)
    curve_hours = np.array(
        [_curve_hours(direct, index, float(row.aqi)) for index, row in test_features.iterrows()]
    )
    derived = np.array(
        [curve_recovery_or_fallback(path, fallback) for path, fallback in zip(curve_hours, aft_prediction, strict=True)]
    )
    variants = {"aft": aft_prediction, "curve": derived}
    for weight in (0.25, 0.5, 0.75):
        variants[f"curve_{int(weight * 100)}"] = (weight * derived + (1 - weight) * aft_prediction)
    actual = event_test["remaining_hours"].to_numpy(dtype=float)
    target_mae: dict[str, float] = {}
    for name, values in direct.items():
        pollutant, hour = name.rsplit("_h", 1)
        target = f"{pollutant}_target_h{hour}"
        observed = test_features[target].notna().to_numpy()
        target_mae[name] = float(
            np.abs(values[observed] - test_features.loc[observed, target].to_numpy(float)).mean()
        )
    recovery_mae = {
        name: float(np.abs(prediction - actual).mean()) for name, prediction in variants.items()
    }
    return {
        "event_rows": len(event_test),
        "curve_train_rows": len(curve_train),
        "target_mae": target_mae,
        "recovery_mae": recovery_mae,
        "curve_recovery_coverage": float(np.mean(derived != aft_prediction)),
    }


def main() -> None:
    metadata = json.loads((ROOT / "artifacts" / "cnemc_2025_deployment_candidate" / "model_metadata.json").read_text(encoding="utf-8"))
    feature_names = list(metadata["feature_names"])
    continuous, events = load_continuous(), load_event_rows()
    results = {str(year): evaluate_year(year, continuous, events, feature_names) for year in (2024, 2025)}
    selection = results["2024"]["recovery_mae"]
    confirmation = results["2025"]["recovery_mae"]
    selected = select_hybrid(selection, confirmation)
    report = {
        "protocol": "direct 1/3/6/12/24-hour pollutant forecasts; 2024 selects, 2025 confirms; each training target strictly before cutoff; recovery is the start of 6 consecutive AQI <= 100 hours",
        "continuous_rows": len(continuous),
        "years": results,
        "selected_from_2024": min(selection, key=selection.__getitem__),
        "decision": {"status": "accept" if selected else "reject", "selected_variant": selected, "deployment_artifact_changed": False},
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
