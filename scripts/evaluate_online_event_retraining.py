"""Simulate deployment-time retraining after each completed pollution event.

At an event's first predicted hour, only events that had completely dissipated
earlier are in the training pool. This evaluates whether an observed regime
change can be absorbed for later events without leaking a target from the
active event.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "artifacts" / "online_event_retraining" / "report.json"


def prepare(filename: str, source: str) -> pd.DataFrame:
    frame = pd.read_parquet(PROCESSED / filename).copy()
    frame["event_key"] = source + ":" + frame["event_id"].astype(str)
    sources = [name for name in frame.columns if name.startswith("boundary_layer_height_m")]
    frame = add_missingness_indicators(frame, sources)
    frame["boundary_layer_height_m_change_3_is_missing"] = (
        frame["boundary_layer_height_m"].isna() | frame["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return frame


def summarize(actual: np.ndarray, prediction: np.ndarray, event_key: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - actual)
    event_mae = pd.DataFrame({"event_key": event_key, "error": error}).groupby("event_key")["error"].mean()
    return {"row_mae": float(error.mean()), "event_equal_mae": float(event_mae.mean())}


def main() -> None:
    all_rows = pd.concat(
        [
            prepare("cnemc_chengdu_2014_training_features.parquet", "prefix_2014"),
            prepare("training_features.parquet", "base"),
            prepare("cnemc_chengdu_2022_training_features.parquet", "external_2022"),
            prepare("cnemc_chengdu_2023_training_features.parquet", "external_2023"),
        ],
        ignore_index=True,
    )
    metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"))
    features = list(metadata["feature_names"])
    result: dict[str, object] = {
        "protocol": "event-isolated annual forward deployment simulation; retrain after each prior event closes",
        "years": {},
    }
    for year in (2022, 2023):
        history, test = event_isolated_year_split(all_rows, year, event_column="event_key")
        static_model = AFTSurvivalRegressor(seed=42, nthread=1).fit(
            history[features], history["remaining_hours"]
        )
        static_prediction = np.asarray(static_model.predict(test[features]), dtype=float)
        state = history.copy()
        online_predictions: list[pd.DataFrame] = []
        for _, event in test.groupby("event_key", sort=False):
            model = AFTSurvivalRegressor(seed=42, nthread=1).fit(
                state[features], state["remaining_hours"]
            )
            prediction = np.asarray(model.predict(event[features]), dtype=float)
            online_predictions.append(
                event.loc[:, ["event_key", "remaining_hours"]].assign(prediction=prediction)
            )
            state = pd.concat([state, event], ignore_index=True)
        online = pd.concat(online_predictions, ignore_index=True)
        result["years"][str(year)] = {
            "prior_completed_event_rows": len(history),
            "test_rows": len(test),
            "test_events": int(test["event_key"].nunique()),
            "static": summarize(
                test["remaining_hours"].to_numpy(float), static_prediction, test["event_key"].to_numpy()
            ),
            "online_retrained": summarize(
                online["remaining_hours"].to_numpy(float),
                online["prediction"].to_numpy(float),
                online["event_key"].to_numpy(),
            ),
        }
        static_mae = result["years"][str(year)]["static"]["row_mae"]
        online_mae = result["years"][str(year)]["online_retrained"]["row_mae"]
        print(f"{year}: {static_mae:.3f} -> {online_mae:.3f} ({1 - online_mae / static_mae:+.2%})")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {OUTPUT}")


if __name__ == "__main__":
    main()
