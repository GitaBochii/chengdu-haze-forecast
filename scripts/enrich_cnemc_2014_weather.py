"""Attach audited NOAA weather to 2014 CNEMC pollution without inventing PBL data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from haze_forecast.cnemc_weather import add_noaa_weather
from haze_forecast.features import training_rows
from haze_forecast.labels import label_city_data, label_summary
from haze_forecast.noaa_isd import prepare_noaa_hourly


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pollution", type=Path, default=Path("data/interim/cnemc_chengdu_2014_hourly.parquet")
    )
    parser.add_argument(
        "--noaa-raw", type=Path, default=Path("data/raw/noaa_isd/2014/56294099999.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/interim/cnemc_chengdu_2014_weather.parquet")
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=Path("data/processed/cnemc_chengdu_2014_training_features.parquet"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/cnemc_2014_weather/report.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pollution = pd.read_parquet(args.pollution)
    weather = prepare_noaa_hourly(pd.read_csv(args.noaa_raw))
    enriched = add_noaa_weather(pollution, weather)
    labeled = label_city_data(enriched)
    training = training_rows(labeled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    training.to_parquet(args.training_output, index=False)
    weather_columns = [
        "temperature_c",
        "relative_humidity_pct",
        "wind_u_ms",
        "wind_v_ms",
        "boundary_layer_height_m",
    ]
    payload = {
        "source": "NOAA ISD station 562940-99999 (Shuangliu) joined to CNEMC 2014 pollution",
        "pollution_input": str(args.pollution),
        "noaa_raw_input": str(args.noaa_raw),
        "noaa_raw_sha256": sha256(args.noaa_raw),
        "rows": len(enriched),
        "weather_nonnull_rate": {
            column: float(enriched[column].notna().mean()) for column in weather_columns
        },
        "label_summary": label_summary(labeled),
        "training_rows": len(training),
        "boundary_layer_height_policy": "missing; retain NaN and prohibit default-model replacement",
        "output": str(args.output),
        "training_output": str(args.training_output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
