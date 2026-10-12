"""Join NOAA weather to the downloaded 2024-2025 Chengdu CNEMC observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from haze_forecast.cnemc_weather import add_noaa_weather
from haze_forecast.features import training_rows
from haze_forecast.labels import label_city_data, label_summary
from haze_forecast.noaa_isd import annual_csv_url, download_annual_csv, prepare_noaa_hourly

ROOT = Path(__file__).resolve().parents[1]
USAF = "562940"
WBAN = "99999"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pollution", type=Path, default=ROOT / "data/interim/cnemc_chengdu_2024_2025_hourly.parquet"
    )
    parser.add_argument("--raw-weather-dir", type=Path, default=ROOT / "data/raw/noaa_isd")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/interim/cnemc_chengdu_2024_2025_weather.parquet"
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=ROOT / "data/processed/cnemc_chengdu_2024_2025_training_features.parquet",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "artifacts/cnemc_2024_2025_weather/report.json"
    )
    args = parser.parse_args()

    pollution = pd.read_parquet(args.pollution)
    weather_frames: list[pd.DataFrame] = []
    weather_sources: dict[str, dict[str, str]] = {}
    for year in (2024, 2025):
        raw_path = args.raw_weather_dir / str(year) / f"{USAF}{WBAN}.csv"
        source_url = annual_csv_url(USAF, WBAN, year)
        download_annual_csv(source_url, raw_path)
        weather_frames.append(prepare_noaa_hourly(pd.read_csv(raw_path)))
        weather_sources[str(year)] = {"url": source_url, "sha256": sha256(raw_path)}
    weather = pd.concat(weather_frames, ignore_index=True)
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
    report = {
        "source": "CNEMC station observations + NOAA ISD station 562940-99999 (Shuangliu)",
        "pollution_input": str(args.pollution),
        "weather_sources": weather_sources,
        "rows": len(enriched),
        "weather_nonnull_rate": {
            column: float(enriched[column].notna().mean()) for column in weather_columns
        },
        "label_summary": label_summary(labeled),
        "boundary_layer_height_policy": "not fabricated; retain NaN with availability indicators",
        "output": str(args.output),
        "training_output": str(args.training_output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
