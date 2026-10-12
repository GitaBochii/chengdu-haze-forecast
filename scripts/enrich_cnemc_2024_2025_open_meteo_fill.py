"""Build a separate 2024-2025 reanalysis-filled weather candidate.

Local NOAA ISD observations remain the first choice.  Open-Meteo archive values
fill only missing temperature, humidity, wind and PBL fields.  The output is
kept separate because archive/reanalysis availability is not an operational
future-weather source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from haze_forecast.features import training_rows
from haze_forecast.labels import label_city_data, label_summary
from haze_forecast.open_meteo_archive import (
    WEATHER_COLUMNS,
    fill_weather_gaps,
    parse_hourly_weather,
)

ROOT = Path(__file__).resolve().parents[1]
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = 30.5723
LONGITUDE = 104.0665


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weather", type=Path, default=ROOT / "data/interim/cnemc_chengdu_2024_2025_weather.parquet"
    )
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "data/raw/open_meteo_archive/chengdu_2024_2025_hourly.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/interim/cnemc_chengdu_2024_2025_weather_open_meteo_fill.parquet",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=ROOT / "data/processed/cnemc_chengdu_2024_2025_open_meteo_fill_training_features.parquet",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "artifacts/cnemc_2024_2025_open_meteo_fill/report.json"
    )
    return parser.parse_args()


def load_archive_weather(raw_path: Path) -> tuple[pd.DataFrame, str]:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height",
        "timezone": "Asia/Shanghai",
    }
    if not raw_path.exists():
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=120)
        response.raise_for_status()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw_path.with_suffix(f"{raw_path.suffix}.part")
        try:
            temporary.write_bytes(response.content)
            temporary.replace(raw_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    return parse_hourly_weather(payload), response_url(params)


def response_url(params: dict[str, object]) -> str:
    request = requests.Request("GET", OPEN_METEO_ARCHIVE_URL, params=params).prepare()
    return str(request.url)


def main() -> None:
    args = parse_args()
    observed = pd.read_parquet(args.weather)
    archive, source_url = load_archive_weather(args.raw)
    enriched = fill_weather_gaps(observed, archive)
    labeled = label_city_data(enriched)
    training = training_rows(labeled)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    training.to_parquet(args.training_output, index=False)
    report = {
        "source": "CNEMC observations + NOAA ISD first-choice weather + Open-Meteo archive gap fill",
        "source_class": "historical archive/reanalysis candidate; not a future operational forecast input",
        "weather_input": str(args.weather),
        "open_meteo_url": source_url,
        "open_meteo_raw": str(args.raw),
        "open_meteo_raw_sha256": sha256(args.raw),
        "rows": len(enriched),
        "archive_rows": len(archive),
        "nonnull_rate_before": {column: float(observed[column].notna().mean()) for column in WEATHER_COLUMNS},
        "nonnull_rate_after": {column: float(enriched[column].notna().mean()) for column in WEATHER_COLUMNS},
        "label_summary": label_summary(labeled),
        "training_rows": len(training),
        "output": str(args.output),
        "training_output": str(args.training_output),
        "deployment_policy": "candidate-only; require no-leak external validation before changing the default model",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
