"""Attach archived Open-Meteo PBL values to the audited 2014 CNEMC weather table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from haze_forecast.features import training_rows
from haze_forecast.labels import label_city_data, label_summary
from haze_forecast.open_meteo_archive import add_boundary_layer_height, parse_boundary_layer_height


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for an immutable raw-source audit trail."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weather", type=Path, default=Path("data/interim/cnemc_chengdu_2014_weather.parquet")
    )
    parser.add_argument(
        "--pbl-raw",
        type=Path,
        default=Path("data/raw/open_meteo_archive/chengdu_2014_boundary_layer_height.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/cnemc_chengdu_2014_weather_pbl.parquet"),
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=Path("data/processed/cnemc_chengdu_2014_pbl_training_features.parquet"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/cnemc_2014_open_meteo_pbl/report.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weather = pd.read_parquet(args.weather)
    raw_payload = json.loads(args.pbl_raw.read_text(encoding="utf-8"))
    pbl = parse_boundary_layer_height(raw_payload)
    enriched = add_boundary_layer_height(weather, pbl)
    labeled = label_city_data(enriched)
    training = training_rows(labeled)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.training_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    training.to_parquet(args.training_output, index=False)
    payload = {
        "source": "Open-Meteo historical archive API boundary_layer_height joined to CNEMC 2014 and NOAA ISD weather",
        "source_class": "historical archive/reanalysis input; not an operational weather forecast",
        "weather_input": str(args.weather),
        "pbl_raw_input": str(args.pbl_raw),
        "pbl_raw_sha256": sha256(args.pbl_raw),
        "rows": len(enriched),
        "boundary_layer_height_nonnull_rate": float(enriched["boundary_layer_height_m"].notna().mean()),
        "label_summary": label_summary(labeled),
        "training_rows": len(training),
        "output": str(args.output),
        "training_output": str(args.training_output),
        "deployment_policy": "candidate-only; do not replace the default model without external-test gate",
    }
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
