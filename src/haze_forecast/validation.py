"""原始小时数据的标准化和质量审计。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from haze_forecast.constants import LOCAL_TIMEZONE, REQUIRED_COLUMNS

SOURCE_COLUMN_MAP = {
    "AQI": "aqi",
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "O3-8h": "o3",
    "CO": "co",
    "U10": "wind_u_ms",
    "V10": "wind_v_ms",
    "T2": "temperature_c",
    "RH2": "relative_humidity_pct",
    "PBLH": "boundary_layer_height_m",
}
NUMERIC_COLUMNS = sorted(REQUIRED_COLUMNS - {"timestamp", "station_id"})


def validate_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields and normalize timestamps to Chengdu local time."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp contains invalid values")
    if timestamps.dt.tz is None:
        result["timestamp"] = timestamps.dt.tz_localize(LOCAL_TIMEZONE)
    else:
        result["timestamp"] = timestamps.dt.tz_convert(LOCAL_TIMEZONE)

    result[NUMERIC_COLUMNS] = result[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if result.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("duplicate station_id/timestamp")
    return result.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def normalize_source_frame(source: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Map one source workbook to the project schema and normalize temperature units."""
    required_source = {"year", "month", "day", "hour", *SOURCE_COLUMN_MAP}
    missing = required_source - set(source.columns)
    if missing:
        raise ValueError(f"{station_id} missing source columns: {sorted(missing)}")

    normalized = source.rename(columns=SOURCE_COLUMN_MAP).copy()
    normalized["station_id"] = station_id
    normalized["timestamp"] = pd.to_datetime(
        normalized[["year", "month", "day", "hour"]], errors="coerce"
    )
    temperature = pd.to_numeric(normalized["temperature_c"], errors="coerce")
    if temperature.dropna().median() > 200:
        normalized["temperature_c"] = temperature - 273.15
    return normalized[[*REQUIRED_COLUMNS]]


def load_chengdu_station_files(source_dir: Path) -> pd.DataFrame:
    """Read the nine source workbooks into the project data contract."""
    workbook_paths = sorted(source_dir.glob("*.xlsx"))
    if not workbook_paths:
        raise FileNotFoundError(f"no Chengdu station workbook found in {source_dir}")

    frames: list[pd.DataFrame] = []
    for workbook in workbook_paths:
        source = pd.read_excel(workbook)
        frames.append(normalize_source_frame(source, workbook.stem))
    return validate_hourly_frame(pd.concat(frames, ignore_index=True))


def city_hourly_median(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate station observations to a robust city-level hourly series."""
    numeric = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    return frame.groupby("timestamp", as_index=False)[numeric].median(numeric_only=True)


def quality_report(frame: pd.DataFrame) -> dict[str, object]:
    """Return auditable coverage and missingness facts without imputing observations."""
    return {
        "rows": len(frame),
        "stations": int(frame["station_id"].nunique()),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "missing_by_column": {name: int(frame[name].isna().sum()) for name in frame.columns},
    }


def prepare_interim_data(raw_dir: Path, interim_dir: Path) -> dict[str, Path]:
    """Create validated station and city datasets plus their quality report."""
    station_data = load_chengdu_station_files(raw_dir / "data" / "chengdu")
    city_data = city_hourly_median(station_data)
    interim_dir.mkdir(parents=True, exist_ok=True)
    station_path = interim_dir / "station_hourly.parquet"
    city_path = interim_dir / "city_hourly.parquet"
    report_path = interim_dir / "quality_report.json"
    station_data.to_parquet(station_path, index=False)
    city_data.to_parquet(city_path, index=False)
    report = {
        "source_schema": {source: target for source, target in SOURCE_COLUMN_MAP.items()},
        "unavailable_in_source": ["pressure_hpa", "precipitation_mm"],
        "station_data": quality_report(station_data),
        "city_data": {
            "rows": len(city_data),
            "start": city_data["timestamp"].min().isoformat(),
            "end": city_data["timestamp"].max().isoformat(),
            "missing_by_column": {
                name: int(city_data[name].isna().sum()) for name in city_data.columns
            },
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"station_data": station_path, "city_data": city_path, "quality_report": report_path}
