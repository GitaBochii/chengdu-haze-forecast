"""On-demand CAMS estimates; deliberately separate from station observations."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import requests

from haze_forecast.features import MEASUREMENT_COLUMNS

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
STANDARD_URL = "https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/jcffbz/201203/W020250407406362045613.pdf"
AIR_FIELDS = ("pm2_5", "pm10", "sulphur_dioxide", "nitrogen_dioxide", "carbon_monoxide", "ozone")
WEATHER_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "wind_speed_10m": "m/s",
    "wind_direction_10m": "°",
}
BREAKPOINTS = {
    "pm2_5": (0, 35, 75, 115, 150, 250, 350, 500),
    "pm10": (0, 50, 150, 250, 350, 420, 500, 600),
    "so2": (0, 150, 500, 650, 800),
    "so2_24": (0, 50, 150, 475, 800, 1600, 2100, 2620),
    "no2": (0, 100, 200, 700, 1200, 2340, 3090, 3840),
    "co": (0, 5, 10, 35, 60, 90, 120, 150),
    "o3_1": (0, 160, 200, 300, 400, 800, 1000, 1200),
    "o3_8": (0, 100, 160, 215, 265, 800),
}
INDEX = (0, 50, 100, 150, 200, 300, 400, 500)
LOCAL = timezone(timedelta(hours=8))
NOTICE = (
    "空气质量来自 Open-Meteo / CAMS 全球模型估算，并非成都监测站实测；气象亦为模型数据。"
    "CAMS 全球原生时间分辨率为 3 小时，接口提供小时序列。AQI 为按 HJ 633—2012 "
    "训练时期口径推算的研究代理，不是现行官方 AQI。该输入未经独立精度验证，预测仅供实验参考。"
    "边界层高度缺失，不编造或插值补齐。"
)


class OnlineEstimateError(ValueError):
    """Sanitized, reader-facing online import failure."""


def _now():
    return datetime.now(UTC)


def _iaqi(value, points):
    if not math.isfinite(value) or value < 0 or value > points[-1]:
        raise OnlineEstimateError("污染物浓度超出训练口径可计算范围，已停止导入")
    for i in range(1, len(points)):
        if value <= points[i]:
            return math.ceil(
                INDEX[i - 1]
                + (value - points[i - 1]) * (INDEX[i] - INDEX[i - 1]) / (points[i] - points[i - 1])
            )
    raise OnlineEstimateError("无法计算训练口径 AQI")


def _request(url, fields, receipt, name):
    params = {
        "latitude": 30.67,
        "longitude": 104.06,
        "hourly": ",".join(fields),
        "past_days": 3,
        "forecast_days": 1,
        "timezone": "GMT",
    }
    if name == "weather":
        params["wind_speed_unit"] = "ms"
    response = None
    try:
        response = requests.get(url, params=params, timeout=(3, 8), stream=True)
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > 2 * 1024 * 1024:
                raise OnlineEstimateError("在线数据响应过大，已停止导入")
            chunks.append(chunk)
        raw = b"".join(chunks)
        received = _now().isoformat()
        (receipt / f"{name}.json").write_bytes(raw)
        metadata = {
            "received_at": received,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "endpoint": url,
            "parameters": params,
        }
        (receipt / f"{name}-receipt.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        body = json.loads(raw)
        if not isinstance(body, dict) or body.get("error"):
            raise OnlineEstimateError("在线数据源未返回有效序列，请稍后重试")
        return body, metadata
    except (requests.RequestException, ValueError, UnicodeError) as error:
        if isinstance(error, OnlineEstimateError):
            raise
        raise OnlineEstimateError("在线数据源连接或响应异常，请稍后重试") from None
    finally:
        if response is not None:
            response.close()


def _parse(body, expected):
    if body.get("utc_offset_seconds") != 0:
        raise OnlineEstimateError("在线数据时区不符，已停止导入")
    units, hourly = body.get("hourly_units", {}), body.get("hourly", {})
    if not isinstance(units, dict) or not isinstance(hourly, dict):
        raise OnlineEstimateError("在线数据结构不完整")
    if units.get("time") != "iso8601" or any(units.get(k) != v for k, v in expected.items()):
        raise OnlineEstimateError("在线数据单位不符，已停止导入")
    times = hourly.get("time")
    if not isinstance(times, list) or any(
        not isinstance(hourly.get(k), list) or len(hourly[k]) != len(times) for k in expected
    ):
        raise OnlineEstimateError("在线小时序列字段不完整")
    result = {}
    for i, stamp in enumerate(times):
        try:
            dt = datetime.fromisoformat(stamp)
            if dt.tzinfo is not None and dt.utcoffset() != timedelta(0):
                raise ValueError()
            dt = dt.replace(tzinfo=UTC)
            if dt.minute or dt.second or dt.microsecond or dt in result:
                raise ValueError()
        except (TypeError, ValueError):
            raise OnlineEstimateError("在线数据时间戳无效或重复") from None
        result[dt] = {key: hourly[key][i] for key in expected}
    return result


def _number(row, field):
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise OnlineEstimateError("所需历史小时存在缺失或无效值，未进行插值，请稍后重试")
    if field != "temperature_2m" and value < 0:
        raise OnlineEstimateError("在线数据含无效负值，已停止导入")
    if (
        field == "relative_humidity_2m"
        and value > 100
        or field == "wind_direction_10m"
        and value > 360
    ):
        raise OnlineEstimateError("在线气象数据超出有效范围")
    return float(value)


def fetch_online_estimate(root: Path) -> dict:
    """Return a 24-hour preview and 25 model rows, including one warmup hour."""
    now = _now()
    end = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start = end - timedelta(hours=23)
    model_start = start - timedelta(hours=1)
    receipt = Path(root) / "artifacts/online_estimates" / f"{now:%Y%m%dT%H%M%S}-{uuid4().hex[:12]}"
    try:
        receipt.mkdir(parents=True, exist_ok=False)
        air_body, air_meta = _request(AIR_URL, AIR_FIELDS, receipt, "air")
        weather_body, weather_meta = _request(WEATHER_URL, WEATHER_UNITS, receipt, "weather")
        air = _parse(air_body, {field: "μg/m³" for field in AIR_FIELDS})
        weather = _parse(weather_body, WEATHER_UNITS)
        # A complete extra 23-hour prefix is required for every trailing PM/SO2 mean.
        needed = [model_start - timedelta(hours=23) + timedelta(hours=i) for i in range(48)]
        for dt in needed:
            if dt not in air:
                raise OnlineEstimateError("空气质量序列存在断档，无法组成连续历史窗口")
            air[dt] = {k: _number(air[dt], k) for k in AIR_FIELDS}
        rows = []
        for i in range(25):
            dt = model_start + timedelta(hours=i)
            if dt not in weather:
                raise OnlineEstimateError("气象序列存在断档，无法组成含预热小时的连续 25 小时")
            met = {k: _number(weather[dt], k) for k in WEATHER_UNITS}
            point = air[dt]

            def mean(field, hours, dt=dt):
                return sum(air[dt - timedelta(hours=h)][field] for h in range(hours)) / hours

            so2 = (
                _iaqi(point["sulphur_dioxide"], BREAKPOINTS["so2"])
                if point["sulphur_dioxide"] <= 800
                else _iaqi(mean("sulphur_dioxide", 24), BREAKPOINTS["so2_24"])
            )
            ozone8 = mean("ozone", 8)
            indices = [
                so2,
                _iaqi(mean("pm2_5", 24), BREAKPOINTS["pm2_5"]),
                _iaqi(mean("pm10", 24), BREAKPOINTS["pm10"]),
                _iaqi(point["nitrogen_dioxide"], BREAKPOINTS["no2"]),
                _iaqi(point["carbon_monoxide"] / 1000, BREAKPOINTS["co"]),
                _iaqi(point["ozone"], BREAKPOINTS["o3_1"]),
            ]
            if ozone8 <= 800:
                indices.append(_iaqi(ozone8, BREAKPOINTS["o3_8"]))
            radians = math.radians(met["wind_direction_10m"])
            rows.append(
                {
                    "timestamp": dt.astimezone(LOCAL).isoformat(),
                    "aqi": max(indices),
                    "pm2_5": point["pm2_5"],
                    "pm10": point["pm10"],
                    "so2": point["sulphur_dioxide"],
                    "no2": point["nitrogen_dioxide"],
                    "co": point["carbon_monoxide"] / 1000,
                    "o3": ozone8,
                    "temperature_c": met["temperature_2m"],
                    "relative_humidity_pct": met["relative_humidity_2m"],
                    "wind_u_ms": -met["wind_speed_10m"] * math.sin(radians),
                    "wind_v_ms": -met["wind_speed_10m"] * math.cos(radians),
                    "boundary_layer_height_m": None,
                }
            )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["timestamp", *MEASUREMENT_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
        return {
            "csv": stream.getvalue(),
            "rows": rows[-24:],
            "start_at": rows[-24]["timestamp"],
            "model_input_start_at": rows[0]["timestamp"],
            "model_input_hours": len(rows),
            "end_at": rows[-1]["timestamp"],
            "fetched_at": _now().isoformat(),
            "notice": NOTICE,
            "source_urls": [
                "https://open-meteo.com/en/docs/air-quality-api",
                "https://open-meteo.com/en/docs",
                STANDARD_URL,
            ],
            "missing_fields": ["boundary_layer_height_m"],
            "aqi_method": "HJ 633—2012 训练口径推算 AQI（非现行官方 AQI）",
            "provenance": {"air": air_meta, "weather": weather_meta, "receipt_id": receipt.name},
        }
    except (OnlineEstimateError, OSError) as error:
        message = (
            str(error)
            if isinstance(error, OnlineEstimateError)
            else "无法保存在线数据来源记录，已停止导入"
        )
        try:
            (receipt / "failure.json").write_text(
                json.dumps({"at": _now().isoformat(), "error": message}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
        raise OnlineEstimateError(message) from None
