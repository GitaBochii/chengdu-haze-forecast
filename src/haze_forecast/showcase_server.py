"""Loopback-only website and inference adapter for the existing research models."""

from __future__ import annotations

import argparse
import io
import json
import math
import mimetypes
import re
import threading
import time
import uuid
from functools import cached_property
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import joblib
import pandas as pd

from haze_forecast.app import build_history_prediction_input, manual_feature_frame
from haze_forecast.features import MEASUREMENT_COLUMNS
from haze_forecast.multimodal_research import MultimodalInputError

ROOT = Path(__file__).resolve().parents[2]
MAX_BODY = 1024 * 1024
STATIC_FILES = frozenset(
    {
        "index.html",
        "styles.css",
        "script.js",
        "effects.js",
        "effects.css",
        "workbench.html",
        "workbench.css",
        "workbench.js",
        "multimodal.js",
        "weather-inputs.js",
        "weather-inputs.json",
        "rebound.js",
        "rebound-analysis.js",
        "rebound.css",
        "rebound-cases.json",
        "assets/chengdu-overcast.png",
        "research-notes.md",
    }
)
FIELD_SPECS = (
    ("aqi", "AQI", 130, 0, 1000, 1),
    ("pm2_5", "PM2.5 (µg/m³)", 90, 0, 2000, 1),
    ("pm10", "PM10 (µg/m³)", 120, 0, 3000, 1),
    ("so2", "SO2 (µg/m³)", 12, 0, 3000, 1),
    ("no2", "NO2 (µg/m³)", 35, 0, 3000, 1),
    ("co", "CO (mg/m³)", 1, 0, 100, 0.1),
    ("o3", "O3-8h (µg/m³)", 50, 0, 2000, 1),
    ("temperature_c", "气温 (°C)", 12, -60, 60, 0.1),
    ("relative_humidity_pct", "相对湿度 (%)", 70, 0, 100, 1),
    ("wind_u_ms", "东西向风速 (m/s)", 1, -100, 100, 0.1),
    ("wind_v_ms", "南北向风速 (m/s)", 0, -100, 100, 0.1),
    ("boundary_layer_height_m", "边界层高度 (m)", 300, 0, 10000, 1),
)


class InputError(ValueError):
    """Validated reader-facing error; safe to expose across the HTTP boundary."""


def timestamp(value):
    if not isinstance(value, str) or not value or len(value) > 64:
        raise InputError("请输入有效的观测时间")
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            raise ValueError()
        result = (
            result.tz_localize("Asia/Shanghai")
            if result.tzinfo is None
            else result.tz_convert("Asia/Shanghai")
        )
        if not 1900 <= result.year <= 2100 or result != result.floor("h"):
            raise ValueError()
        return result
    except (ValueError, TypeError, OverflowError) as error:
        raise InputError("观测时间须为 1900—2100 年间的整点时间") from error


def trend_records(frame):
    records = []
    for _, row in frame.iterrows():
        entry = {"timestamp": pd.Timestamp(row["timestamp"]).isoformat()}
        for column in ("aqi", "pm2_5", "pm10"):
            value = row[column]
            entry[column] = (
                float(value) if pd.notna(value) and math.isfinite(float(value)) else None
            )
        records.append(entry)
    return records


class ShowcaseService:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.online_lock = threading.RLock()
        self.online_snapshot = None

    @cached_property
    def multimodal(self):
        from haze_forecast.multimodal_research import MultimodalResearch
        return MultimodalResearch(self.root)

    def import_online(self):
        from haze_forecast.online_estimates import OnlineEstimateError, fetch_online_estimate

        with self.online_lock:
            if self.online_snapshot and time.monotonic() - self.online_snapshot[0] < 300:
                data = self.online_snapshot[1]
            else:
                self.online_snapshot = None
                try:
                    data = fetch_online_estimate(self.root)
                except OnlineEstimateError as error:
                    raise InputError(str(error)) from error
                data = {**data, "token": uuid.uuid4().hex}
                data["can_predict"] = bool(data.get("csv"))
                self.online_snapshot = (time.monotonic(), data)
            return {key: value for key, value in data.items() if key != "csv"}

    def predict_online(self, payload):
        with self.online_lock:
            if payload.get("acknowledge_estimates") is not True:
                raise InputError("请先确认在线序列是模型估算，非监测站实测")
            if not self.online_snapshot:
                raise InputError("请先获取并预览在线序列")
            created, snapshot = self.online_snapshot
            if payload.get("token") != snapshot["token"] or time.monotonic() - created > 900:
                raise InputError("在线序列已过期，请重新获取")
            if not snapshot.get("can_predict"):
                raise InputError("在线序列不完整，暂不能预测")
            output = self.predict({"mode": "sequence", "csv": snapshot["csv"]})
            output["trend"] = output["trend"][-24:]
            output.update(
                mode="online",
                model_label="XGBoost AFT · 在线估算实验（非实测）",
                notice=(
                    (
                        "当前估算序列不满足颗粒物污染预测条件，不给出消散时长。"
                        if not output["eligible"]
                        else ""
                    )
                    + snapshot["notice"]
                    + " 此输入组合未经独立精度验证，不用于公共预警。"
                ),
                provenance={
                    key: snapshot.get(key)
                    for key in (
                        "start_at",
                        "model_input_start_at",
                        "model_input_hours",
                        "end_at",
                        "fetched_at",
                        "source_urls",
                        "missing_fields",
                        "aqi_method",
                        "provenance",
                    )
                },
            )
            return output

    @cached_property
    def models(self):
        directory = self.root / "artifacts/cnemc_2025_deployment_candidate"
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        return (
            metadata,
            joblib.load(directory / "static_aft_model.joblib"),
            joblib.load(directory / "aft_model.joblib"),
        )

    @cached_property
    def backtest(self):
        data = pd.read_parquet(self.root / "artifacts/strict_history_backtest/predictions.parquet")
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert(
            "Asia/Shanghai"
        )
        return data.sort_values("timestamp").reset_index(drop=True)

    @cached_property
    def observations(self):
        data = pd.read_parquet(
            self.root / "data/processed/regional_air_models_full_history/city_hourly.parquet",
            columns=["timestamp", "aqi", "pm2_5", "pm10"],
        )
        dates = pd.to_datetime(data["timestamp"])
        data["timestamp"] = (
            dates.dt.tz_localize("Asia/Shanghai")
            if dates.dt.tz is None
            else dates.dt.tz_convert("Asia/Shanghai")
        )
        return data.sort_values("timestamp")

    def status(self):
        with self.lock:
            _ = self.models
            count = len(self.backtest)
            _ = self.observations
        return {"ready": True, "history_records": count, "notice": "本地研究模型；不提供实时监测。"}

    def schema(self):
        return {
            "fields": [
                dict(zip(("name", "label", "default", "min", "max", "step"), field))
                for field in FIELD_SPECS
            ],
            "columns": ["timestamp", *MEASUREMENT_COLUMNS],
        }

    def dates(self):
        return {
            "dates": sorted(
                self.backtest["timestamp"].dt.strftime("%Y-%m-%d").unique().tolist(), reverse=True
            )
        }

    def history(self, date):
        if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise InputError("请选择有效日期")
        data = self.backtest.loc[self.backtest["timestamp"].dt.strftime("%Y-%m-%d").eq(date)]
        return {
            "records": [
                {
                    "id": row["timestamp"].isoformat(),
                    "timestamp": row["timestamp"].isoformat(),
                    "aqi": float(row["aqi"]),
                }
                for _, row in data.iterrows()
            ]
        }

    def replay(self, value):
        time = timestamp(value)
        rows = self.backtest.loc[self.backtest["timestamp"].eq(time)]
        if rows.empty:
            raise InputError("该时刻没有严格历史回测记录")
        row = rows.iloc[0]
        prediction, actual = float(row["strict_prediction_hours"]), float(row["remaining_hours"])
        data = self.observations
        data = data.loc[data["timestamp"].between(time - pd.Timedelta(hours=47), time)]
        return {
            "timestamp": time.isoformat(),
            "prediction_hours": prediction,
            "actual_hours": actual,
            "error_hours": abs(prediction - actual),
            "eligible": True,
            "model_label": "严格历史回测（既有结果）",
            "notice": "使用未纳入该污染过程及之后事件训练的历史预测；气象数据的实时可得性未得到完整证明。不是当前重拟合模型的独立精度。",
            "trend": trend_records(data),
        }

    def predict(self, payload):
        if isinstance(payload, dict) and payload.get("mode") == "online":
            return self.predict_online(payload)
        if not isinstance(payload, dict) or payload.get("mode") not in ("manual", "sequence"):
            raise InputError("请选择手动输入或序列输入")
        mode = payload["mode"]
        if mode == "manual":
            time = timestamp(payload.get("timestamp"))
            values = payload.get("values")
            if not isinstance(values, dict) or set(values) != set(MEASUREMENT_COLUMNS):
                raise InputError("请完整填写所有观测字段")
            for name, label, _, low, high, _ in FIELD_SPECS:
                value = values[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not low <= value <= high
                ):
                    raise InputError(f"{label} 须为 {low} 至 {high} 范围内的有效数值")
            eligible = values["aqi"] > 100 and (values["pm2_5"] >= 75 or values["pm10"] >= 150)
            history = pd.DataFrame([{**values, "timestamp": time}])
            notice = "单时刻输入采用简化的颗粒物判断，不能替代 24 小时观测。此模型没有可用于该输入方式的独立精度或可信区间。"
        else:
            csv = payload.get("csv")
            if not isinstance(csv, str) or not csv.strip() or len(csv.encode("utf-8")) > MAX_BODY:
                raise InputError("请上传不超过 1 MB 的 CSV")
            try:
                history = pd.read_csv(io.StringIO(csv))
            except (ValueError, pd.errors.ParserError) as error:
                raise InputError("无法读取 CSV，请检查表头和数据格式") from error
            if not 24 <= len(history) <= 10000 or set(history.columns) != {
                "timestamp",
                *MEASUREMENT_COLUMNS,
            }:
                raise InputError("CSV 须包含 timestamp 和模板的 12 个测量字段，且有 24 至 10000 行")
            history["timestamp"] = history["timestamp"].map(timestamp)
            history = history.sort_values("timestamp").reset_index(drop=True)
            if (
                history["timestamp"].duplicated().any()
                or not history["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
            ):
                raise InputError("观测时间须连续且不重复，每小时一条")
            for name, label, _, low, high, _ in FIELD_SPECS:
                if pd.api.types.is_bool_dtype(history[name]):
                    raise InputError(f"{label} 包含无效数值")
                try:
                    history[name] = pd.to_numeric(history[name], errors="raise")
                except (TypeError, ValueError) as error:
                    raise InputError(f"{label} 包含无效数值") from error
                present = history[name].dropna()
                if not present.between(low, high).all():
                    raise InputError(f"{label} 包含超出范围的数值")
            if history[["aqi", "pm2_5", "pm10"]].tail(24).isna().any().any():
                raise InputError("最近 24 小时的 AQI、PM2.5 和 PM10 不能缺失")
            time = history.iloc[-1]["timestamp"]
            notice = "依据所上传的小时观测计算；当前模型重拟合至 2025 年底，结果不是历史独立验证，不提供可信区间。缺失的气象数据可能影响结果。"
        with self.lock:
            metadata, snapshot_model, sequence_model = self.models
            if mode == "manual":
                frame = manual_feature_frame(metadata["snapshot_feature_names"], values, time)
                model = snapshot_model
            else:
                latest, frame = build_history_prediction_input(history, metadata["feature_names"])
                eligible = bool(latest["aqi"] > 100 and latest["particle_primary"])
                model = sequence_model
            prediction = float(model.predict(frame)[0]) if eligible else None
            if prediction is not None and not math.isfinite(prediction):
                raise RuntimeError("Model output is not finite")
            if prediction is not None:
                prediction = max(0.0, prediction)
        if not eligible:
            notice = "当前输入不满足颗粒物污染预测条件，不给出消散时长。" + notice
        return {
            "timestamp": time.isoformat(),
            "prediction_hours": prediction,
            "actual_hours": None,
            "error_hours": None,
            "eligible": eligible,
            "model_label": "XGBoost AFT · "
            + ("单时刻输入" if mode == "manual" else "小时序列输入"),
            "notice": notice,
            "trend": trend_records(history.tail(48)),
        }


def create_server(port=8510, service=None):
    service = service or ShowcaseService()

    class Handler(BaseHTTPRequestHandler):
        def send_data(self, code, content, content_type="application/json; charset=utf-8"):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def valid_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_GET(self):
            if not self.valid_host():
                self.send_data(403, {"error": "仅允许本机访问"})
                return
            try:
                url = urlsplit(self.path)
                query = parse_qs(url.query)
                if url.path == "/api/status":
                    result = service.status()
                elif url.path == "/api/schema":
                    result = service.schema()
                elif url.path == "/api/history/dates":
                    result = service.dates()
                elif url.path == "/api/history":
                    result = service.history(query.get("date", [None])[0])
                elif url.path == "/api/replay":
                    result = service.replay(query.get("timestamp", [None])[0])
                elif url.path == "/api/multimodal/status":
                    result = service.multimodal.status()
                elif url.path == "/api/multimodal/cases":
                    result = service.multimodal.cases()
                elif url.path == "/api/multimodal/replay":
                    seed = query.get("seed", ["24"])[0]
                    if seed not in {"24", "42", "2026"}:
                        raise InputError("请选择有效的实验种子")
                    result = service.multimodal.replay(
                        query.get("timestamp", [None])[0], int(seed),
                        query.get("arm", ["spatial_fusion"])[0])
                else:
                    name = unquote(url.path).lstrip("/") or "index.html"
                    if name not in STATIC_FILES:
                        self.send_data(404, {"error": "页面不存在"})
                        return
                    directory = (service.root / "project-showcase").resolve()
                    file = (directory / name).resolve()
                    if not file.is_relative_to(directory) or not file.is_file():
                        self.send_data(404, {"error": "页面不存在"})
                        return
                    # Windows registry associations can incorrectly label scripts text/plain.
                    content_type = {".js": "application/javascript", ".css": "text/css",
                                    ".html": "text/html"}.get(file.suffix)
                    content_type = content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
                    self.send_data(
                        200,
                        file.read_bytes(),
                        content_type
                        + ("; charset=utf-8" if content_type.startswith("text/") else ""),
                    )
                    return
                self.send_data(200, result)
            except (InputError, MultimodalInputError) as error:
                self.send_data(400, {"error": str(error)})
            except Exception:  # noqa: BLE001 - HTTP boundary must not disclose model internals.
                self.send_data(503, {"error": "本地模型或数据暂不可用，请检查项目文件是否完整"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (
                not self.valid_host()
                or (origin is not None and origin != f"http://{self.headers.get('Host')}")
                or self.headers.get("Sec-Fetch-Site") == "cross-site"
            ):
                self.send_data(403, {"error": "不接受跨站请求"})
                return
            if self.path not in {"/api/predict", "/api/online/import", "/api/multimodal/reinfer"}:
                self.send_data(404, {"error": "接口不存在"})
                return
            try:
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError as error:
                    raise InputError("提交内容长度无效") from error
                if (
                    not 0 < size <= MAX_BODY
                    or self.headers.get_content_type() != "application/json"
                ):
                    raise InputError("请提交不超过 1 MB 的 JSON 数据")
                self.connection.settimeout(15)
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise InputError("提交内容不完整")
                try:
                    payload = json.loads(raw)
                except (ValueError, UnicodeError) as error:
                    raise InputError("JSON 格式无效") from error
                if self.path == "/api/multimodal/reinfer":
                    if not isinstance(payload, dict):
                        raise InputError("请提交有效的样本选择")
                    result = service.multimodal.reinfer(
                        payload.get("timestamp"), payload.get("seed", 24),
                        payload.get("arm", "spatial_fusion"))
                else:
                    result = service.import_online() if self.path == "/api/online/import" else service.predict(payload)
                self.send_data(200, result)
            except (InputError, MultimodalInputError) as error:
                self.send_data(400, {"error": str(error)})
            except Exception:  # noqa: BLE001 - HTTP boundary must not disclose model internals.
                self.send_data(503, {"error": "本地预测暂不可用，请检查项目文件是否完整"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8510)
    args = parser.parse_args()
    with create_server(port=args.port) as server:
        print(f"Project website: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
