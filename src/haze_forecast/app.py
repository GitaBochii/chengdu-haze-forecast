"""成都雾霾消散时间预测本地界面。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from haze_forecast import labels
from haze_forecast.features import MEASUREMENT_COLUMNS, build_features
from haze_forecast.labels import add_particle_primary
from haze_forecast.prediction_interval import DEFAULT_COVERAGE, prediction_interval
from haze_forecast.spatial import build_city_spatial_features

# Keep a hot-reloaded Streamlit process compatible while the label module refreshes.
AQI_RECOVERY_THRESHOLD = float(getattr(labels, "AQI_RECOVERY_THRESHOLD", 100.0))
RECOVERY_STABLE_HOURS = int(getattr(labels, "RECOVERY_STABLE_HOURS", 6))

POLLUTANT_DISPLAY = (
    ("aqi", "AQI", ""),
    ("pm2_5", "PM2.5", "µg/m³"),
    ("pm10", "PM10", "µg/m³"),
    ("so2", "SO2", "µg/m³"),
    ("no2", "NO2", "µg/m³"),
    ("co", "CO", "mg/m³"),
    ("o3", "O3-8h", "µg/m³"),
)
TARGET_OPTIONS = ("整体空气质量恢复", "颗粒物雾霾消散")
HISTORY_REPLAY_SOURCES = (
    "历史记录（2014-2021）",
    "历史记录（2022）",
    "历史记录（2023）",
    "历史记录（2024）",
    "历史记录（2025）",
)


@dataclass(frozen=True)
class DissipationCriterion:
    """Describe the recovery endpoint used by one prediction target."""

    threshold_label: str
    detail: str
    aqi_threshold: float | None


def dissipation_criterion(target: str) -> DissipationCriterion:
    """Return the target-specific definition shown beside a duration prediction."""
    if target == "整体空气质量恢复":
        return DissipationCriterion(
            threshold_label=f"AQI ≤ {AQI_RECOVERY_THRESHOLD:.0f}",
            detail=(
                f"AQI 降至 {AQI_RECOVERY_THRESHOLD:.0f} 以下，且不再属于颗粒物主导污染，"
                f"连续保持 {RECOVERY_STABLE_HOURS} 小时。"
            ),
            aqi_threshold=AQI_RECOVERY_THRESHOLD,
        )
    if target == "颗粒物雾霾消散":
        return DissipationCriterion(
            threshold_label="不设固定 AQI 阈值",
            detail=(
                "PM2.5 与 PM10 的 24 小时 IAQI 均不再超过 100，"
                f"连续保持 {RECOVERY_STABLE_HOURS} 小时。"
            ),
            aqi_threshold=None,
        )
    raise ValueError(f"未知预测目标：{target}")


def apply_workbench_style() -> None:
    """Keep the Streamlit controls compact and make prediction results easier to scan."""
    st.markdown(
        """
        <style>
        .stApp { background: #f3f6f4; color: #13221c; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; }
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stMetricLabel"] p,
        .stApp h1, .stApp h2, .stApp h3 { color: #22352b !important; }
        .stApp [data-testid="stCaptionContainer"] p { color: #485e53 !important; }
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricValue"] p { color: #0b6b45 !important; }
        .stApp [data-baseweb="select"] > div,
        .stApp input { background-color: #ffffff !important; color: #22352b !important; }
        .stApp [data-baseweb="select"] span { color: #22352b !important; }
        .stApp [data-testid="stHeader"] { background: #f3f6f4; color: #22352b; }
        .block-container { max-width: 1540px; padding-top: 4.5rem; padding-bottom: 2.25rem; }
        section[data-testid="stSidebar"] { background: #f8fbf9; border-right: 1px solid #d6e0da; }
        section[data-testid="stSidebar"] > div:first-child { padding: 1.4rem 1rem; }
        h1 { font-size: 2.05rem !important; font-weight: 750 !important; letter-spacing: 0 !important; margin-bottom: 0.15rem !important; }
        h2, h3 { letter-spacing: 0 !important; }
        .stCaption { color: #60756b; }
        .app-shell {
            align-items: center; background: #0b3028; border-radius: 8px; color: #ffffff;
            display: flex; justify-content: space-between; padding: 1rem 1.25rem;
        }
        .app-shell-name { font-size: 1.05rem; font-weight: 700; letter-spacing: 0; }
        .app-shell-meta { margin-top: 0.25rem; color: #bad1c3; font-size: 0.8rem; }
        .app-shell-mark {
            border: 1px solid #4f7967; border-radius: 999px; color: #d5efde; font-size: 0.75rem;
            font-weight: 700; padding: 0.28rem 0.6rem; white-space: nowrap;
        }
        div[data-testid="stRadio"] > div { gap: 0.35rem; }
        div[data-testid="stRadio"] label { border-radius: 5px; padding: 0.25rem 0.35rem; }
        div[data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #d5e0d9; border-radius: 7px;
            padding: 0.9rem 1rem; min-height: 116px; box-shadow: 0 2px 8px rgba(14, 48, 38, 0.04);
        }
        div[data-testid="stMetricLabel"] { color: #567066; font-size: 0.85rem; }
        div[data-testid="stMetricValue"] { color: #0b6b45; font-size: 1.55rem; }
        .stButton > button, .stDownloadButton > button {
            border-radius: 6px; border-color: #0d6d49; color: #ffffff; background: #0d6d49;
            font-weight: 600;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { background: #073f2d; border-color: #073f2d; }
        div[data-testid="stAlert"] {
            border: 1px solid #cfe0d5; border-radius: 6px; background: #f2f8f4; color: #234a39;
        }
        div[data-testid="stSelectbox"] > div > div, div[data-testid="stNumberInput"] input,
        div[data-testid="stDateInput"] input {
            border-color: #c9d8d0; border-radius: 6px;
        }
        div[data-testid="stDataFrame"] { border: 1px solid #d5e0d9; border-radius: 7px; overflow: hidden; background: #ffffff; }
        .workbench-kicker { color: #0d6d49; font-size: 0.78rem; font-weight: 750; letter-spacing: 0; margin-top: 1.35rem; }
        .workbench-rule { border-top: 1px solid #d5e0d9; margin: 1.15rem 0 1.45rem; }
        .workbench-result { color: #13221c; font-size: 1.15rem; font-weight: 750; margin: 0.2rem 0 0.75rem; }
        @media (max-width: 700px) {
            .block-container { padding: 4.5rem 1rem 1.8rem; }
            .app-shell { align-items: flex-start; gap: 0.65rem; }
            .app-shell-mark { font-size: 0.7rem; }
            h1 { font-size: 1.65rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def prediction_message(is_pollution_event: bool, remaining_hours: float, dissipates_at: str) -> str:
    if not is_pollution_event:
        return "当前不满足颗粒物超标污染条件"
    return f"预计还需 {remaining_hours:.1f} 小时消散，预计消散时刻：{dissipates_at}"


def manual_feature_frame(feature_names: list[str], values: dict[str, float], timestamp: pd.Timestamp) -> pd.DataFrame:
    frame = pd.DataFrame(np.nan, index=[0], columns=feature_names)
    for name, value in values.items():
        if name in frame:
            frame.at[0, name] = value
    derived = {
        "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
        "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
        "month_sin": np.sin(2 * np.pi * timestamp.month / 12),
        "month_cos": np.cos(2 * np.pi * timestamp.month / 12),
        "wind_speed_ms": np.hypot(values["wind_u_ms"], values["wind_v_ms"]),
        "particle_primary": float(
            values["aqi"] > 100 and (values["pm2_5"] >= 75 or values["pm10"] >= 150)
        ),
    }
    for name, value in derived.items():
        if name in frame:
            frame.at[0, name] = value
    return populate_missingness_indicators(frame)


def populate_missingness_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Populate candidate-model availability flags from their corresponding inputs."""
    result = frame.copy()
    suffix = "_is_missing"
    for name in result.columns:
        if not name.endswith(suffix):
            continue
        source = name.removesuffix(suffix)
        if source in result:
            result[name] = result[source].isna().astype(float)
        elif source.endswith("_change_3"):
            measured = source.removesuffix("_change_3")
            lagged = f"{measured}_lag_3"
            if measured in result and lagged in result:
                result[name] = (result[measured].isna() | result[lagged].isna()).astype(float)
    return result


def history_backtest_feature_frame(row: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    """Align a stored historical row with the model schema before predicting."""
    frame = row.to_frame().T.reindex(columns=feature_names)
    return populate_missingness_indicators(frame).astype(float)


def training_sample_count(metadata: dict[str, object]) -> int | None:
    """Read the training-row count from either baseline or candidate metadata."""
    rows = metadata.get("rows")
    if isinstance(rows, dict) and isinstance(rows.get("train"), int):
        return rows["train"]
    final_training_rows = metadata.get("final_training_rows")
    return final_training_rows if isinstance(final_training_rows, int) else None


def build_history_prediction_input(
    history: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.Series, pd.DataFrame]:
    """Build one feature row from continuous observations up to the prediction timestamp."""
    required = {"timestamp", *MEASUREMENT_COLUMNS}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"缺少必要列：{', '.join(sorted(missing))}")

    prepared = history.copy()
    timestamps = pd.to_datetime(prepared["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp 存在无法识别的时间")
    if timestamps.dt.tz is None:
        prepared["timestamp"] = timestamps.dt.tz_localize("Asia/Shanghai")
    else:
        prepared["timestamp"] = timestamps.dt.tz_convert("Asia/Shanghai")
    for name in MEASUREMENT_COLUMNS:
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    if len(prepared) < 24:
        raise ValueError("至少需要最近 24 个连续小时的观测")
    recent_timestamps = prepared["timestamp"].tail(24)
    if not recent_timestamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("最近 24 个观测时刻必须连续，每小时一条")

    enriched = build_features(add_particle_primary(prepared))
    latest = enriched.iloc[-1]
    prediction_input = latest.to_frame().T.reindex(columns=feature_names)
    return latest, populate_missingness_indicators(prediction_input).astype(float)


def build_station_history_prediction_input(
    history: pd.DataFrame, feature_names: list[str], coordinates: dict[str, list[float]]
) -> tuple[pd.Series, pd.DataFrame]:
    """Build city and transport features from at least 24 recent multi-station observations."""
    required = {"timestamp", "station_id", *MEASUREMENT_COLUMNS}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"缺少必要列：{', '.join(sorted(missing))}")

    prepared = history.copy()
    timestamps = pd.to_datetime(prepared["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp 存在无法识别的时间")
    if timestamps.dt.tz is None:
        prepared["timestamp"] = timestamps.dt.tz_localize("Asia/Shanghai")
    else:
        prepared["timestamp"] = timestamps.dt.tz_convert("Asia/Shanghai")
    for name in MEASUREMENT_COLUMNS:
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
    prepared = prepared.sort_values(["timestamp", "station_id"]).reset_index(drop=True)
    recent_timestamps = prepared["timestamp"].drop_duplicates().tail(24)
    if len(recent_timestamps) < 24 or not recent_timestamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("最近至少需要 24 个连续小时的站点观测")
    recent = prepared[prepared["timestamp"].isin(recent_timestamps)]
    if recent.groupby("timestamp")["station_id"].nunique().lt(2).any():
        raise ValueError("最近每个小时至少需要两个站点的观测")

    city = recent.groupby("timestamp", as_index=False)[list(MEASUREMENT_COLUMNS)].median()
    spatial = build_city_spatial_features(recent, coordinates)
    enriched = build_features(add_particle_primary(city)).merge(
        spatial, on="timestamp", how="left", validate="one_to_one"
    )
    latest = enriched.iloc[-1]
    prediction_input = pd.DataFrame(np.nan, index=[0], columns=feature_names)
    available = [name for name in feature_names if name in latest]
    prediction_input.loc[:, available] = latest[available].astype(float).to_numpy()
    return latest, populate_missingness_indicators(prediction_input).astype(float)


def pollutant_display_frame(row: pd.Series) -> pd.DataFrame:
    """Convert model field names to a concise, unit-aware display table."""
    return pd.DataFrame(
        [
            {"污染物": label, "数值": float(row[key]), "单位": unit}
            for key, label, unit in POLLUTANT_DISPLAY
            if key in row
        ]
    )


def primary_particle_text(values: pd.Series | dict[str, float]) -> str:
    primary = []
    if float(values["pm2_5"]) >= 75:
        primary.append("PM2.5")
    if float(values["pm10"]) >= 150:
        primary.append("PM10")
    return "、".join(primary) if primary else "无颗粒物超标"


def target_resource_paths(target: str) -> tuple[str, str]:
    """Return the processed data file and artifact directory for one prediction target."""
    if target == "整体空气质量恢复":
        return "training_features.parquet", ""
    if target == "颗粒物雾霾消散":
        return "particle_dissipation_training_features.parquet", "particle_dissipation"
    raise ValueError(f"未知预测目标：{target}")


def trend_data_filename(target: str, use_spatial: bool) -> str:
    """Return the continuous labeled observations used only for the visible trend."""
    if target == "整体空气质量恢复":
        return "spatial_city_labeled.parquet" if use_spatial else "city_labeled.parquet"
    if target == "颗粒物雾霾消散":
        return "particle_dissipation_city_labeled.parquet"
    raise ValueError(f"未知预测目标：{target}")


def prepare_trend_context(observations: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Keep an hourly, causal observation window; missing hours remain gaps."""
    end = pd.Timestamp(timestamp)
    end = end.tz_localize("Asia/Shanghai") if end.tzinfo is None else end.tz_convert("Asia/Shanghai")
    frame = observations.loc[:, ["timestamp", "aqi", "pm2_5", "pm10"]].copy()
    times = pd.to_datetime(frame["timestamp"])
    frame["timestamp"] = (
        times.dt.tz_localize("Asia/Shanghai") if times.dt.tz is None
        else times.dt.tz_convert("Asia/Shanghai")
    )
    frame = frame.loc[frame["timestamp"].between(end - pd.Timedelta(hours=24), end)]
    frame = frame.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    grid = pd.date_range(end - pd.Timedelta(hours=24), end, freq="h", name="timestamp")
    return frame.reindex(grid).reset_index()


def build_trend_figure(context: pd.DataFrame, timestamp: pd.Timestamp, threshold: float | None):
    """Use explicit local clock labels, visible single points and separate units."""
    end = pd.Timestamp(timestamp)
    end = end.tz_localize("Asia/Shanghai") if end.tzinfo is None else end.tz_convert("Asia/Shanghai")
    local_end = end.tz_localize(None)
    local_times = context["timestamp"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for column, name, color, secondary in (
        ("aqi", "AQI", "#08754f", False),
        ("pm2_5", "PM2.5", "#286b9a", True),
        ("pm10", "PM10", "#ae6618", True),
    ):
        figure.add_trace(go.Scatter(
            x=local_times, y=context[column], name=name, mode="lines+markers",
            connectgaps=False, line={"color": color, "width": 2.5}, marker={"size": 5},
            hovertemplate="%{x|%m月%d日 %H:%M}<br>%{y:.1f}<extra>%{fullData.name}</extra>",
        ), secondary_y=secondary)
    figure.update_layout(
        template="plotly_white", title="过去24小时污染物观测", height=420,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"family": "Microsoft YaHei UI, Segoe UI, sans-serif", "color": "#22352b", "size": 14},
        title_font={"color": "#22352b", "size": 18},
        legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"color": "#22352b", "size": 14}},
        margin={"l": 60, "r": 65, "t": 100, "b": 65}, hovermode="x unified",
    )
    figure.update_xaxes(
        range=[local_end - pd.Timedelta(hours=24), local_end],
        title_text="北京时间", title_font={"color": "#22352b", "size": 14},
        tickformat="%m/%d<br>%H:%M", dtick=6 * 3600 * 1000,
        tickfont={"color": "#22352b", "size": 13}, showgrid=False, linecolor="#9dac9f",
    )
    figure.update_yaxes(
        tickfont={"color": "#22352b", "size": 13}, title_font={"color": "#22352b", "size": 14},
        gridcolor="#dce5df", zeroline=False,
    )
    figure.update_yaxes(title_text="AQI", secondary_y=False)
    figure.update_yaxes(title_text="颗粒物浓度 (µg/m³)", showgrid=False, secondary_y=True)
    if threshold is not None:
        figure.add_hline(y=threshold, secondary_y=False, line_color="#08754f", line_dash="dot")
    return figure


def external_history_data_filename(year: int) -> str:
    """Return the labelled external-year data used for an honest history backtest."""
    if year not in (2022, 2023, 2024, 2025):
        raise ValueError("历史记录目前提供 2022 至 2025 年")
    return f"cnemc_chengdu_{year}_training_features.parquet"


def history_backtest_source_year(source: str) -> int | None:
    """Map a visible historical replay source to its data year."""
    source_years = {
        "历史记录（2014-2021）": None,
        "历史记录（2022）": 2022,
        "历史记录（2023）": 2023,
        "历史记录（2024）": 2024,
        "历史记录（2025）": 2025,
    }
    try:
        return source_years[source]
    except KeyError as error:
        raise ValueError(f"未知回测数据集：{source}") from error


def training_history_backtest_data(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the visible training-history choice aligned with its 2014-2021 label."""
    timestamps = pd.to_datetime(data["timestamp"], errors="raise")
    return data.loc[timestamps.dt.year.le(2021)].copy()


def history_error_label(is_training_replay: bool) -> str:
    """Avoid presenting fitted-history differences as an external forecast error."""
    return "历史回放差异" if is_training_replay else "本次绝对误差"


def model_version_options(candidate_available: bool) -> tuple[str, ...]:
    """Expose one future-facing model instead of mixing deployment and comparison artifacts."""
    return ("2025全量部署模型（预测未来）",) if candidate_available else ("基准模型",)


def prediction_resource_paths(target: str, use_spatial: bool) -> tuple[str, str]:
    """Return the feature dataset and artifact directory for the selected prediction path."""
    if use_spatial:
        if target != "整体空气质量恢复":
            raise ValueError("空间传播模型暂仅适用于整体空气质量恢复")
        return "spatial_training_features.parquet", "spatial_transport"
    return target_resource_paths(target)


def candidate_prediction_resource_paths(
    target: str,
    use_spatial: bool,
    use_cnemc_2014_candidate: bool,
    use_full_history_deployment_candidate: bool = False,
) -> tuple[str, str]:
    """Select isolated 2014 candidate artifacts without changing input features."""
    if use_cnemc_2014_candidate and use_full_history_deployment_candidate:
        raise ValueError("一次只能选择一个候选模型")
    if use_full_history_deployment_candidate:
        if target != "整体空气质量恢复" or use_spatial:
            raise ValueError("2025 全量部署模型仅适用于整体空气质量恢复")
        return (
            "cnemc_chengdu_2014_2025_deployment_training_features.parquet",
            "cnemc_2025_deployment_candidate",
        )
    data_name, artifact_directory = prediction_resource_paths(target, use_spatial)
    if use_cnemc_2014_candidate:
        if target != "整体空气质量恢复" or use_spatial:
            raise ValueError("2014 历史扩展模型不适用于当前预测路径")
        return data_name, "cnemc_2014_candidate"
    return data_name, artifact_directory


def constrain_live_prediction(
    previous_timestamp: pd.Timestamp | None,
    previous_prediction: float | None,
    current_timestamp: pd.Timestamp,
    raw_prediction: float,
) -> float:
    """Apply the event-state constraint only to consecutive hourly updates."""
    if (
        previous_timestamp is None
        or previous_prediction is None
        or current_timestamp - previous_timestamp != pd.Timedelta(hours=1)
    ):
        return raw_prediction
    return min(raw_prediction, max(0.0, previous_prediction - 1.0))


def evaluation_metric_key(manual: bool, sequence: bool, model_filename: str) -> str:
    """Select the held-out metric that matches the active prediction path."""
    if "aft" in model_filename:
        return "xgboost_aft_snapshot" if manual else "xgboost_aft"
    if manual:
        return "catboost_snapshot"
    if sequence:
        return "lightgbm_state_constrained"
    return "lightgbm"


def model_specification(
    metadata: dict[str, object], metrics: dict[str, object], manual: bool
) -> tuple[list[str], float | None, str]:
    """Choose a model whose training inputs match the selected prediction mode."""
    if manual:
        if "selected_snapshot_model_filename" in metadata:
            # Legacy full-history artifacts copied the history-only score here.
            unvalidated = metadata.get("deployment_role") == "full_history_refit_for_future_predictions"
            value = metadata.get("selected_snapshot_validation_mae")
            return (
                list(metadata["snapshot_feature_names"]),
                None if unvalidated or value is None else float(value),
                str(metadata["selected_snapshot_model_filename"]),
            )
        if "snapshot_aft_model_filename" in metadata:
            return (
                list(metadata["snapshot_feature_names"]),
                float(metrics["aft_snapshot_validation_mae"]),
                str(metadata["snapshot_aft_model_filename"]),
            )
        return (
            list(metadata["snapshot_feature_names"]),
            float(metrics["snapshot_validation_mae"]),
            str(metadata.get("snapshot_model_filename", "static_lightgbm_model.joblib")),
        )
    if "selected_history_model_filename" in metadata:
        return (
            list(metadata["feature_names"]),
            float(metadata["selected_history_validation_mae"]),
            str(metadata["selected_history_model_filename"]),
        )
    if "aft_model_filename" in metadata:
        return (
            list(metadata["feature_names"]),
            float(metrics["aft_validation_mae"]),
            str(metadata["aft_model_filename"]),
        )
    return list(metadata["feature_names"]), float(metrics["validation_mae"]), "lightgbm_model.joblib"


@st.cache_data
def load_data(
    target: str,
    use_spatial: bool = False,
    use_cnemc_2014_candidate: bool = False,
    use_full_history_deployment_candidate: bool = False,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    root = Path(__file__).resolve().parents[2]
    data_name, artifact_directory = candidate_prediction_resource_paths(
        target,
        use_spatial,
        use_cnemc_2014_candidate,
        use_full_history_deployment_candidate,
    )
    artifacts = root / "artifacts" / artifact_directory
    data = pd.read_parquet(root / "data" / "processed" / data_name)
    metadata = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    return data, metadata, metrics


@st.cache_data
def load_trend_data(target: str, use_spatial: bool) -> pd.DataFrame:
    """Load continuous hourly observations rather than event-only training rows."""
    root = Path(__file__).resolve().parents[2]
    continuous = root / "data/processed/regional_air_models_full_history/city_hourly.parquet"
    path = continuous if continuous.exists() else root / "data" / "processed" / trend_data_filename(target, use_spatial)
    return pd.read_parquet(path, columns=["timestamp", "aqi", "pm2_5", "pm10"])


@st.cache_data
def load_external_history_backtest_data(year: int) -> pd.DataFrame:
    """Load a completed later year that was not used by the default model fit."""
    root = Path(__file__).resolve().parents[2]
    data = pd.read_parquet(root / "data" / "processed" / external_history_data_filename(year))
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    return data.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def load_strict_history_backtest_data() -> pd.DataFrame:
    """Load predictions materialized without the evaluated event or any later events."""
    root = Path(__file__).resolve().parents[2]
    data = pd.read_parquet(root / "artifacts" / "strict_history_backtest" / "predictions.parquet")
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data["train_end_timestamp"] = pd.to_datetime(data["train_end_timestamp"], errors="raise")
    return data.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def load_interval_calibration() -> dict[str, object] | None:
    """Load the coverage-calibrated interval, or None when it has not been built."""
    root = Path(__file__).resolve().parents[2]
    path = root / "artifacts" / "prediction_interval_calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def uncertainty_band(
    prediction: float,
    fallback_width: float | None,
    calibration: dict[str, object] | None,
) -> tuple[str, str]:
    """Render the prediction band and the wording that explains it."""
    if fallback_width is None:
        return "暂不可用", "当前输入模式尚未单独验证，暂不提供预测区间。"
    levels = calibration.get("calibration") if calibration else None
    if not levels:
        lower = max(0.0, prediction - fallback_width)
        return (
            f"{lower:.1f} - {prediction + fallback_width:.1f} 小时",
            f"经验区间：预测值 ±{fallback_width:.1f} 小时（未经覆盖率校准）",
        )
    coverage = float(calibration.get("default_coverage", DEFAULT_COVERAGE))
    lower, upper = prediction_interval(prediction, levels, coverage)
    coverage_text = (
        f"{coverage:.0%} 覆盖率区间：同类样本外数据实测覆盖 "
        f"{_calibration_coverage(calibration, coverage)}"
    )
    return f"{lower:.1f} - {upper:.1f} 小时", coverage_text


def _calibration_coverage(calibration: dict[str, object], coverage: float) -> str:
    rows = [
        entry
        for entry in calibration.get("evaluation", [])
        if float(entry["coverage_target"]) == coverage
    ]
    if not rows:
        return "见校准报告"
    observed = [float(entry["observed_coverage"]) for entry in rows]
    return f"{min(observed):.0%}–{max(observed):.0%}"


@st.cache_resource
def load_model(artifact_directory: str, filename: str):
    root = Path(__file__).resolve().parents[2]
    return joblib.load(root / "artifacts" / artifact_directory / filename)


def main() -> None:
    from haze_forecast.workbench_ui import apply_style, render_header, render_results

    st.set_page_config(page_title="成都雾霾消散预测", layout="wide", initial_sidebar_state="collapsed")
    apply_style()
    mode = render_header()
    filters = st.columns([1.3, 1, 1.4, 0.8])
    target = filters[0].selectbox("预测目标", TARGET_OPTIONS)
    is_strict_history_backtest = mode == "严格历史回测"
    is_history_mode = mode in {"严格历史回测", "历史回放"}
    history_source = HISTORY_REPLAY_SOURCES[0]
    if mode == "历史回放" and target == "整体空气质量恢复":
        history_source = st.selectbox("历史数据集", HISTORY_REPLAY_SOURCES)
    external_history_year = history_backtest_source_year(history_source)
    use_external_history = external_history_year is not None
    candidate_available = target == "整体空气质量恢复" and mode != "24小时多站点序列输入"
    model_version = model_version_options(candidate_available)[0]
    use_cnemc_2014_candidate = model_version == "2014历史扩展模型"
    use_full_history_deployment_candidate = model_version == "2025全量部署模型（预测未来）"
    use_spatial = (
        not use_cnemc_2014_candidate
        and not use_full_history_deployment_candidate
        and not use_external_history
        and target == "整体空气质量恢复"
        and mode in {"历史回放", "24小时多站点序列输入"}
    )
    try:
        data, metadata, metrics = load_data(
            target,
            use_spatial,
            use_cnemc_2014_candidate,
            use_full_history_deployment_candidate,
        )
    except FileNotFoundError:
        st.error("尚未找到训练产物。请先完成数据处理与模型训练。")
        return
    history_data = data
    if is_strict_history_backtest:
        if target != "整体空气质量恢复":
            st.info("严格历史回测目前仅适用于整体空气质量恢复目标。")
            return
        try:
            history_data = load_strict_history_backtest_data()
        except FileNotFoundError:
            st.error("尚未生成严格历史回测结果。")
            return
    elif external_history_year is not None:
        try:
            history_data = load_external_history_backtest_data(external_history_year)
        except FileNotFoundError:
            st.error(f"尚未找到 {external_history_year} 年历史记录。")
            return
    elif use_full_history_deployment_candidate:
        history_data = training_history_backtest_data(data)

    _, artifact_directory = candidate_prediction_resource_paths(
        target,
        use_spatial,
        use_cnemc_2014_candidate,
        use_full_history_deployment_candidate,
    )
    is_manual = mode == "手动输入"
    feature_names, interval_width, model_filename = model_specification(metadata, metrics, is_manual)
    try:
        model = load_model(artifact_directory, model_filename)
    except FileNotFoundError:
        st.error("尚未找到与当前预测方式匹配的模型。请重新训练模型。")
        return
    backtest_error_hours: float | None = None
    if is_history_mode:
        dates = pd.Series(history_data["timestamp"].dt.date.unique()).sort_values()
        selected_date = filters[1].selectbox("选择历史日期", dates.tolist(), index=len(dates) - 1)
        candidates = history_data[history_data["timestamp"].dt.date.eq(selected_date)].copy()
        selected_time = filters[2].selectbox("选择污染时刻（北京时间）", candidates["timestamp"].tolist(),
                                            format_func=lambda value: value.strftime("%Y-%m-%d %H:%M"))
        row = candidates.loc[candidates["timestamp"].eq(selected_time)].iloc[0]
        if is_strict_history_backtest:
            prediction = max(0.0, float(row["strict_prediction_hours"]))
        else:
            prediction_input = history_backtest_feature_frame(row, feature_names)
            prediction = max(0.0, float(model.predict(prediction_input)[0]))
        actual_text = f"{row['remaining_hours']:.1f} 小时"
        backtest_error_hours = abs(prediction - float(row["remaining_hours"]))
    elif mode == "24小时序列输入":
        template = pd.DataFrame(columns=["timestamp", *MEASUREMENT_COLUMNS]).to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载 CSV 模板", template, "haze_history_template.csv", "text/csv")
        uploaded = st.file_uploader("上传最近 24 小时 CSV", type=["csv"])
        if uploaded is None:
            st.info("上传数据后开始预测。")
            return
        try:
            uploaded_history = pd.read_csv(uploaded)
            row, prediction_input = build_history_prediction_input(uploaded_history, feature_names)
        except (ValueError, pd.errors.ParserError) as error:
            st.error(f"无法使用该序列：{error}")
            return
        prediction = max(0.0, float(model.predict(prediction_input)[0]))
        is_pollution_event = bool(row["aqi"] > 100 and row["particle_primary"])
        state_key = f"dissipation_state_{target}"
        if is_pollution_event and model_filename == "lightgbm_model.joblib":
            state = st.session_state.get(state_key, {})
            prediction = constrain_live_prediction(
                state.get("timestamp"), state.get("prediction"), row["timestamp"], prediction
            )
            st.session_state[state_key] = {"timestamp": row["timestamp"], "prediction": prediction}
        else:
            st.session_state.pop(state_key, None)
        actual_text = "序列输入无历史真实值"
        context = uploaded_history
    elif mode == "24小时多站点序列输入":
        if target != "整体空气质量恢复":
            st.info("多站点空间传播模型目前用于整体空气质量恢复目标。")
            return
        template = pd.DataFrame(columns=["timestamp", "station_id", *MEASUREMENT_COLUMNS]).to_csv(index=False).encode(
            "utf-8-sig"
        )
        st.download_button("下载多站点 CSV 模板", template, "haze_station_history_template.csv", "text/csv")
        uploaded = st.file_uploader("上传最近 24 小时多站点 CSV", type=["csv"])
        if uploaded is None:
            st.info("上传各站点数据后开始预测。")
            return
        try:
            uploaded_history = pd.read_csv(uploaded)
            root = Path(__file__).resolve().parents[2]
            coordinates = json.loads(
                (root / "data" / "raw" / "data" / "chengdu" / "stations.json").read_text("utf-8")
            )
            row, prediction_input = build_station_history_prediction_input(
                uploaded_history, feature_names, coordinates
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(f"无法使用该序列：{error}")
            return
        prediction = max(0.0, float(model.predict(prediction_input)[0]))
        is_pollution_event = bool(row["aqi"] > 100 and row["particle_primary"])
        actual_text = "多站点序列输入无历史真实值"
        context = uploaded_history.groupby("timestamp", as_index=False)[list(MEASUREMENT_COLUMNS)].median()
    else:
        defaults = {
            "aqi": 130.0,
            "pm2_5": 90.0,
            "pm10": 120.0,
            "so2": 12.0,
            "no2": 35.0,
            "co": 1.0,
            "o3": 50.0,
            "temperature_c": 12.0,
            "relative_humidity_pct": 70.0,
            "wind_u_ms": 1.0,
            "wind_v_ms": 0.0,
            "boundary_layer_height_m": 300.0,
        }
        labels = {
            "aqi": "AQI",
            "pm2_5": "PM2.5 (µg/m³)",
            "pm10": "PM10 (µg/m³)",
            "so2": "SO2 (µg/m³)",
            "no2": "NO2 (µg/m³)",
            "co": "CO (mg/m³)",
            "o3": "O3-8h (µg/m³)",
            "temperature_c": "气温 (°C)",
            "relative_humidity_pct": "相对湿度 (%)",
            "wind_u_ms": "东西向风速 (m/s)",
            "wind_v_ms": "南北向风速 (m/s)",
            "boundary_layer_height_m": "边界层高度 (m)",
        }
        with st.form("manual_prediction"):
            selected_date = st.date_input("观测日期", value=pd.Timestamp.now(tz="Asia/Shanghai").date())
            selected_hour = st.slider("观测小时", min_value=0, max_value=23, value=pd.Timestamp.now(tz="Asia/Shanghai").hour)
            st.caption("污染物浓度")
            values = {name: st.number_input(labels[name], value=value) for name, value in defaults.items()}
            submitted = st.form_submit_button("预测消散时间")
        if not submitted:
            st.info("填写观测值后，点击“预测消散时间”。")
            return
        timestamp = pd.Timestamp(selected_date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=selected_hour)
        row = pd.Series(values)
        is_pollution_event = bool(values["aqi"] > 100 and (values["pm2_5"] >= 75 or values["pm10"] >= 150))
        prediction = max(0.0, float(model.predict(manual_feature_frame(feature_names, values, timestamp))[0])) if is_pollution_event else 0.0
        row["timestamp"] = timestamp
        actual_text = "手动输入无历史真实值"
    if is_history_mode:
        is_pollution_event = True
    dissipates_at = row["timestamp"] + pd.Timedelta(hours=prediction)
    criterion = dissipation_criterion(target)

    band_text, band_caption = "", ""
    if is_pollution_event:
        band_text, band_caption = uncertainty_band(
            prediction, interval_width, load_interval_calibration()
        )

    if is_history_mode:
        context = load_trend_data(target, use_spatial)
    elif mode not in {"24小时序列输入", "24小时多站点序列输入"}:
        context = pd.DataFrame([row])
    context = prepare_trend_context(context, row["timestamp"])
    observed_count = int(context[["aqi", "pm2_5", "pm10"]].notna().any(axis=1).sum())
    error_label = "严格回测绝对误差" if is_strict_history_backtest else history_error_label(use_full_history_deployment_candidate)
    render_results(
        row=row, context=context, figure=build_trend_figure(context, row["timestamp"], criterion.aqi_threshold),
        mode=mode, prediction=prediction, dissipates_at=dissipates_at,
        band_text=band_text, band_caption=band_caption, actual_text=actual_text,
        error=backtest_error_hours, error_label=error_label, criterion=criterion,
        observed_count=observed_count, particle=primary_particle_text(row),
        pollutants=pollutant_display_frame(row), is_pollution_event=is_pollution_event,
        download_slot=filters[3],
    )
    metric_key = evaluation_metric_key(is_manual, mode == "24小时序列输入", model_filename)
    model_label = (
        "滚动 XGBoost AFT 生存模型"
        if is_strict_history_backtest
        else (
            "XGBoost AFT 生存模型"
            if "aft" in model_filename
            else "CatBoost"
            if "catboost" in model_filename
            else "LightGBM"
        )
    )
    training_rows = training_sample_count(metadata)
    metric_caption = (
        f"严格历史回测 MAE：{history_data['strict_absolute_error_hours'].mean():.2f} 小时"
        if is_strict_history_backtest
        else "当前输入模式 MAE：尚未单独验证"
        if interval_width is None
        else (
            f"严格外部参考 MAE：{metrics[metric_key]['mae']:.2f} 小时"
            if metadata.get("accuracy_claim") == "external_validation_only"
            else f"保留测试集 MAE：{metrics[metric_key]['mae']:.2f} 小时"
        )
    )
    if is_strict_history_backtest:
        st.caption(
            f"预测目标：{target}；模型版本：{model_label}；{metric_caption}；"
            f"可评估样本：{len(history_data)}。"
        )
    else:
        st.caption(
            f"预测目标：{target}；模型版本：{model_label}；{metric_caption}；"
            f"训练样本：{training_rows if training_rows is not None else '未记录'}。"
            "历史真实值仅用于回测对照，不参与预测输入。"
        )
    if metadata.get("accuracy_claim") == "external_validation_only" and not is_strict_history_backtest:
        st.caption(
            "该版本已使用截至 2025 年的完整历史训练，适用于后续未知过程；"
            "历史日期的单次差异不作为其真实预测精度。"
        )
    if is_strict_history_backtest:
        st.caption(
            f"本次预测仅使用截至 {row['train_end_timestamp']:%Y-%m-%d %H:%M} 的已结束污染过程训练，"
            f"训练样本：{int(row['train_rows'])}。当前事件及之后数据均未参与训练。"
        )
    if external_history_year is not None:
        if use_full_history_deployment_candidate:
            st.caption(
                f"{external_history_year} 年已用于该部署模型训练；当前展示为历史回放，"
                "不计入严格预测精度。"
            )
        else:
            st.caption(
                f"历史记录年份：{external_history_year}。该年度真实标签仅用于展示预测误差，"
                "未作为当前模型的预测输入。"
            )


if __name__ == "__main__":
    main()
