"""Chronological, masked evaluation utilities for isolated specialist-model pilots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrainingScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training: np.ndarray) -> TrainingScaler:
        values = np.asarray(training, dtype=float)
        if values.ndim != 3 or not len(values):
            raise ValueError("expected non-empty [time, node, feature] training data")
        if not np.isfinite(values).any(axis=(0, 1)).all():
            raise ValueError("each feature must have an observed training value")
        mean = np.nanmean(values, axis=(0, 1))
        scale = np.nanstd(values, axis=(0, 1))
        return cls(mean, np.where(scale > 1e-6, scale, 1.0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)


def causal_fill(values: np.ndarray, fallback: np.ndarray, limit: int = 6) -> np.ndarray:
    """Forward-fill inputs only, then use training-only fallback values."""
    if limit < 1:
        raise ValueError("fill limit must be positive")
    result = np.asarray(values, dtype=float).copy()
    for node in range(result.shape[1]):
        result[:, node] = pd.DataFrame(result[:, node]).ffill(limit=limit).to_numpy()
    return np.where(np.isfinite(result), result, np.asarray(fallback)).astype(np.float32)


def sequence_origins(
    timestamps: pd.DatetimeIndex, *, history: int, horizon: int,
    start: pd.Timestamp, end: pd.Timestamp, stride: int = 1,
) -> np.ndarray:
    """Indices with entirely contiguous history/target windows and target < end."""
    if min(history, horizon, stride) < 1:
        raise ValueError("window sizes and stride must be positive")
    times = pd.DatetimeIndex(timestamps)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted and unique")
    candidates = np.arange(history - 1, len(times) - horizon, stride, dtype=int)
    if not len(candidates):
        return candidates
    bad_step = np.r_[0, np.diff(times.asi8) != pd.Timedelta(hours=1).value]
    cumulative = np.cumsum(bad_step)
    continuous = cumulative[candidates + horizon] == cumulative[candidates - history + 1]
    allowed = (times[candidates] >= start) & (times[candidates + horizon] < end)
    return candidates[continuous & allowed]


def stable_recovery(curve: np.ndarray, threshold: float = 100, stable_hours: int = 6) -> float:
    values = np.asarray(curve, dtype=float)
    if values.ndim != 1 or stable_hours < 1:
        raise ValueError("expected a one-dimensional curve and a positive stable window")
    for start in range(len(values) - stable_hours + 1):
        window = values[start:start + stable_hours]
        if np.isfinite(window).all() and (window <= threshold).all():
            return float(start + 1)
    return float("nan")


def masked_mae(prediction: np.ndarray, actual: np.ndarray) -> float:
    if prediction.shape != actual.shape:
        raise ValueError("prediction and target shapes differ")
    valid = np.isfinite(actual)
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("non-finite predictions for observed targets")
    return float(np.abs(prediction[valid] - actual[valid]).mean()) if valid.any() else float("nan")


def prepare_panel(
    stations: pd.DataFrame, weather: pd.DataFrame, training_end: pd.Timestamp,
    minimum_coverage: float = 0.7,
    *, training_start: pd.Timestamp | None = None, ozone_column: str = "o3",
) -> dict:
    """Prepare a fixed training-selected node set; never fill evaluation labels."""
    if stations.duplicated(["timestamp", "node_id"]).any():
        raise ValueError("station timestamps must be unique")
    if weather["timestamp"].duplicated().any():
        raise ValueError("weather timestamps must be unique")
    times = pd.DatetimeIndex(sorted(stations["timestamp"].unique()))
    if training_start is None:
        training_start = times[0]
    if training_start >= training_end:
        raise ValueError("training_start must precede training_end")
    training = stations.loc[stations["timestamp"].ge(training_start)
                            & stations["timestamp"].lt(training_end)]
    coverage = training.groupby("node_id")[["pm2_5", "aqi"]].agg(lambda s: s.notna().mean())
    nodes = sorted(coverage.index[coverage.min(axis=1) >= minimum_coverage].tolist())
    if not nodes:
        raise ValueError("no nodes meet training-only coverage requirements")
    if ozone_column not in ("o3", "o3_8h"):
        raise ValueError("ozone column must declare hourly or eight-hour semantics")
    pollution = ["pm2_5", "pm10", "aqi", "so2", "no2", "co", ozone_column]
    columns = [*pollution, "temperature_c", "relative_humidity_pct", "wind_u_ms", "wind_v_ms"]
    inputs = np.stack([
        stations.pivot(index="timestamp", columns="node_id", values=column)
        .reindex(index=times, columns=nodes).to_numpy() for column in pollution
    ], axis=-1)
    targets = inputs[:, :, [0, 2]].copy()
    weather_values = weather.set_index("timestamp").reindex(times)[columns[7:]].to_numpy()
    weather_nodes = np.repeat(weather_values[:, None, :], len(nodes), axis=1)
    hour = 2 * np.pi * times.hour.to_numpy() / 24
    month = 2 * np.pi * times.month.to_numpy() / 12
    calendar = np.stack([np.sin(hour), np.cos(hour), np.sin(month), np.cos(month)], axis=-1)
    calendar = np.repeat(calendar[:, None, :], len(nodes), axis=1)
    # Meteorological FROM direction in degrees, speed in km/h (official GNN convention).
    speed = np.hypot(weather_nodes[:, :, 2], weather_nodes[:, :, 3]) * 3.6
    direction = np.degrees(np.arctan2(-weather_nodes[:, :, 2], -weather_nodes[:, :, 3])) % 360
    inputs = np.concatenate([inputs, weather_nodes, calendar,
                             np.stack([speed, direction], axis=-1)], axis=-1)
    columns += ["hour_sin", "hour_cos", "month_sin", "month_cos", "wind_speed_kmh", "wind_from_deg"]
    train_mask = (times >= training_start) & (times < training_end)
    scaler = TrainingScaler.fit(inputs[train_mask])
    filled = causal_fill(inputs, scaler.mean)
    coords = stations.drop_duplicates("node_id").set_index("node_id").loc[
        nodes, ["longitude", "latitude"]
    ].to_numpy()
    return {
        "timestamps": times, "nodes": nodes, "coords": coords,
        "inputs": scaler.transform(filled), "targets": targets.astype(np.float32),
        "scaler": scaler, "feature_names": columns,
        "training_coverage": coverage.to_dict(),
        "weather_missing_fraction": float(np.isnan(weather_values).mean()),
    }


def boundary_event_mask(labels: pd.DataFrame, boundaries: list[pd.Timestamp]) -> np.ndarray:
    """Mark whole completed events crossing a partition, including six clean hours."""
    times = pd.DatetimeIndex(labels.index)
    blocked = np.zeros(len(times), dtype=bool)
    for _, group in labels.loc[labels.event_id.gt(0)].groupby("event_id"):
        start = group.index.min()
        confirmation = group.index.max() + pd.Timedelta(hours=6)
        if any(start < boundary <= confirmation for boundary in boundaries):
            blocked |= (times >= start) & (times <= confirmation)
    return blocked


def purge_event_windows(origins: np.ndarray, blocked: np.ndarray, *, history: int,
                        horizon: int) -> np.ndarray:
    """Reject windows whose history or targets overlap a partition-crossing event."""
    cumulative = np.r_[0, np.cumsum(blocked)]
    clean = cumulative[origins + horizon + 1] == cumulative[origins - history + 1]
    return origins[clean]


def recovery_metrics(records: pd.DataFrame, horizon: int = 72) -> dict:
    """Use capped-time error for all events, not a success-only duration score."""
    observed = records.loc[records["actual_hours"].notna()].copy()
    if observed.empty:
        return {"event_rows": 0, "events": 0}
    # A 72-hour curve confirms six clean hours only for starts through hour 67.
    cap = horizon - 6 + 2
    actual = observed["actual_hours"].to_numpy()
    predicted = observed["predicted_hours"].to_numpy()
    actual_recovery = actual < cap
    predicted_recovery = np.isfinite(predicted) & (predicted < cap)
    error = np.abs(np.minimum(actual, cap) - np.minimum(np.nan_to_num(predicted, nan=cap), cap))
    observed["restricted_error"] = error
    both = actual_recovery & predicted_recovery
    return {
        "event_rows": len(observed), "events": int(observed["event_id"].nunique()),
        "restricted_time_cap_hours": cap,
        "restricted_time_mae": float(error.mean()),
        "event_equal_restricted_time_mae": float(
            observed.groupby("event_id")["restricted_error"].mean().mean()),
        "predicted_recovery_fraction": float(predicted_recovery.mean()),
        "actual_recovery_fraction": float(actual_recovery.mean()),
        "missed_recovery_rate": (float((~predicted_recovery[actual_recovery]).mean())
                                 if actual_recovery.any() else None),
        "false_recovery_rate": (float(predicted_recovery[~actual_recovery].mean())
                                if (~actual_recovery).any() else None),
        "conditional_recovery_mae": (float(np.abs(actual[both] - predicted[both]).mean())
                                     if both.any() else None),
        "conditional_recovery_rows": int(both.sum()),
    }
