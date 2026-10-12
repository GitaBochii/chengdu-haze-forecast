"""同一时刻站点观测构造的无未来泄漏空间传播特征。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

POLLUTANTS = ("pm2_5", "pm10")
REQUIRED_COLUMNS = {"timestamp", "station_id", *POLLUTANTS, "wind_u_ms", "wind_v_ms"}


def _projected_positions(
    station_ids: Sequence[str], coordinates: Mapping[str, Sequence[float]]
) -> np.ndarray:
    longitude_latitude = []
    for station_id in station_ids:
        if station_id not in coordinates:
            raise ValueError(f"missing coordinates for station: {station_id}")
        longitude, latitude = coordinates[station_id]
        longitude_latitude.append((float(longitude), float(latitude)))
    values = np.asarray(longitude_latitude, dtype=float)
    reference_latitude = np.deg2rad(values[:, 1].mean())
    return np.column_stack((values[:, 0] * 111.32 * np.cos(reference_latitude), values[:, 1] * 110.57))


def _upwind_city_mean(
    concentrations: np.ndarray, wind: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Return one wind-aligned concentration mean per timestamp without row-wise group loops."""
    source_to_receptor = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(source_to_receptor, axis=2)
    direction = np.divide(
        source_to_receptor,
        distances[:, :, None],
        out=np.zeros_like(source_to_receptor),
        where=distances[:, :, None] > 1e-6,
    )
    speed = np.linalg.norm(wind, axis=2)
    normalized_wind = np.divide(
        wind,
        speed[:, :, None],
        out=np.zeros_like(wind),
        where=speed[:, :, None] > 1e-6,
    )
    alignment = np.maximum(0.0, np.einsum("jic,tjc->tji", direction, normalized_wind))
    weights = alignment / np.where(distances > 1e-6, distances + 1.0, np.inf)
    weights *= np.isfinite(concentrations)[:, None, :]
    denominator = weights.sum(axis=2)
    numerator = np.einsum("tji,ti->tj", weights, np.nan_to_num(concentrations, nan=0.0))
    receptor_mean = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    active_receptors = denominator > 0
    return np.divide(
        (receptor_mean * active_receptors).sum(axis=1),
        active_receptors.sum(axis=1),
        out=np.full(len(concentrations), np.nan),
        where=active_receptors.sum(axis=1) > 0,
    )


def _wind_direction_consistency(wind: np.ndarray) -> np.ndarray:
    """Measure whether the available station winds point in a common direction."""
    speed = np.linalg.norm(wind, axis=2)
    unit_wind = np.divide(
        wind,
        speed[:, :, None],
        out=np.zeros_like(wind),
        where=speed[:, :, None] > 1e-6,
    )
    active = speed > 1e-6
    count = active.sum(axis=1)
    return np.divide(
        np.linalg.norm(unit_wind.sum(axis=1), axis=1),
        count,
        out=np.full(len(wind), np.nan),
        where=count > 0,
    )


def _directional_pollutant_features(
    concentrations: np.ndarray, wind: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return same-hour directional contrast, spatial gradient, and usable pair count.

    Wind vectors follow the transport direction.  Therefore a source west of a
    receptor is upwind when the receptor's wind points east.
    """
    station_count = len(positions)
    source_from_receptor = positions[None, :, :] - positions[:, None, :]
    distances = np.linalg.norm(source_from_receptor, axis=2)
    distinct_stations = distances > 1e-6
    inverse_distance = np.divide(
        1.0,
        distances + 1.0,
        out=np.zeros_like(distances),
        where=distinct_stations,
    )
    upper_triangle = np.triu(np.ones((station_count, station_count), dtype=bool), k=1)
    contrasts = np.full(len(concentrations), np.nan)
    gradients = np.full(len(concentrations), np.nan)
    pair_counts = np.zeros(len(concentrations), dtype=float)

    for time_index, concentration in enumerate(concentrations):
        finite = np.isfinite(concentration)
        concentration_pairs = finite[:, None] & finite[None, :]
        valid_gradient_pairs = upper_triangle & concentration_pairs & distinct_stations
        if valid_gradient_pairs.any():
            gradients[time_index] = np.median(
                np.abs(concentration[:, None] - concentration[None, :])[valid_gradient_pairs]
                / distances[valid_gradient_pairs]
            )

        local_wind = wind[time_index]
        speed = np.linalg.norm(local_wind, axis=1)
        unit_wind = np.divide(
            local_wind,
            speed[:, None],
            out=np.zeros_like(local_wind),
            where=speed[:, None] > 1e-6,
        )
        available_receptor = finite & (speed > 1e-6)
        usable_directed_pairs = available_receptor[:, None] & finite[None, :] & distinct_stations
        pair_counts[time_index] = float(usable_directed_pairs.sum())
        if not usable_directed_pairs.any():
            continue

        alignment = np.einsum("rsc,rc->rs", source_from_receptor, unit_wind)
        upwind = usable_directed_pairs & (alignment < 0)
        downwind = usable_directed_pairs & (alignment > 0)
        upwind_weights = inverse_distance * upwind
        downwind_weights = inverse_distance * downwind
        upwind_denominator = upwind_weights.sum(axis=1)
        downwind_denominator = downwind_weights.sum(axis=1)
        upwind_mean = np.divide(
            upwind_weights @ np.nan_to_num(concentration, nan=0.0),
            upwind_denominator,
            out=np.zeros(station_count),
            where=upwind_denominator > 0,
        )
        downwind_mean = np.divide(
            downwind_weights @ np.nan_to_num(concentration, nan=0.0),
            downwind_denominator,
            out=np.zeros(station_count),
            where=downwind_denominator > 0,
        )
        comparable_receptors = (upwind_denominator > 0) & (downwind_denominator > 0)
        if comparable_receptors.any():
            contrasts[time_index] = np.mean(
                upwind_mean[comparable_receptors] - downwind_mean[comparable_receptors]
            )

    return contrasts, gradients, pair_counts


def build_city_spatial_features(
    station_hourly: pd.DataFrame, coordinates: Mapping[str, Sequence[float]]
) -> pd.DataFrame:
    """Aggregate station spread and wind-aligned upwind pollution for each hour."""
    missing = REQUIRED_COLUMNS - set(station_hourly.columns)
    if missing:
        raise ValueError(f"missing station fields: {sorted(missing)}")

    prepared = station_hourly.sort_values(["timestamp", "station_id"]).copy()
    if prepared.duplicated(["timestamp", "station_id"]).any():
        raise ValueError("station data must contain at most one row per station and timestamp")
    station_ids = sorted(prepared["station_id"].unique())
    positions = _projected_positions(station_ids, coordinates)
    timestamps = pd.Index(prepared["timestamp"].drop_duplicates(), name="timestamp")
    grid_index = pd.MultiIndex.from_product([timestamps, station_ids], names=["timestamp", "station_id"])
    indexed = prepared.set_index(["timestamp", "station_id"])
    grid = indexed.reindex(grid_index)
    shape = (len(timestamps), len(station_ids))
    wind = grid[["wind_u_ms", "wind_v_ms"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    wind = wind.reshape(*shape, 2)
    present = indexed.assign(_present=1)["_present"].reindex(grid_index, fill_value=0).to_numpy()
    result = pd.DataFrame(
        {"timestamp": timestamps, "spatial_station_count": present.reshape(shape).sum(axis=1).astype(float)}
    )
    result["spatial_wind_direction_consistency"] = _wind_direction_consistency(wind)
    for pollutant in POLLUTANTS:
        values = pd.to_numeric(grid[pollutant], errors="coerce").to_numpy(dtype=float).reshape(shape)
        valid_count = np.isfinite(values).sum(axis=1)
        mean = np.divide(
            np.nansum(values, axis=1), valid_count, out=np.zeros(len(values)), where=valid_count > 0
        )
        variance = np.divide(
            np.nansum((values - mean[:, None]) ** 2, axis=1),
            valid_count,
            out=np.zeros(len(values)),
            where=valid_count > 0,
        )
        minimum = np.where(np.isfinite(values), values, np.inf).min(axis=1)
        maximum = np.where(np.isfinite(values), values, -np.inf).max(axis=1)
        result[f"spatial_{pollutant}_std"] = np.where(valid_count > 0, np.sqrt(variance), np.nan)
        result[f"spatial_{pollutant}_range"] = np.where(valid_count > 0, maximum - minimum, np.nan)
        result[f"spatial_upwind_{pollutant}_mean"] = _upwind_city_mean(values, wind, positions)
        contrast, gradient, pair_count = _directional_pollutant_features(values, wind, positions)
        result[f"spatial_{pollutant}_upwind_downwind_difference"] = contrast
        result[f"spatial_{pollutant}_gradient_median"] = gradient
        result[f"spatial_{pollutant}_transport_pair_count"] = pair_count
    return result


def enrich_city_with_spatial_features(
    city_hourly: pd.DataFrame,
    station_hourly: pd.DataFrame,
    coordinates: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Attach same-hour station transport signals to one row per city timestamp."""
    if city_hourly["timestamp"].duplicated().any():
        raise ValueError("city data must contain at most one row per timestamp")
    spatial = build_city_spatial_features(station_hourly, coordinates)
    return city_hourly.merge(spatial, on="timestamp", how="left", validate="one_to_one")
