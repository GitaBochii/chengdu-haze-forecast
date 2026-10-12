"""Pollution histories with fitting restricted to unique training-window hours."""
import numpy as np
import pandas as pd
import torch

from haze_forecast.regional_air_experiment import TrainingScaler, causal_fill
from haze_forecast.regional_air_scoring import active_node_mask
from haze_forecast.recovery_survival import validate_hourly_grid

POLLUTION = ['pm2_5', 'pm10', 'aqi', 'so2', 'no2', 'co', 'o3_8h']
WEATHER = ['temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']
FEATURES = POLLUTION + WEATHER + ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                                 'wind_speed_kmh', 'wind_from_deg']


def prepare_pollution_histories(stations, weather, candidates):
    """Return [origin,24,node,17] inputs and the exact training fit population.

    Missing-mask True means an unobserved raw feature before causal filling.
    Calendar and wind conventions match regional_air_experiment.prepare_panel.
    No label or spatial embedding is accessed. Candidate order is preserved.
    """
    if not candidates or stations.duplicated(['timestamp','node_id']).any() or weather.timestamp.duplicated().any():
        raise ValueError('nonempty candidates and unique station/weather times required')
    origins = pd.DatetimeIndex([pd.Timestamp(row['timestamp']) for row in candidates])
    if origins.tz is None or origins.hasnans or origins.has_duplicates:
        raise ValueError('unique timezone-aware origins required')
    partitions = [row['partition'] for row in candidates]
    if not set(partitions).issubset({'train','dev','test'}) or 'train' not in partitions:
        raise ValueError('train/dev/test partitions and at least one train row required')
    times = pd.DatetimeIndex(sorted(stations.timestamp.unique()))
    validate_hourly_grid(times)
    origins = origins.tz_convert(times.tz)
    positions = times.get_indexer(origins)
    if (positions < 23).any():
        raise ValueError('each origin needs 24 actual hourly grid positions')
    windows = positions[:, None] + np.arange(-23,1)
    train_rows = np.asarray([part == 'train' for part in partitions])
    fit_positions = np.unique(windows[train_rows].ravel())
    fit_times = times[fit_positions]
    all_nodes = sorted(stations.node_id.unique().tolist())
    support = np.stack([stations.pivot(index='timestamp',columns='node_id',values=column)
                        .reindex(index=fit_times,columns=all_nodes).to_numpy(dtype=float)
                        for column in ('pm2_5','aqi')],axis=-1)
    coverage = np.isfinite(support).mean(axis=0)
    nodes = [node for node, keep in zip(all_nodes,(coverage >= .7).all(axis=1)) if keep]
    if len(nodes) < 2:
        raise ValueError('at least two training-qualified nodes required for GNN')
    raw_pollution = np.stack([stations.pivot(index='timestamp',columns='node_id',values=column)
                             .reindex(index=times,columns=nodes).to_numpy(dtype=float)
                             for column in POLLUTION],axis=-1)
    met = weather.set_index('timestamp').reindex(times)[WEATHER].to_numpy(dtype=float)
    met = np.repeat(met[:,None,:],len(nodes),axis=1)
    hour = 2*np.pi*times.hour.to_numpy()/24
    month = 2*np.pi*times.month.to_numpy()/12
    calendar = np.stack([np.sin(hour),np.cos(hour),np.sin(month),np.cos(month)],axis=-1)
    calendar = np.repeat(calendar[:,None,:],len(nodes),axis=1)
    speed = np.hypot(met[:,:,2],met[:,:,3])*3.6
    direction = np.degrees(np.arctan2(-met[:,:,2],-met[:,:,3])) % 360
    raw = np.concatenate([raw_pollution,met,calendar,np.stack([speed,direction],axis=-1)],axis=-1)
    if np.isinf(raw).any():
        raise ValueError('infinite measurements are invalid')
    scaler = TrainingScaler.fit(raw[fit_positions])
    filled = causal_fill(raw,scaler.mean)
    inputs = scaler.transform(filled)
    raw_targets = raw_pollution[:,:,[0,2]]
    active = active_node_mask(raw_targets,positions,lookback=24)
    if not active.any(axis=1).all():
        raise ValueError('each target needs an observed station in every history')
    for node in nodes:
        coords = stations.loc[stations.node_id.eq(node),['longitude','latitude']].drop_duplicates()
        if len(coords) != 1 or not np.isfinite(coords.to_numpy()).all():
            raise ValueError('station coordinates must be finite and time-invariant')
    coords = stations.drop_duplicates('node_id').set_index('node_id').loc[nodes,['longitude','latitude']].to_numpy()
    return {'x':torch.from_numpy(inputs[windows]), 'active':torch.from_numpy(active),
            'coords':coords, 'scaler_mean':scaler.mean.copy(), 'scaler_scale':scaler.scale.copy(),
            'nodes':nodes, 'feature_names':list(FEATURES),
            'training_timestamps':[time.isoformat() for time in fit_times],
            'raw_missing_mask':torch.from_numpy(~np.isfinite(raw[windows]))}
