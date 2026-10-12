"""Run an isolated, chronological real-data specialist-model pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from haze_forecast.labels import label_city_data
from haze_forecast.regional_air_experiment import (
    boundary_event_mask,
    masked_mae,
    prepare_panel,
    purge_event_windows,
    recovery_metrics,
    sequence_origins,
    stable_recovery,
)
from haze_forecast.regional_air_networks import PM25GNNAdapter, PM25GNNHistoryAdapter, SequenceGRU
from haze_forecast.regional_air_scoring import (
    active_node_mask,
    aggregate_active_predictions,
    verified_recovery_mask,
)
from haze_forecast.regional_airformer import create_airformer
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/regional_air_models"


def hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.glob("*")) if p.is_file()}


def write_json(path, value):
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            obj = obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


class Windows(Dataset):
    def __init__(self, panel, origins, history, horizon):
        self.panel, self.origins = panel, origins
        self.history, self.horizon = history, horizon
        self.mean = panel["scaler"].mean[[0, 2]].astype(np.float32)
        self.scale = panel["scaler"].scale[[0, 2]].astype(np.float32)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, item):
        i = self.origins[item]
        x = self.panel["inputs"][i - self.history + 1:i + 1]
        raw = self.panel["targets"][i + 1:i + self.horizon + 1]
        mask = np.isfinite(raw)
        y = np.nan_to_num((raw - self.mean) / self.scale, nan=0.0)
        return torch.from_numpy(x.copy()), torch.from_numpy(y), torch.from_numpy(mask)


def masked_loss(prediction, actual, mask):
    if not mask.any():
        return None
    return torch.abs(prediction - actual)[mask].mean()


def predict(model, panel, origins, args, device):
    loader = DataLoader(Windows(panel, origins, args.history, args.horizon),
                        batch_size=args.batch_size, shuffle=False)
    model.eval()
    pieces = []
    with torch.no_grad():
        for x, _, _ in loader:
            pieces.append(model(x.to(device)).cpu().numpy())
    normalized = np.concatenate(pieces)
    return np.maximum(0, normalized * panel["scaler"].scale[[0, 2]]
                      + panel["scaler"].mean[[0, 2]])


def train_model(model, panel, train_origins, val_origins, args, output, name, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    train = DataLoader(Windows(panel, train_origins, args.history, args.horizon),
                       batch_size=args.batch_size, shuffle=True, generator=generator)
    valid = DataLoader(Windows(panel, val_origins, args.history, args.horizon),
                       batch_size=args.batch_size, shuffle=False)
    best_loss, best_state, stale, trace = float("inf"), None, 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y, mask in train:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), y, mask)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss in {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = int(mask.sum())
            total += float(loss.detach()) * n
            count += n
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for x, y, mask in valid:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                loss = masked_loss(model(x), y, mask)
                if loss is not None:
                    n = int(mask.sum())
                    val_total += float(loss) * n
                    val_count += n
        if not count or not val_count:
            raise ValueError("training/validation must contain observed labels")
        val_loss = val_total / val_count
        trace.append({"epoch": epoch, "training_loss": total / count,
                      "validation_2024_loss": val_loss,
                      "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"model": name, **trace[-1]}), flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output / f"{name}_state.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    write_json(output / f"{name}_training.json", trace)
    return {"best_validation_loss": best_loss, "epochs_ran": len(trace),
            "elapsed_seconds": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters())}


def evaluate(prediction, panel, origins, city, labels, args, output, name, year):
    times = panel["timestamps"]
    target_positions = origins[:, None] + np.arange(1, args.horizon + 1)[None, :]
    targets = panel["targets"][target_positions]
    active = active_node_mask(panel["targets"], origins, lookback=args.history)
    city_prediction = aggregate_active_predictions(prediction, active)
    city_truth = city.reindex(times)[["pm2_5", "aqi"]].to_numpy()[target_positions]
    results = {"forecast_origins": len(origins), "station_metrics": {}, "city_metrics": {}}
    for channel, pollutant in enumerate(("pm2_5", "aqi")):
        results["station_metrics"][pollutant] = masked_mae(
            prediction[..., channel], targets[..., channel])
        results["city_metrics"][pollutant] = masked_mae(
            city_prediction[..., channel], city_truth[..., channel])
    results["city_metrics_by_lead"] = {
        str(lead): {pollutant: masked_mae(city_prediction[:, lead - 1, channel],
                                        city_truth[:, lead - 1, channel])
                    for channel, pollutant in enumerate(("pm2_5", "aqi"))}
        for lead in (1, 6, 12, 24, 48, 72) if lead <= args.horizon
    }
    # Keep every forecast origin. Restrict duration scoring to the original labeled events.
    event = labels.reindex(times[origins])
    records = pd.DataFrame({
        "timestamp": times[origins], "model": name,
        "event_id": event["event_id"].to_numpy(),
        "actual_hours": event["remaining_hours"].to_numpy(),
        "predicted_hours": [stable_recovery(curve) for curve in city_prediction[:, :, 1]],
        "actual_aqi_future_observed_fraction": np.isfinite(city_truth[:, :, 1]).mean(axis=1),
    })
    verified = verified_recovery_mask(records["actual_hours"].to_numpy(), city_truth[:, :, 1])
    records["label_verified_in_scoring_window"] = verified
    results["duration_unverified_sensitivity_only"] = recovery_metrics(records, args.horizon)
    results["duration"] = recovery_metrics(records.loc[verified], args.horizon)
    results["unverified_labeled_origins_excluded"] = int((
        records["actual_hours"].notna() & ~verified).sum())
    results["mean_active_stations_by_target"] = active.sum(axis=1).mean(axis=0).tolist()
    records.to_parquet(output / f"{name}_{year}_recovery.parquet", index=False)
    np.savez_compressed(output / f"{name}_{year}_curves.npz",
                        origin_ns=times[origins].asi8, prediction=prediction.astype(np.float32),
                        actual=targets, city_prediction=city_prediction.astype(np.float32),
                        city_actual=city_truth.astype(np.float32))
    return results


def build_model(name, panel, args):
    """Pair model initialization independently of execution order."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if name == 'gru':
        return SequenceGRU(len(panel['feature_names']), args.horizon)
    if name in ('pm25_gnn', 'pm25_gnn_history'):
        model_type = PM25GNNAdapter if name == 'pm25_gnn' else PM25GNNHistoryAdapter
        return model_type(len(panel['feature_names']), panel['coords'],
                          panel['scaler'].mean[-2:], panel['scaler'].scale[-2:],
                          args.history, args.horizon)
    if name == 'airformer':
        return create_airformer(len(panel['feature_names']), panel['coords'], args.history, args.horizon)
    raise ValueError(f'unknown model: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/regional_air_models/pilot_20260905")
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--train-start-year", type=int, default=2023)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--ozone-column", choices=["o3", "o3_8h"], default="o3")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=12)
    parser.add_argument("--eval-stride", type=int, default=6)
    parser.add_argument("--validation-stride", type=int, default=24)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-eval", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", default=["gru", "pm25_gnn", "airformer"],
                        choices=["gru", "pm25_gnn", "pm25_gnn_history", "airformer"])
    args = parser.parse_args()
    if not 2014 <= args.train_start_year <= args.train_end_year < 2024:
        parser.error("training years must lie within 2014-2023; 2024 selects checkpoints")
    if min(args.epochs, args.history, args.horizon, args.batch_size,
           args.train_stride, args.eval_stride, args.validation_stride) < 1:
        parser.error("training dimensions must be positive")
    if args.horizon < 6:
        parser.error("horizon must permit six-hour recovery confirmation")
    output = ROOT / args.output
    if (output / "report.json").exists():
        raise FileExistsError("choose a new output directory; existing results are immutable")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose --device cpu for a CPU pilot")
    deployed = ROOT / "artifacts/cnemc_2025_deployment_candidate"
    before = hashes(deployed)
    write_json(output / "deployment_before.json", before)
    data = ROOT / args.data
    stations = pd.read_parquet(data / "station_hourly.parquet")
    weather = pd.read_parquet(data / "weather_proxy.parquet")
    city_frame = pd.read_parquet(data / "city_hourly.parquet")
    city = city_frame.set_index("timestamp")
    labels = label_city_data(city_frame).set_index("timestamp")
    cutoff = pd.Timestamp(f"{args.train_end_year + 1}-01-01", tz="Asia/Shanghai")
    training_start = pd.Timestamp(f"{args.train_start_year}-01-01", tz="Asia/Shanghai")
    panel = prepare_panel(stations, weather, cutoff, training_start=training_start,
                          ozone_column=args.ozone_column)
    times = panel["timestamps"]
    blocked = boundary_event_mask(labels.reindex(times), [training_start, cutoff, *[
        pd.Timestamp(f"{year}-01-01", tz=times.tz) for year in (2024, 2025, 2026)]])

    def origins(year, stride, maximum=0):
        start = pd.Timestamp(f"{year}-01-01", tz=times.tz)
        end = min(pd.Timestamp(f"{year + 1}-01-01", tz=times.tz), times[-1] + pd.Timedelta(hours=1))
        selected = sequence_origins(times, history=args.history, horizon=args.horizon,
                                    start=start, end=end, stride=stride)
        selected = purge_event_windows(selected, blocked, history=args.history, horizon=args.horizon)
        # Drop forecast origins lacking contemporaneous proxy weather, not future weather.
        known_weather = weather.set_index("timestamp").reindex(times)[
            ["temperature_c", "relative_humidity_pct", "wind_u_ms", "wind_v_ms"]
        ].notna().all(axis=1).to_numpy()
        selected = selected[known_weather[selected]]
        active = active_node_mask(panel["targets"], selected, lookback=args.history)
        selected = selected[active.any(axis=1).all(axis=1)]
        return selected[:maximum] if maximum else selected

    training_by_year = {year: origins(year, args.train_stride)
                        for year in range(args.train_start_year, args.train_end_year + 1)}
    if any(not len(selected) for selected in training_by_year.values()):
        raise ValueError("every requested training year must contribute valid windows")
    train_origins = np.concatenate(list(training_by_year.values()))
    if args.max_train:
        train_origins = train_origins[:args.max_train]
    val_origins = origins(2024, args.validation_stride, args.max_eval)
    tests = {year: origins(year, args.eval_stride, args.max_eval) for year in (2025, 2026)}
    if not len(train_origins) or not len(val_origins) or any(not len(v) for v in tests.values()):
        raise ValueError("all declared partitions must contain valid windows")
    report = {
        "status": "bounded_pilot_not_deployment_evidence", "configuration": vars(args),
        "deployment_artifact_changed": False,
        "protocol": f"train {args.train_start_year}-{args.train_end_year}; checkpoint selection "
                    "2024; freeze weights for 2025 and 2026",
        "scope": "Chengdu intra-city stations, not inter-city transport validation",
        "scoring_protocol": "city median uses origin-known active stations; primary duration "
                            "scores require observed AQI through recovery confirmation or horizon",
        "limitations": [
            "2025 has been inspected in previous experiments; not a pristine holdout",
            "2026 is partial-year and previously inspected; not a pristine holdout",
            "ERA5 is retrospective shared-city proxy, not as-of station weather",
            "PM25-GNN uses persisted origin features, not future observed weather",
            "pm25_gnn uses latest state only; pm25_gnn_history warms the same state with earlier inputs",
            "PM25-GNN transport coefficient adapted from 3-hour upstream data to 1 hour",
            "PM25-GNN AQI head is an adaptation, not original PM2.5-only published model",
            "AirFormer deterministic regional adapter, not the full national paper setting",
            "Single-seed short training cannot establish converged model-family superiority",
            "72h curves cannot provide uncensored durations for long events",
            "Historical station coordinates not independently verified against official metadata",
        ],
        "nodes": panel["nodes"], "coordinates_lon_lat": panel["coords"].tolist(),
        "feature_names": panel["feature_names"],
        "training_node_coverage": panel["training_coverage"],
        "input_data_sha256": hashes(data),
        "scaler": {"mean": panel["scaler"].mean.tolist(), "scale": panel["scaler"].scale.tolist()},
        "partitions": {"training": len(train_origins), "validation": len(val_origins),
                       **{str(k): len(v) for k, v in tests.items()}},
        "training_windows_by_year": {str(year): int((times[train_origins].year == year).sum())
                                     for year in training_by_year},
        "partition_crossing_event_hours_excluded": int(blocked.sum()),
        "training_last_target": str(times[train_origins[-1] + args.horizon]),
        "validation_last_target": str(times[val_origins[-1] + args.horizon]),
        "models": {},
    }
    write_json(output / "protocol.json", report)

    # Same-input duration baseline; it does not use the deployed model's larger training history.
    active_history = active_node_mask(panel["targets"], np.arange(len(times)),
                                      lookback=args.history).any(axis=2)
    history_mask = np.broadcast_to(~active_history[:, :, None], panel["inputs"].shape)
    history_values = pd.DataFrame(np.ma.median(
        np.ma.array(panel["inputs"], mask=history_mask), axis=1).filled(0), index=times)
    aft_features = pd.concat([history_values.add_prefix("now_"),
                              history_values.rolling(args.history).mean().add_prefix("mean_")], axis=1)
    recovery_end = labels["remaining_hours"].reindex(times)
    available_by = times + pd.to_timedelta(recovery_end + 5, unit="h")
    aft_train = np.asarray((times >= training_start) & (times < cutoff) & (available_by < cutoff)
                           & recovery_end.notna().to_numpy(), dtype=bool)
    aft_train &= ~blocked
    aft_history_safe = np.zeros(len(times), dtype=bool)
    aft_history_safe[purge_event_windows(np.arange(args.history - 1, len(times)), blocked,
                                         history=args.history, horizon=0)] = True
    aft_train &= aft_history_safe
    aft_train &= aft_features.notna().all(axis=1).to_numpy()
    # Exact AFT labels require no unobserved AQI between origin and confirmation.
    city_aqi = city.reindex(times)["aqi"].to_numpy()
    cumulative_missing = np.cumsum(~np.isfinite(city_aqi))
    for origin in np.flatnonzero(aft_train):
        finish = origin + int(recovery_end.iloc[origin]) + 5
        aft_train[origin] = (finish < len(times) and
                             cumulative_missing[finish] == cumulative_missing[origin])
    if aft_train.sum() < 30:
        raise ValueError("insufficient completed training events for matched AFT baseline")
    aft = AFTSurvivalRegressor(nthread=1).fit(aft_features.loc[aft_train], recovery_end.loc[aft_train])
    joblib.dump(aft, output / f"matched_{args.train_start_year}_{args.train_end_year}_aft.joblib")
    report["matched_aft_training_rows"] = int(aft_train.sum())
    report["models"]["matched_aft"] = {"years": {}}
    report["models"]["persistence"] = {"years": {}}
    for year, selected in tests.items():
        event = labels.reindex(times[selected])
        aft_records = pd.DataFrame({"timestamp": times[selected],
                                         "actual_hours": event["remaining_hours"].to_numpy(),
                                         "event_id": event["event_id"].to_numpy(),
                                         "predicted_hours": aft.predict(aft_features.iloc[selected])})
        positions = selected[:, None] + np.arange(1, args.horizon + 1)[None, :]
        verified = verified_recovery_mask(aft_records["actual_hours"].to_numpy(),
                                           city_aqi[positions])
        aft_records["label_verified_in_scoring_window"] = verified
        aft_records.to_parquet(output / f"matched_aft_{year}_recovery.parquet", index=False)
        report["models"]["matched_aft"]["years"][str(year)] = {
            "duration": recovery_metrics(aft_records.loc[verified], args.horizon),
            "duration_unverified_sensitivity_only": recovery_metrics(aft_records, args.horizon)}
        last = panel["inputs"][selected][:, :, [0, 2]] * panel["scaler"].scale[[0, 2]]
        last += panel["scaler"].mean[[0, 2]]
        baseline = np.repeat(last[:, None], args.horizon, axis=1)
        report["models"]["persistence"]["years"][str(year)] = evaluate(
            baseline, panel, selected, city, labels, args, output, "persistence", year)
    write_json(output / "progress.json", report)
    for name in args.models:
        model = build_model(name, panel, args)
        training_report = train_model(model, panel, train_origins, val_origins,
                                       args, output, name, device)
        report["models"][name] = {"training": training_report, "years": {}}
        for year, selected in tests.items():
            prediction = predict(model, panel, selected, args, device)
            result = evaluate(prediction, panel, selected, city, labels, args, output, name, year)
            report["models"][name]["years"][str(year)] = result
            print(json.dumps({"model": name, "year": year, "city": result["city_metrics"],
                              "duration": result["duration"]}), flush=True)
        write_json(output / "progress.json", report)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    after = hashes(deployed)
    report["deployment_artifact_changed"] = before != after
    write_json(output / "deployment_after.json", after)
    if before != after:
        raise RuntimeError("deployment changed during the experiment; investigate concurrent edits")
    report["decision"] = "do_not_promote_short_single_seed_retrospective_pilot"
    write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
