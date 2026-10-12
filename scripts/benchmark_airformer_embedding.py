"""Frozen spatiotemporal representations paired with uncapped interval AFT."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.regional_air_experiment import prepare_panel
from haze_forecast.regional_air_scoring import active_node_mask
from scripts.benchmark_regional_air_models import build_model

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    old = base / 'regional_rolling_2023_20260916'
    original = base / 'grouped_forecast_aft_2023_20260916'
    out = base / 'airformer_embedding_2023_20260918_v2'
    if out.exists():
        raise FileExistsError(out)
    neural = json.loads((old / 'protocol.json').read_bytes())
    aft = json.loads((original / 'protocol.json').read_bytes())
    hashes = {**neural['input_sha256'], **aft['input_sha256']}
    old_report = json.loads((old / 'report.json').read_bytes())
    for name, expected in old_report['output_sha256'].items():
        assert digest(old / name) == expected, name
    extra = [Path(__file__).resolve(), ROOT / 'airformer_embedding_plan_20260918.md',
             old / 'airformer_state.pt', old / 'protocol.json', old / 'report.json',
             original / 'expanded.parquet',
             base / 'station_dispersion_2023_20260918/all_predictions.parquet']
    hashes.update({str(p): digest(p) for p in extra})
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    train = pd.read_parquet(base / 'gefs_2021_extension_v1/f024_training_rows.parquet')
    valid = pd.read_parquet(base / 'gefs_2021_extension_v1/inference_rows.parquet')
    cohort = pd.read_parquet(original / 'expanded.parquet')
    reference = pd.read_parquet(extra[-1]).set_index('timestamp').loc[valid.timestamp, 'native'].to_numpy()
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert not set(train.continuity_block) & set(cohort.continuity_block)
    assert neural['training_years'] == [2014, 2021] and neural['selection_year'] == 2022
    for filename, end in [('training_windows.parquet', cutoff - pd.DateOffset(years=1)),
                          ('selection_windows.parquet', cutoff)]:
        windows = pd.read_parquet(old / filename)
        assert (windows.timestamp + pd.Timedelta(hours=72) < end).all()
    data = ROOT / 'data/processed/regional_air_models_full_history'
    stations = pd.read_parquet(data / 'station_hourly.parquet')
    weather = pd.read_parquet(data / 'weather_proxy.parquet')
    end = pd.Timestamp('2024-01-01', tz='Asia/Shanghai')
    panel = prepare_panel(stations.loc[stations.timestamp.lt(end)], weather.loc[weather.timestamp.lt(end)],
                          pd.Timestamp('2022-01-01', tz='Asia/Shanghai'),
                          training_start=pd.Timestamp('2014-01-01', tz='Asia/Shanghai'), ozone_column='o3_8h')
    assert panel['nodes'] == neural['nodes']
    for key in ('mean', 'scale'):
        np.testing.assert_array_equal(getattr(panel['scaler'], key), neural['scaler'][key])
    combined = pd.concat([train, valid], ignore_index=True)
    assert combined.timestamp.is_unique
    positions = panel['timestamps'].get_indexer(combined.timestamp)
    assert (positions >= 23).all()
    for pos in positions:
        assert np.all(np.diff(panel['timestamps'][pos-23:pos+1].asi8) == 3600000000000)
    active = active_node_mask(panel['targets'], positions)[:, :, 1]
    assert active.any(axis=1).all()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for frozen-path replay')
    torch.set_num_threads(2)
    out.mkdir()
    protocol = {'input_sha256': hashes, 'baseline_features': aft['features'], 'params': aft['params'],
                'rounds': 500, 'training_rows': len(train), 'inference_rows': len(valid),
                'pool': 'mean of past-24h AQI-active nodes', 'target': 'full remaining hours',
                'encoder_training_years': neural['training_years'], 'encoder_selection_year': 2022,
                'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    columns = [f'air_encoding_{i:02d}' for i in range(64)]
    scores = {}
    for arm in ('native', 'random', 'trained'):
        features = combined[aft['features']].copy()
        if arm != 'native':
            model = build_model('airformer', panel, SimpleNamespace(**neural['configuration']))
            if arm == 'trained':
                model.load_state_dict(torch.load(old / 'airformer_state.pt', weights_only=True, map_location='cpu'))
            model.to('cuda').eval()
            captured = []
            handle = model.end_conv_1.register_forward_pre_hook(
                lambda module, args, captured=captured: captured.append(args[0].detach()))
            embeddings = []
            with torch.inference_mode():
                for start in range(0, len(positions), 256):
                    pos = positions[start:start+256]
                    x = torch.from_numpy(np.stack([panel['inputs'][i-23:i+1] for i in pos])).to('cuda')
                    captured.clear()
                    model(x)
                    z = captured[0].squeeze(-1)
                    assert z.shape == (len(pos), 64, len(panel['nodes']))
                    if start == 0:
                        state = model.start_conv(x.permute(0, 3, 2, 1))
                        states = []
                        for spatial, temporal, norm in zip(model.s_modules, model.t_modules, model.bn, strict=True):
                            state = norm(temporal(spatial(state)))
                            states.append(state)
                        explicit = torch.relu(torch.cat(states, dim=1)[..., -1])
                        torch.testing.assert_close(z, explicit, rtol=0, atol=0)
                    mask = torch.from_numpy(active[start:start+256]).to('cuda')
                    pooled = (z * mask[:, None]).sum(-1) / mask.sum(-1)[:, None]
                    embeddings.append(pooled.cpu().numpy())
            handle.remove()
            values = np.concatenate(embeddings)
            assert values.shape == (len(combined), 64) and np.isfinite(values).all()
            features = pd.concat([features, pd.DataFrame(values, columns=columns)], axis=1)
            torch.save(model.cpu().state_dict(), out / f'{arm}_encoder.pt')
            del model
        features.insert(0, 'timestamp', combined.timestamp)
        features.to_parquet(out / f'{arm}_features.parquet', index=False)
        x = features.drop(columns='timestamp')
        matrix = xgb.DMatrix(x.iloc[:len(train)], missing=np.nan)
        matrix.set_float_info('label_lower_bound', train.lower_bound_hours)
        matrix.set_float_info('label_upper_bound', train.upper_bound_hours)
        inference = xgb.DMatrix(x.iloc[len(train):], missing=np.nan)
        booster = xgb.train(aft['params'], matrix, num_boost_round=500)
        prediction = booster.predict(inference)
        if arm == 'native':
            np.testing.assert_array_equal(prediction, reference)
        booster.save_model(out / f'{arm}.ubj')
        restored = xgb.Booster()
        restored.load_model(out / f'{arm}.ubj')
        np.testing.assert_array_equal(prediction, restored.predict(inference))
        valid[arm] = prediction
        cohort[arm] = pd.Series(prediction, index=valid.timestamp).loc[cohort.timestamp].to_numpy()
        scores[arm] = score_predictions(cohort, cohort[arm].to_numpy())
        print(json.dumps({'arm': arm, 'score': scores[arm]}), flush=True)
    paired = {arm: paired_error_summary(cohort, cohort[arm], cohort.trained) for arm in ('native', 'random')}
    gates = {arm: all(scores['trained'][k] < scores[arm][k] for k in ('exact_row_mae', 'exact_block_mae'))
             and scores['trained']['tail24_block_mae'] <= scores[arm]['tail24_block_mae']
             and paired[arm]['all_row_delta_lower'] <= 0 and paired[arm]['all_block_delta_lower'] <= 0
             for arm in ('native', 'random')}
    valid[['timestamp', 'native', 'random', 'trained']].to_parquet(out / 'all_predictions.parquet', index=False)
    cohort.to_parquet(out / 'scored_predictions.parquet', index=False)
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    report = {'scores': scores, 'paired': paired, 'gates': gates, 'passed_development': all(gates.values()),
              'deployed': False, 'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'gates': gates, 'paired': paired}), flush=True)


if __name__ == '__main__':
    main()
