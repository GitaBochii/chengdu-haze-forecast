"""Two-step real-input integration test, never a generalization evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from haze_forecast.interval_recovery_labels import interval_recovery_bounds
from haze_forecast.regional_air_experiment import prepare_panel
from haze_forecast.regional_air_scoring import active_node_mask
from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from experiments.prithvi_wxc_probe_20260924.fusion_head import SpatialRecoveryFusion

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISION = 'e2f62b3e19890f1a5e225bb123acb4b33df85ef8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(folder):
    folder = Path(folder)
    required = ['pretrained_spatial_embedding.pt', 'pretrained_encoder_report.json',
                'official_batch.pt', 'paired_sample_report.json']
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        return {'ready': False, 'reason': 'missing_artifacts', 'missing': missing}
    report = json.loads((folder / 'pretrained_encoder_report.json').read_text('utf-8'))
    if (report.get('status') != 'real_pretrained_encoder_passed'
            or report.get('pretrained_inference_done') is not True
            or report.get('revision') != REVISION
            or report.get('load', {}).get('strict_load') is not True
            or report.get('future_truth_used') is not False):
        return {'ready': False, 'reason': 'unverified_pretrained_encoder'}
    for filename, key in [('pretrained_spatial_embedding.pt', 'embedding_file_sha256'),
                          ('official_batch.pt', 'input_file_sha256')]:
        if sha(folder / filename) != report.get(key):
            return {'ready': False, 'reason': f'hash_mismatch:{filename}'}
    return {'ready': True, 'reason': 'verified_encoder_artifacts'}


def exact_survival_nll(logits, hours):
    if (logits.ndim != 2 or logits.shape[1] != 67 or logits.shape[0] < 1
            or isinstance(hours, bool) or not float(hours).is_integer() or not 1 <= hours <= 67):
        raise ValueError('requires [batch,67] logits and exact integer recovery in 1..67')
    index = int(hours) - 1
    return (F.softplus(logits[:, :index]).sum(dim=1) + F.softplus(-logits[:, index])).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    check = preflight(HERE)
    if not check['ready'] or args.preflight_only:
        print(json.dumps(check, indent=2))
        return
    paired = json.loads((HERE / 'paired_sample_report.json').read_text('utf-8'))
    stored = torch.load(HERE / 'pretrained_spatial_embedding.pt', weights_only=True, map_location='cpu')
    embedding = stored['embedding'].detach().float().clone()
    if (stored.get('revision') != REVISION or stored.get('origin_utc') != paired['origin_utc']
            or stored.get('weather_times_utc') != paired['weather_times_utc']
            or embedding.shape != (1, 2560) or not torch.isfinite(embedding).all()
            or embedding.std() <= 0):
        raise ValueError('embedding identity, dimensions or values invalid')
    origin = pd.Timestamp(paired['origin_utc']).tz_convert('Asia/Shanghai')
    if origin != pd.Timestamp('2020-01-01T14:00:00+08:00'):
        raise ValueError('unexpected fixed sample origin')
    data = ROOT / 'data/processed/regional_air_models_full_history'
    paths = [data / name for name in ('station_hourly.parquet', 'weather_proxy.parquet', 'city_hourly.parquet')]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    stations, weather, city = [pd.read_parquet(p) for p in paths]
    panel = prepare_panel(stations, weather, pd.Timestamp('2020-01-01', tz='Asia/Shanghai'),
                          training_start=pd.Timestamp('2014-01-01', tz='Asia/Shanghai'), ozone_column='o3_8h')
    position = panel['timestamps'].get_indexer([origin])[0]
    if position < 23:
        raise ValueError('missing real history')
    x = torch.from_numpy(panel['inputs'][position - 23:position + 1]).unsqueeze(0)
    active = torch.from_numpy(active_node_mask(panel['targets'], np.array([position]), lookback=24))
    target = interval_recovery_bounds(city, pd.DatetimeIndex([origin])).iloc[0]
    if target.reason != 'exact' or target.exact_hours != 46:
        raise ValueError('fixed real sample supervision mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    scaler = panel['scaler']
    baseline = RecoveryPM25GNN(x.shape[-1], panel['coords'],
                              torch.tensor(scaler.mean[-2:], dtype=torch.float32),
                              torch.tensor(scaler.scale[-2:], dtype=torch.float32))
    model = SpatialRecoveryFusion(baseline, 2560, freeze_baseline=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    present = torch.ones(1, dtype=torch.bool)
    losses, gradients = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(x, active, embedding, present)
        loss = exact_survival_nll(logits, 46)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        norm = float(model.residual[-1].weight.grad.abs().sum())
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('fusion residual received no finite gradient')
        gradients.append(norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        _, logits = model(x, active, embedding, present)
        _, changed = model(x, active, -embedding, present)
        _, missing = model(x, active, torch.full_like(embedding, float('nan')), ~present)
        _, base = baseline(x, active)
    if torch.equal(logits, changed):
        raise ValueError('embedding replacement had no effect')
    torch.testing.assert_close(missing, base, rtol=0, atol=0)
    report = {'status': 'real_multimodal_two_step_smoke_passed', 'origin': origin.isoformat(),
              'preflight': check, 'input_shape': list(x.shape), 'embedding_shape': list(embedding.shape),
              'active_shape': list(active.shape), 'hazard_shape': list(logits.shape),
              'exact_target_hours': 46, 'label_available_at': target.label_available_at.isoformat(),
              'training_steps': 2, 'training_nll': losses, 'fusion_output_weight_gradient_l1': gradients,
              'embedding_replacement_max_change': float((logits - changed).abs().max()),
              'missing_embedding_exact_baseline_fallback': True,
              'pollutant_initialization': 'new_random_seed42_no_old_checkpoint',
              'scaler_fit_period': '2014-01-01 inclusive to 2020-01-01 exclusive',
              'source_sha256': source_hashes,
              'embedding_sha256': sha(HERE / 'pretrained_spatial_embedding.pt'),
              'encoder_report_sha256': sha(HERE / 'pretrained_encoder_report.json'),
              'feature_names': panel['feature_names'], 'nodes': panel['nodes'],
              'limitations': ['one sample trained and inspected; no generalization MAE',
                              'weather proxy and MERRA-2 are offline reanalysis',
                              'no deployment change; no fitted model checkpoint saved',
                              '2020 is within foundation-model validation-year range']}
    (HERE / 'real_fusion_smoke_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
