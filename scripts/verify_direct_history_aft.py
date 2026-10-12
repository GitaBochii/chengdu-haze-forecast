"""Independent input reconstruction and saved-checkpoint scoring checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from haze_forecast.direct_history_aft import DirectHistoryAFT

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / 'artifacts/task01_recovery_head/direct_history_aft_2023_20260918'
    r = json.loads((p / 'report.json').read_bytes())
    protocol = json.loads((p / 'protocol.json').read_bytes())
    original = json.loads((ROOT / 'artifacts/task01_recovery_head/grouped_forecast_aft_2023_20260916/protocol.json').read_bytes())
    for path, value in original['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == value, path
    train = pd.read_parquet(p / 'training.parquet')
    d = pd.read_parquet(p / 'scored_predictions.parquet')
    allp = pd.read_parquet(p / 'all_predictions.parquet')
    raw = np.load(p / 'raw_inputs.npz')
    source = pd.read_parquet(ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet').set_index('timestamp')
    times = pd.to_datetime(raw['timestamps'], utc=True).tz_convert('Asia/Shanghai')
    # Build each hour separately rather than slicing the production positional window.
    rebuilt = np.stack([source.loc[times - pd.Timedelta(hours=lag), protocol['history_features']].to_numpy()
                        for lag in range(23, -1, -1)], axis=1)
    np.testing.assert_array_equal(rebuilt, raw['history'])
    torch.set_num_threads(2)
    for arm in ('snapshot', 'history'):
        selection = json.loads((p / f'{arm}_selection.json').read_bytes())
        best = min(selection['trace'], key=lambda row: row['selection_block_mae'])
        assert best['epoch'] == selection['selected_epoch'] == r['training'][arm]['selected_epochs']
        assert len(r['training'][arm]['refit']) == best['epoch']
        inputs = np.load(p / f'{arm}_inputs.npz')
        history = raw['history'] if arm == 'history' else np.repeat(raw['history'][:, -1:], 24, axis=1)
        for name, values in [('history', history), ('snapshot', raw['snapshot'])]:
            fitted = values[:len(train)].reshape(-1, values.shape[-1])
            means, scales, support = [], [], []
            for column in fitted.T:
                observed = column[np.isfinite(column)]
                mean = observed.mean() if len(observed) else 0.
                scale = observed.std() if len(observed) else 1.
                means.append(mean)
                scales.append(scale if scale > 1e-6 else 1.)
                support.append(bool(len(observed)))
            np.testing.assert_allclose(means, inputs[name + '_mean'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(scales, inputs[name + '_scale'], rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(support, inputs[name + '_supported'])
            numerical = (values - means) / scales
            numerical = np.where(np.isfinite(values) & np.array(support), numerical, 0.)
            expected = np.concatenate([numerical, np.isnan(values).astype(float)], axis=-1).astype(np.float32)
            np.testing.assert_allclose(expected, inputs[name], rtol=1e-6, atol=1e-6)
        model = DirectHistoryAFT(24, 208).cuda().eval()
        model.load_state_dict(torch.load(p / f'{arm}.pt', weights_only=True, map_location='cuda'))
        outputs = []
        with torch.inference_mode():
            for offset in range(len(train), len(times), 128):
                h = torch.from_numpy(inputs['history'][offset:offset+128]).cuda()
                x = torch.from_numpy(inputs['snapshot'][offset:offset+128]).cuda()
                outputs.append(model(h, x).double().exp().cpu().numpy())
        np.testing.assert_array_equal(np.concatenate(outputs), allp[arm])
    for arm, column in [('native', 'forecast_aft'), ('snapshot', 'snapshot'), ('history', 'history')]:
        exact = d[d.exact_hours.notna()].copy()
        exact['error'] = (exact[column] - exact.exact_hours).abs()
        values = [exact.error.mean(), exact.groupby('continuity_block').error.mean().mean(),
                  exact[exact.exact_hours >= 24].groupby('continuity_block').error.mean().mean()]
        for key, value in zip(['exact_row_mae', 'exact_block_mae', 'tail24_block_mae'], values, strict=True):
            np.testing.assert_allclose(value, r['scores'][arm][key], rtol=0, atol=1e-10)
    for arm, column in [('native', 'forecast_aft'), ('snapshot', 'snapshot')]:
        rows = []
        for row in d.itertuples():
            a, b, lo, hi = getattr(row, column), row.history, row.lower_bound_hours, row.upper_bound_hours
            points = [lo, hi, np.clip(a, lo, hi), np.clip(b, lo, hi)]
            diff = [abs(b-y) - abs(a-y) for y in points]
            rows.append([min(diff), max(diff)])
        bounds = pd.DataFrame(rows, columns=['lower', 'upper'])
        bounds['block'] = d.continuity_block.to_numpy()
        for side in ('lower', 'upper'):
            np.testing.assert_allclose(bounds[side].mean(), r['paired'][arm]['all_row_delta_' + side], atol=1e-10)
            np.testing.assert_allclose(bounds.groupby('block')[side].mean().mean(),
                                       r['paired'][arm]['all_block_delta_' + side], atol=1e-10)
    for path, value in protocol['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == value, path
    for name, value in r['output_sha256'].items():
        assert hashlib.sha256((p / name).read_bytes()).hexdigest() == value, name
    print('7289 raw histories, both scaled inputs, 1038 predictions, 17 metrics and hashes passed')


if __name__ == '__main__':
    main()
