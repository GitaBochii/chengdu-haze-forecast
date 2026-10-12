"""Check whether temporal selection folds can learn forecast feature effects."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    search = base / 'aft_temporal_search_20260918'
    out = base / 'temporal_feature_support_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [search / 'protocol.json',
             base / 'gefs_2021_extension_v1/f024_training_rows.parquet',
             base / 'gefs_2021_extension_v1/inference_rows.parquet',
             Path(__file__).resolve()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol = json.loads(paths[0].read_bytes())
    data = pd.read_parquet(paths[1])
    infer = pd.read_parquet(paths[2])
    features = protocol['features']
    forecast = [n for n in features if n.startswith(('gefs_', 'forecast_'))]
    by_year = {}
    for year, frame in data.groupby(data.timestamp.dt.year):
        values = frame[features].to_numpy(dtype=float)
        counts = np.isfinite(values).sum(axis=0)
        # Scalar counting is an independent check of the vectorized support table.
        assert list(counts) == [sum(np.isfinite(float(v)) for v in frame[n]) for n in features]
        by_year[str(year)] = {'rows': len(frame), 'blocks': int(frame.continuity_block.nunique()),
                             'finite_counts': dict(zip(features, map(int, counts)))}
    folds = {}
    for year in protocol['years']:
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        valid = data.loc[data.timestamp.ge(cut) & data.timestamp.lt(end) & data.label_available_at.lt(end)]
        train = data.loc[data.timestamp.lt(cut) & data.label_available_at.lt(cut)
                         & ~data.continuity_block.isin(valid.continuity_block)]
        models = {}
        for i in range(len(protocol['configs'])):
            path = search / f'{year}_{i}.ubj'
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = xgb.Booster()
            model.load_model(path)
            splits = model.get_score(importance_type='weight')
            assert set(splits).issubset(features)
            models[str(i)] = {n: int(splits.get(n, 0)) for n in forecast}
        folds[str(year)] = {
            'training_rows': len(train), 'validation_rows': len(valid),
            'training_forecast_finite': train[forecast].notna().sum().to_dict(),
            'validation_forecast_finite': valid[forecast].notna().sum().to_dict(),
            'model_forecast_split_counts': models,
        }
    report = {'forecast_features': forecast, 'training_by_year': by_year, 'folds': folds,
              'full_training_forecast_complete_rows': int(data[forecast].notna().all(axis=1).sum()),
              'inference_rows': len(infer),
              'inference_forecast_complete_rows': int(infer[forecast].notna().all(axis=1).sum()),
              'input_sha256': hashes,
              'scope': 'Feature support and saved tree inspection only; no new accuracy evaluation.'}
    for path, digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    out.mkdir()
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('input_sha256', 'training_by_year')}, indent=2))


if __name__ == '__main__':
    main()
