"""Fixed historical temporal classification diagnostic, not a duration forecast."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'duration_separation_20260918_v2'
    if out.exists():
        raise FileExistsError(out)
    paths = [base / 'gefs_2021_extension_v1/f024_training_rows.parquet',
             base / 'grouped_forecast_aft_2023_20260916/protocol.json',
             ROOT / 'duration_separation_plan_20260918.md', Path(__file__).resolve()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    d = pd.read_parquet(paths[0])
    names = json.loads(paths[1].read_bytes())['features']
    d['target'] = np.where(d.lower_bound_hours >= 24, 1.,
                           np.where(d.upper_bound_hours < 24, 0., np.nan))
    params = dict(loss='log_loss', learning_rate=.05, max_iter=100, max_leaf_nodes=8,
                  max_depth=3, min_samples_leaf=30, l2_regularization=3,
                  early_stopping=False, random_state=42)
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(
        {'input_sha256': hashes, 'features': names, 'params': params}, indent=2), encoding='utf-8')
    reports = {}
    for year in (2020, 2021, 2022):
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        v = d.loc[d.timestamp.ge(cut) & d.timestamp.lt(end) & d.label_available_at.lt(end)].copy()
        t = d.loc[d.timestamp.lt(cut) & d.label_available_at.lt(cut)
                  & ~d.continuity_block.isin(v.continuity_block) & d.target.notna()].copy()
        assert not set(t.continuity_block) & set(v.continuity_block)
        assert t.target.nunique() == 2 and v.timestamp.is_unique
        unsupported = t[names].isna().all()
        supported_names = unsupported.index[~unsupported].tolist()
        x, xv = t[supported_names].copy(), v[supported_names].copy()
        model = HistGradientBoostingClassifier(**params)
        with threadpool_limits(limits=2):
            model.fit(x, t.target)
            p = model.predict_proba(xv)[:, 1]
        q = float(t.target.mean())
        v['probability'], v['constant'] = p, q
        known = v.target.notna()
        z = v.loc[known].copy()
        z['model_brier'] = (z.probability - z.target)**2
        z['constant_brier'] = (q - z.target)**2
        # Evaluate paired loss at both possible classes for ambiguous labels.
        diff0, diff1 = p**2-q**2, (p-1)**2-(q-1)**2
        actual = (p-v.target.to_numpy())**2-(q-v.target.to_numpy())**2
        v['delta_lower'] = np.where(known, actual, np.minimum(diff0, diff1))
        v['delta_upper'] = np.where(known, actual, np.maximum(diff0, diff1))
        row = z[['model_brier', 'constant_brier']].mean()
        block = z.groupby('continuity_block')[['model_brier', 'constant_brier']].mean().mean()
        bound = v.groupby('continuity_block')[['delta_lower', 'delta_upper']].mean().mean()
        result = {'train_rows': len(t), 'validation_rows': len(v), 'known_rows': len(z),
                  'known_fraction': float(known.mean()), 'training_prevalence': q,
                  'row_brier': row.to_dict(), 'block_brier': block.to_dict(),
                  'auc': float(roc_auc_score(z.target, z.probability)),
                  'average_precision': float(average_precision_score(z.target, z.probability)),
                  'validation_prevalence': float(z.target.mean()),
                  'all_row_delta': v[['delta_lower', 'delta_upper']].mean().to_dict(),
                  'all_block_delta': bound.to_dict(),
                  'confusion': {f'true{a}_pred{b}': int(((z.target == a) &
                     ((z.probability >= .5).astype(int) == b)).sum()) for a in (0, 1) for b in (0, 1)}}
        result['pass'] = bool(row.model_brier < row.constant_brier
                              and block.model_brier < block.constant_brier
                              and v.delta_upper.mean() <= 0 and bound.delta_upper <= 0)
        reports[str(year)] = result
        t[['timestamp', 'continuity_block', 'label_available_at', 'target']].to_parquet(out / f'{year}_train.parquet', index=False)
        v.to_parquet(out / f'{year}_validation.parquet', index=False)
        joblib.dump({'model': model, 'unsupported': unsupported}, out / f'{year}.joblib')
    for p, h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    result = {'years': reports, 'pass': all(r['pass'] for r in reports.values()),
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
