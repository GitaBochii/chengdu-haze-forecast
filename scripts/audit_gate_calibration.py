"""Forward-only calibration preflight on frozen historical OOF probabilities."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'duration_separation_20260918_v2'
    out = base / 'gate_calibration_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / f'{year}_validation.parquet' for year in (2020, 2021, 2022)]
    paths += [ROOT / 'gate_calibration_plan_20260918.md', Path(__file__).resolve()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    frames = {year: pd.read_parquet(paths[i]) for i, year in enumerate((2020, 2021, 2022))}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes}, indent=2), encoding='utf-8')
    results = {}
    for year in (2021, 2022):
        t = pd.concat([frames[y] for y in range(2020, year)], ignore_index=True)
        t = t.loc[t.target.notna()].copy()
        v = frames[year].copy()
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        assert t.label_available_at.lt(cut).all() and t.timestamp.lt(cut).all()
        assert not set(t.continuity_block) & set(v.continuity_block)
        assert t.probability.between(0, 1, inclusive='neither').all()
        assert v.probability.between(0, 1, inclusive='neither').all()
        model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, tol=1e-8)
        model.fit(logit(t.probability.to_numpy()).reshape(-1, 1), t.target)
        assert model.n_iter_[0] < 1000
        p = v.probability.to_numpy()
        q = model.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        known = v.target.notna().to_numpy()
        target = v.target.to_numpy()
        a, b = q*q-p*p, (q-1)**2-(p-1)**2
        diff = (q-target)**2-(p-target)**2
        v['calibrated'] = q
        v['calibration_delta_lower'] = np.where(known, diff, np.minimum(a, b))
        v['calibration_delta_upper'] = np.where(known, diff, np.maximum(a, b))
        v['original_brier'] = (p-target)**2
        v['calibrated_brier'] = (q-target)**2
        fields = ['original_brier', 'calibrated_brier']
        row = v.loc[known, fields].mean()
        block = v.loc[known].groupby('continuity_block')[fields].mean().mean()
        delta_fields = ['calibration_delta_lower', 'calibration_delta_upper']
        row_delta = v[delta_fields].mean()
        block_delta = v.groupby('continuity_block')[delta_fields].mean().mean()
        passed = bool(model.coef_[0, 0] > 0 and row.calibrated_brier < row.original_brier
                      and block.calibrated_brier < block.original_brier
                      and row_delta.iloc[1] <= 0 and block_delta.iloc[1] <= 0)
        results[str(year)] = {'training_rows': len(t), 'validation_rows': len(v),
            'slope': float(model.coef_[0, 0]), 'intercept': float(model.intercept_[0]),
            'row_brier': row.to_dict(), 'block_brier': block.to_dict(),
            'row_delta': row_delta.to_dict(), 'block_delta': block_delta.to_dict(), 'pass': passed}
        joblib.dump(model, out / f'{year}.joblib')
        t[['timestamp', 'target', 'probability', 'label_available_at', 'continuity_block']].to_parquet(
            out / f'{year}_training.parquet', index=False)
        v.to_parquet(out / f'{year}_validation.parquet', index=False)
    for p, h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    report = {'years': results, 'pass': all(r['pass'] for r in results.values()),
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
