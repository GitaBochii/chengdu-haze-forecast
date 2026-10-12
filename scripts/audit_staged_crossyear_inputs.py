"""Input-only readiness check; never load model predictions or compute accuracy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'staged_crossyear_input_audit_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [b / 'interval_business_labels_v1/origins.parquet',
             b / 'staged_forecast_aft_2022_20260918/protocol.json', Path(__file__).resolve()]
    canonical = pd.read_parquet(paths[0])
    config = json.loads(paths[1].read_bytes())
    names, weather = config['observed'] + config['weather'], config['weather']
    results = {}
    for year in (2023, 2024, 2025):
        cut = pd.Timestamp(f'{year-1}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        finish = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        if year == 2023:
            tp = b / 'gefs_2021_extension_v1/training_rows.parquet'
            vp = b / 'gefs_2021_extension_v1/inference_rows.parquet'
            v = pd.read_parquet(vp)
        else:
            tp = b / f'original_gefs_without_pbl_crossyear_v1/{year}_training_rows.parquet'
            vp = b / f'reusable_gefs_crossyear_v1/{year}_features.parquet'
            identity = b / f'reusable_gefs_crossyear_v1/{year}_predictions.parquet'
            paths.append(identity)
            v = pd.read_parquet(vp)
            # Read only the timestamp column, not any stored predictions.
            timestamps = pd.read_parquet(identity, columns=['timestamp'])
            assert len(v) == len(timestamps)
            v['timestamp'] = timestamps.timestamp.to_numpy()
        paths.extend([tp, vp])
        t = pd.read_parquet(tp)
        assert t.timestamp.is_unique and v.timestamp.is_unique
        assert t.timestamp.lt(end).all() and t.label_available_at.lt(end).all()
        labels = canonical.set_index('timestamp').loc[t.timestamp]
        for n in ('lower_bound_hours', 'upper_bound_hours', 'exact_hours', 'continuity_block'):
            np.testing.assert_array_equal(t[n], labels[n])
        cohort = canonical.loc[canonical.timestamp.dt.year.eq(year)
                               & canonical.label_available_at.lt(finish)].copy()
        assert set(cohort.timestamp).issubset(v.timestamp)
        assert not set(t.continuity_block) & set(cohort.continuity_block)
        recent = t.loc[t.timestamp.ge(cut)]
        h = recent.loc[recent[weather].notna().all(axis=1)]
        backbone = t.loc[t.timestamp.lt(cut) & t.label_available_at.lt(cut)
                         & ~t.continuity_block.isin(h.continuity_block)]
        assert not set(backbone.continuity_block) & set(h.continuity_block)
        assert backbone[names].shape[1] == len(names) == 104
        evaluable = v.set_index('timestamp').loc[cohort.timestamp]
        assert np.isfinite(evaluable[names].to_numpy(dtype=float)[evaluable[names].notna().to_numpy()]).all()
        results[str(year)] = {
            'full_training_rows': len(t), 'backbone_rows': len(backbone),
            'recent_year_rows': len(recent), 'head_supported_rows': len(h),
            'head_blocks': int(h.continuity_block.nunique()),
            'recent_rows_excluded_from_head': len(recent) - len(h),
            'prior_rows_excluded_by_maturity_or_block': int(t.timestamp.lt(cut).sum()) - len(backbone),
            'inference_rows': len(v), 'scored_mature_rows': len(cohort),
            'score_blocks': int(cohort.continuity_block.nunique()),
            'scored_forecast_complete': int(evaluable[weather].notna().all(axis=1).sum()),
            'scored_fallback_rows': int((~evaluable[weather].notna().all(axis=1)).sum()),
            'training_2021_forecast_complete': int((t.timestamp.dt.year.eq(2021) & t[weather].notna().all(axis=1)).sum()),
        }
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out.mkdir()
    report = {'years': results, 'input_sha256': hashes,
              'scope': 'Input-only audit. No training, candidate forecast or score loaded. Not crossyear authorization.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
