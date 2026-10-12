"""Describe frozen model disagreement; never fit or select a routing policy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.paired_interval_score import paired_error_bounds, paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'regional_frozen_complete_2025_20260916_v2'
    out = base / 'frozen_model_disagreement_20260916'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((source / 'report.json').read_bytes())
    protocol = json.loads((source / 'protocol.json').read_bytes())
    hashes = dict(protocol['input_sha256'])
    for name, value in report['output_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Invalid artifact name')
        hashes[str(source / name)] = value
    for path in [Path(__file__).resolve(), source / 'report.json',
                 ROOT / 'src/haze_forecast/paired_interval_score.py']:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path, value in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != value:
            raise ValueError(f'Frozen input changed: {path}')
    frame = pd.read_parquet(source / 'capped_predictions.parquet')
    bounds = paired_error_bounds(frame, frame.matched_aft, frame.airformer)
    # Independent piecewise endpoints/kinks: verify every origin's paired bounds.
    lo, hi = frame.lower_bound_hours.to_numpy(), frame.upper_bound_hours.to_numpy()
    a, b = frame.matched_aft.to_numpy(), frame.airformer.to_numpy()
    truth = np.stack([lo, hi, np.clip(a, lo, hi), np.clip(b, lo, hi)])
    delta = abs(b - truth) - abs(a - truth)
    np.testing.assert_allclose(bounds.lower, delta.min(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(bounds.upper, delta.max(axis=0), rtol=0, atol=1e-12)
    masks = {'all': np.ones(len(frame), dtype=bool),
             'airformer_no_confirmed_recovery': frame.airformer.eq(68),
             'airformer_confirms_recovery': frame.airformer.lt(68),
             'airformer_later_than_aft': frame.airformer.gt(frame.matched_aft),
             'airformer_not_later_than_aft': frame.airformer.le(frame.matched_aft)}
    summaries = {}
    for name, mask in masks.items():
        selected = frame.loc[mask]
        paired = paired_error_summary(selected, selected.matched_aft, selected.airformer)
        summaries[name] = {
            'paired': paired,
            'certainly_better_origins': int(bounds.loc[mask, 'upper'].lt(0).sum()),
            'certainly_worse_origins': int(bounds.loc[mask, 'lower'].gt(0).sum()),
            'tie_or_uncertain_origins': int((bounds.loc[mask, 'lower'].le(0)
                                           & bounds.loc[mask, 'upper'].ge(0)).sum()),
        }
    for path, value in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != value:
            raise ValueError(f'Input changed during audit: {path}')
    result = {'input_sha256': hashes, 'summaries': summaries,
              'no_recovery_but_actual_within_24': int((frame.airformer.eq(68)
                                                     & frame.upper_bound_hours.le(24)).sum()),
              'training_performed': False, 'routing_policy_selected': False,
              'deployment_authorized': False,
              'limitations': ['Exploratory reused 2025 data; no validation of a learned gate',
                              'Positive paired delta means AirFormer worse than matched AFT',
                              'Capped target only; 68 is not exact uncapped duration',
                              'Groups are origin-level; the same process can appear in multiple groups',
                              'Do not select a routing rule from favorable subgroups',
                              'Forecast disagreement alone is not calibrated confidence']}
    out.mkdir()
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: {'rows': v['paired']['rows'],
                         'better': v['certainly_better_origins'],
                         'worse': v['certainly_worse_origins'],
                         'row_delta_lower': v['paired']['all_row_delta_lower'],
                         'row_delta_upper': v['paired']['all_row_delta_upper']}
                      for k, v in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
