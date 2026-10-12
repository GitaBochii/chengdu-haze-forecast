"""No-fit hourly support repair evaluated without touching frozen forecasts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.hourly_recovery_support import enforce_hourly_support
from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'hourly_support_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = {2023: base / 'direct_history_aft_2023_20260918/scored_predictions.parquet',
             2024: base / 'reusable_gefs_crossyear_v1/2024_expanded.parquet',
             2025: base / 'reusable_gefs_crossyear_v1/2025_expanded.parquet'}
    label_path = base / 'interval_business_labels_v1/origins.parquet'
    labels = pd.read_parquet(label_path).set_index('timestamp')
    dependencies = [*paths.values(), label_path, Path(__file__).resolve(),
                    ROOT / 'src/haze_forecast/hourly_recovery_support.py',
                    ROOT / 'src/haze_forecast/observed_business.py',
                    ROOT / 'src/haze_forecast/cohort_scorecard.py',
                    ROOT / 'src/haze_forecast/paired_interval_score.py']
    hashes = {str(p): digest(p) for p in dependencies}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes,
        'rule': 'For causally eligible hourly origins only, max(prediction,1 hour)',
        'no_training': True, 'no_upper_cap': True, 'deployed': False}, indent=2), encoding='utf-8')
    results = {}
    for year, path in paths.items():
        source = pd.read_parquet(path)
        frame = labels.loc[source.timestamp].reset_index()
        assert frame.eligible.all() and frame.aqi.gt(100).all()
        assert frame.lower_bound_hours.ge(1).all()
        original = source.forecast_aft.to_numpy()
        candidate = enforce_hourly_support(original, frame.eligible.to_numpy())
        assert (candidate >= original).all()
        frame['native'], frame['supported'] = original, candidate
        scores = {name: score_predictions(frame, frame[name].to_numpy()) for name in ('native', 'supported')}
        paired = paired_error_summary(frame, original, candidate)
        # For every T >= 1, repaired rows reduce absolute error by exactly 1-p.
        improvement = candidate - original
        np.testing.assert_allclose(paired['all_row_delta_lower'], -improvement.mean(), atol=1e-12)
        np.testing.assert_allclose(paired['all_row_delta_upper'], -improvement.mean(), atol=1e-12)
        block_delta = -pd.Series(improvement).groupby(frame.continuity_block).mean().mean()
        np.testing.assert_allclose(paired['all_block_delta_lower'], block_delta, atol=1e-12)
        np.testing.assert_allclose(paired['all_block_delta_upper'], block_delta, atol=1e-12)
        frame.to_parquet(out / f'{year}_predictions.parquet', index=False)
        changed = frame.loc[candidate != original, ['timestamp', 'native', 'supported', 'lower_bound_hours']]
        results[str(year)] = {'scores': scores, 'paired': paired,
                             'changed_rows': len(changed), 'changes': changed.astype({'timestamp': str}).to_dict('records')}
    for p, expected in hashes.items():
        assert digest(Path(p)) == expected, p
    result = {'years': results, 'deployed': False,
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
