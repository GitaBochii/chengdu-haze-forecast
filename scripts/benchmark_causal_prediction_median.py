"""Fixed causal-median development comparison without fitting or deployment."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.causal_prediction_median import causal_prediction_median
from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'station_dispersion_2023_20260918'
    out = base / 'causal_prediction_median_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    previous = json.loads((source / 'report.json').read_bytes())
    for name, expected in previous['output_sha256'].items():
        assert digest(source / name) == expected, name
    paths = [Path(__file__), ROOT / 'causal_median_plan_20260918.md',
             ROOT / 'src/haze_forecast/causal_prediction_median.py',
             ROOT / 'src/haze_forecast/cohort_scorecard.py',
             ROOT / 'src/haze_forecast/paired_interval_score.py',
             source / 'all_predictions.parquet', source / 'scored_predictions.parquet']
    hashes = {str(p): digest(p) for p in paths}
    all_rows = pd.read_parquet(paths[-2]).sort_values('timestamp')
    frame = pd.read_parquet(paths[-1]).sort_values('timestamp')
    out.mkdir()
    protocol = {'input_sha256': hashes, 'window': 6, 'aggregation': 'median',
                'target': 'full remaining hours', 'fitted_parameters': 0,
                'gap_rule': 'reset on non-hourly interval', 'deployed': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    all_rows['causal_median'] = causal_prediction_median(all_rows.timestamp, all_rows.native)
    indexed = all_rows.set_index('timestamp')
    np.testing.assert_array_equal(frame.native, indexed.loc[frame.timestamp, 'native'])
    frame['causal_median'] = indexed.loc[frame.timestamp, 'causal_median'].to_numpy()
    scores = {name: score_predictions(frame, frame[name].to_numpy()) for name in ('native', 'causal_median')}
    paired = paired_error_summary(frame, frame.native, frame.causal_median)
    candidate, reference = scores['causal_median'], scores['native']
    passed = (all(candidate[k] < reference[k] for k in ('exact_row_mae', 'exact_block_mae'))
              and candidate['tail24_block_mae'] <= reference['tail24_block_mae']
              and paired['all_row_delta_lower'] <= 0 and paired['all_block_delta_lower'] <= 0)
    all_rows.to_parquet(out / 'all_predictions.parquet', index=False)
    frame.to_parquet(out / 'scored_predictions.parquet', index=False)
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    report = {'scores': scores, 'paired': paired, 'passed_development': passed, 'deployed': False,
              'output_sha256': {p.name: digest(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
