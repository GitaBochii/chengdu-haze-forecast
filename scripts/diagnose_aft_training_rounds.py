"""Time-split iteration diagnostic: select on 2022, then inspect fixed 2023."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'interval_history_counts_v1'
    out = base / 'aft_rounds_temporal_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [base / 'interval_business_labels_v1/origins.parquet',
             base / 'observed_business_dataset_v1/features.parquet',
             base / 'trajectory_reset_audit_v1/origins.parquet', dev / 'protocol.json',
             dev / 'model.joblib', dev / 'expanded_predictions.parquet', Path(__file__),
             ROOT / 'src/haze_forecast/observed_aft.py', ROOT / 'src/haze_forecast/training.py',
             ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    origins = pd.read_parquet(paths[0])
    full = pd.read_parquet(paths[1])
    history = pd.read_parquet(paths[2]).set_index('timestamp')
    names = json.loads((dev / 'protocol.json').read_bytes())['features']
    origins[names[87:]] = history.loc[origins.timestamp, names[87:]].to_numpy()
    start, end = [pd.Timestamp(f'{y}-01-01', tz='Asia/Shanghai') for y in (2022, 2023)]
    def crossing(cut):
        return set(full.loc[full.timestamp.lt(cut), 'continuity_block']) & set(full.loc[full.timestamp.ge(cut), 'continuity_block'])
    cross_start, cross_end = crossing(start), crossing(end)
    train = origins.loc[origins.timestamp.lt(start) & origins.label_available_at.lt(start)
                        & origins.informative & ~origins.continuity_block.isin(cross_start)]
    val = origins.loc[origins.timestamp.ge(start) & origins.timestamp.lt(end)].copy()
    val['score_allowed'] = val.label_available_at.lt(end) & ~val.continuity_block.isin(cross_start | cross_end)
    assert not set(train.continuity_block) & set(val.continuity_block)
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'round_grid': [50, 100, 200, 300, 500],
                'selection': '2022 only; minimum block MAE among rounds with row and tail no worse than 500; tie fewer rounds',
                'train_cutoff': str(start), 'score_end': str(end), 'seed': 42,
                'limitations': ['2022/2023 historical diagnostics, not untouched validation',
                                '2024/2025 not read or used for selection', 'same 92 features and normal AFT loss']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train.to_parquet(out / 'training_rows.parquet', index=False)
    model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'pre2022.joblib')
    matrix = xgb.DMatrix(val[names].astype(float), missing=np.nan)
    scores = {}
    for rounds in protocol['round_grid']:
        pred = np.exp(model.booster.predict(matrix, output_margin=True, iteration_range=(0, rounds)))
        assert np.isfinite(pred).all()
        val[f'rounds_{rounds}'] = pred
        scores[str(rounds)] = metrics(val.loc[val.score_allowed], pred[val.score_allowed])
    ref = scores['500']
    eligible = [n for n in protocol['round_grid'] if scores[str(n)]['row_mae'] <= ref['row_mae']
                and scores[str(n)]['tail_24h_block_mae'] <= ref['tail_24h_block_mae']]
    selected = min(eligible, key=lambda n: (scores[str(n)]['block_mae'], n))
    (out / 'selection.json').write_text(json.dumps({'rounds': selected, 'scores_2022': scores}, indent=2), encoding='utf-8')
    next_val = pd.read_parquet(dev / 'expanded_predictions.parquet')
    next_val[names[87:]] = history.loc[next_val.timestamp, names[87:]].to_numpy()
    frozen = joblib.load(dev / 'model.joblib')
    pred = np.exp(frozen.booster.predict(xgb.DMatrix(next_val[names].astype(float), missing=np.nan),
                                       output_margin=True, iteration_range=(0, selected)))
    next_val['selected_round_prediction'] = pred
    allowed = next_val.score_allowed
    next_scores = {'selected': metrics(next_val.loc[allowed], pred[allowed]),
                   'control_500': metrics(next_val.loc[allowed], next_val.loc[allowed, 'history_counts_aft'].to_numpy())}
    val.to_parquet(out / '2022_predictions.parquet', index=False)
    next_val.to_parquet(out / '2023_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'selected_rounds': selected, 'scores_2022': scores, 'scores_2023': next_scores,
              'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
