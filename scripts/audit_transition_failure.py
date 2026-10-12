"""Observed-path probability diagnostics; never used as inference inputs."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'run_length_transition_2022_v2'
    out = base / 'transition_failure_audit_v1'
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((source / 'protocol.json').read_bytes())
    report = json.loads((source / 'report.json').read_bytes())
    for p, digest in protocol['input_sha256'].items():
        assert sha(Path(p)) == digest
    for name, digest in report['output_sha256'].items():
        assert sha(source / name) == digest
    paths = [source / name for name in report['output_sha256']]
    paths += [source / 'report.json', Path(__file__)]
    before = dict(protocol['input_sha256'])
    before.update({str(p): sha(p) for p in paths})
    train = pd.read_parquet(base / 'gefs_temporal_calibration_data_v1/training_rows.parquet')
    val = pd.read_parquet(source / 'predictions.parquet')
    transitions = pd.read_parquet(source / 'transitions.parquet')
    grid = pd.read_parquet(base / 'observed_business_dataset_v1/features.parquet').set_index('timestamp')
    # Construct evaluation states from observed outcomes, only for diagnostics.
    observed = []
    for owner, row in enumerate(val.itertuples()):
        begin = grid.index.get_loc(row.timestamp) + 1
        values = grid.aqi.iloc[begin:begin + int(row.upper_bound_hours) + 5].to_numpy()
        state = 0
        for hour, value in enumerate(values, start=1):
            if not np.isfinite(value):
                break
            target = int(value <= 100)
            observed.append((owner, hour, state, target))
            state = state + 1 if target else 0
            if state == 6:
                break
    evaluation = pd.DataFrame(observed, columns=transitions.columns)
    model = lgb.Booster(model_str=(source / 'model.txt').read_text(encoding='utf-8'))
    summaries, tables = [], {}
    for label, origins, table in [('train', train, transitions), ('validation', val, evaluation)]:
        x = origins[protocol['features']].iloc[table.owner.to_numpy()].reset_index(drop=True)
        x['horizon_hours'] = table.horizon_hours
        x['clean_run'] = table.clean_run
        table = table.copy()
        table['probability'] = model.predict(x, num_threads=1)
        table['block'] = origins.continuity_block.iloc[table.owner.to_numpy()].to_numpy()
        table['brier'] = (table.probability - table.target) ** 2
        for state, subset in table.groupby('clean_run'):
            summaries.append({'split': label, 'state': int(state), 'transitions': len(subset),
                              'blocks': int(subset.block.nunique()),
                              'observed_clean_rate': float(subset.target.mean()),
                              'predicted_clean_rate': float(subset.probability.mean()),
                              'brier': float(subset.brier.mean())})
        tables[label] = table
    block_counts = tables['train'].groupby('block').size().sort_values(ascending=False)
    prediction = val[['timestamp', 'continuity_block', 'exact_hours', 'forecast_aft',
                      'transition_prediction']].copy()
    prediction['aft_error'] = prediction.forecast_aft - prediction.exact_hours
    prediction['candidate_error'] = prediction.transition_prediction - prediction.exact_hours
    prediction['aft_absolute_error'] = prediction.aft_error.abs()
    prediction['candidate_absolute_error'] = prediction.candidate_error.abs()
    blocks = prediction.groupby('continuity_block').agg(
        rows=('timestamp', 'size'), aft_mae=('aft_absolute_error', 'mean'),
        candidate_mae=('candidate_absolute_error', 'mean'),
        aft_bias=('aft_error', 'mean'), candidate_bias=('candidate_error', 'mean'))
    groups = {}
    for label, mask in [('less_than_24h', prediction.exact_hours.lt(24)),
                        ('at_least_24h', prediction.exact_hours.ge(24))]:
        rows = prediction.loc[mask]
        groups[label] = {'rows': len(rows), 'aft_mae': float(rows.aft_absolute_error.mean()),
                         'candidate_mae': float(rows.candidate_absolute_error.mean()),
                         'aft_bias': float(rows.aft_error.mean()),
                         'candidate_bias': float(rows.candidate_error.mean())}
    out.mkdir()
    for label, table in tables.items():
        table.to_parquet(out / f'{label}_probabilities.parquet', index=False)
    pd.DataFrame(summaries).to_csv(out / 'state_summary.csv', index=False)
    blocks.to_csv(out / 'blocks.csv')
    block_counts.rename('transitions').to_csv(out / 'training_block_counts.csv')
    result = {'input_sha256': before, 'training_top10_block_transition_share': float(block_counts.head(10).sum() / block_counts.sum()),
              'validation_observed_transition_rows': len(evaluation),
              'validation_origins_without_observed_transition': len(val) - int(evaluation.owner.nunique()),
              'state_summary': summaries, 'duration_groups': groups,
              'better_blocks': int((blocks.candidate_mae < blocks.aft_mae).sum()),
              'worse_blocks': int((blocks.candidate_mae > blocks.aft_mae).sum()),
              'limitations': ['Observed-path diagnostics condition on actual future state; not forecast inference',
                              'Training probabilities are in-sample, not independent calibration evidence',
                              'Repeated transitions and overlapping origins are not independent observations',
                              'State averages do not prove a global calibration correction is valid',
                              'No new fit, deployment or later-year scoring'],
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    assert before == {str(p): sha(Path(p)) for p in before}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_sha256', 'output_sha256')}, indent=2))


if __name__ == '__main__':
    main()
