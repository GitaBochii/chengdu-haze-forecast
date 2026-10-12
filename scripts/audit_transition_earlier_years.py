"""Fixed 2020/2021 out-of-time state-probability diagnostic, no calibration."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'run_length_transition_2022_v2'
    origin_path = base / 'gefs_temporal_calibration_data_v1/training_rows.parquet'
    out = base / 'transition_earlier_years_v1'
    if out.exists():
        raise FileExistsError(out)
    old = json.loads((source / 'protocol.json').read_bytes())
    old_report = json.loads((source / 'report.json').read_bytes())
    for name, digest in old_report['output_sha256'].items():
        assert sha(source / name) == digest
    for p, digest in old['input_sha256'].items():
        assert sha(Path(p)) == digest
    rows = pd.read_parquet(origin_path)
    transitions = pd.read_parquet(source / 'transitions.parquet')
    assert rows.timestamp.lt(pd.Timestamp('2022-01-01', tz='Asia/Shanghai')).all()
    grid_path = base / 'observed_business_dataset_v1/features.parquet'
    grid = pd.read_parquet(grid_path)
    paths = [origin_path, source / 'transitions.parquet', source / 'protocol.json', source / 'report.json',
              grid_path, Path(__file__), ROOT / 'src/haze_forecast/app.py']
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir() if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    protocol = {'input_sha256': before, 'years': [2020, 2021], 'features': old['features'],
                'params': old['params'], 'rounds': old['rounds'],
                'purpose': 'Check whether positive state-5 probability bias repeats before 2022',
                'selection': 'No calibration, parameter search, or model promotion',
                'limitations': ['Observed-path probability evaluation is not recovery-time accuracy',
                                'Different annual weather support; no historical forecast evidence invented',
                                'Rows inherited from pre-2022 pool; not a newly rebuilt complete annual cohort',
                                'Repeated transition rows are not independent observations'],
                'deployment_authorized': False}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    summaries = {}
    for year in protocol['years']:
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year + 1}-01-01', tz='Asia/Shanghai')
        cross_start = set(grid.loc[grid.timestamp.lt(cut), 'continuity_block']) & set(
            grid.loc[grid.timestamp.ge(cut), 'continuity_block'])
        cross_end = set(grid.loc[grid.timestamp.lt(end), 'continuity_block']) & set(
            grid.loc[grid.timestamp.ge(end), 'continuity_block'])
        train_ids = rows.index[rows.timestamp.lt(cut) & rows.label_available_at.lt(cut)
                               & ~rows.continuity_block.isin(cross_start)]
        val_ids = rows.index[rows.timestamp.ge(cut) & rows.timestamp.lt(end)
                             & rows.label_available_at.lt(end)
                             & ~rows.continuity_block.isin(cross_start | cross_end)]
        assert not set(rows.loc[train_ids, 'continuity_block']) & set(rows.loc[val_ids, 'continuity_block'])
        tr = transitions.loc[transitions.owner.isin(train_ids)].copy()
        va = transitions.loc[transitions.owner.isin(val_ids)].copy()
        assert len(tr) and len(va)
        for table, cutoff in ((tr, cut), (va, end)):
            valid = pd.DatetimeIndex(rows.timestamp.iloc[table.owner.to_numpy()]) + pd.to_timedelta(table.horizon_hours.to_numpy(), unit='h')
            assert (valid < cutoff).all()
        matrices = []
        for table in (tr, va):
            x = rows[old['features']].iloc[table.owner.to_numpy()].reset_index(drop=True)
            x['horizon_hours'] = table.horizon_hours.to_numpy()
            x['clean_run'] = table.clean_run.to_numpy()
            matrices.append(x)
        model = lgb.train(old['params'], lgb.Dataset(matrices[0], label=tr.target), num_boost_round=old['rounds'])
        model_path = out / f'{year}_model.txt'
        model_path.write_text(model.model_to_string(), encoding='utf-8')
        prediction = model.predict(matrices[1], num_threads=1)
        loaded = lgb.Booster(model_str=model_path.read_text(encoding='utf-8'))
        np.testing.assert_array_equal(prediction, loaded.predict(matrices[1], num_threads=1))
        va['probability'] = prediction
        va['valid_time'] = pd.DatetimeIndex(rows.timestamp.iloc[va.owner.to_numpy()]) + pd.to_timedelta(va.horizon_hours.to_numpy(), unit='h')
        va['block'] = rows.continuity_block.iloc[va.owner.to_numpy()].to_numpy()
        va.to_parquet(out / f'{year}_probabilities.parquet', index=False)
        states = []
        for state, subset in va.groupby('clean_run'):
            states.append({'state': int(state), 'rows': len(subset),
                           'observed_clean_rate': float(subset.target.mean()),
                           'predicted_clean_rate': float(subset.probability.mean()),
                           'bias': float((subset.probability - subset.target).mean()),
                           'physical_times': int(subset.valid_time.nunique()),
                           'failure_physical_times': int(subset.loc[subset.target.eq(0), 'valid_time'].nunique()),
                           'failure_blocks': int(subset.loc[subset.target.eq(0), 'block'].nunique())})
        forecast = [c for c in old['features'] if c.startswith('gefs_')]
        summaries[str(year)] = {'training_origins': len(train_ids), 'supervised_origins': int(tr.owner.nunique()),
                                'validation_origins': len(val_ids), 'validation_supervised_origins': int(va.owner.nunique()),
                                'training_transitions': len(tr), 'validation_transitions': len(va),
                                'training_forecast_complete_origins': int(rows.loc[train_ids, forecast].notna().all(axis=1).sum()),
                                'validation_forecast_complete_origins': int(rows.loc[val_ids, forecast].notna().all(axis=1).sum()),
                                'states': states}
        print(json.dumps({str(year): summaries[str(year)]}, indent=2), flush=True)
    assert before == {str(p): sha(Path(p)) for p in before}
    report = {'years': summaries, 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
