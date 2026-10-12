"""Paired full-column AFT control for unused-column perturbation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.benchmark_gefs_72h_aft import ROOT, compare_cohort, sha, verify_hashes


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_dual_horizon_v1'
    out = base / 'gefs_full_columns_v1'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((source / 'report.json').read_bytes())
    protocol = json.loads((source / 'protocol.json').read_bytes())
    inputs = {str(source / n): h for n, h in report['output_sha256'].items()}
    inputs.update(protocol['input_sha256'])
    inputs[str(Path(__file__))] = sha(Path(__file__))
    inputs[str(source / 'report.json')] = sha(source / 'report.json')
    verify_hashes(inputs)
    names = protocol['features']
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'inference_rows.parquet')
    params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
              'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.2,
              'eta': 0.05, 'max_depth': 4, 'min_child_weight': 8, 'subsample': 0.9,
              'colsample_bytree': 1.0, 'lambda': 3.0, 'seed': 42,
              'tree_method': 'hist', 'verbosity': 0, 'nthread': 1}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({
        'input_sha256': inputs, 'features': names, 'params': params, 'rounds': 500,
        'selection': 'Three pre-specified arms; only column sampling changes from frozen settings.',
        'arms': ['f024', 'empty', 'dual'], 'deployment_authorized': False
    }, indent=2), encoding='utf-8')
    results, predictions, margins = {}, {}, {}
    for arm in ['f024', 'empty', 'dual']:
        columns = names[:104] if arm == 'f024' else names
        x, v = train[columns].astype(float).copy(), val[columns].astype(float).copy()
        if arm == 'empty':
            x[names[104:]], v[names[104:]] = np.nan, np.nan
        matrix = xgb.DMatrix(x, missing=np.nan)
        matrix.set_float_info('label_lower_bound', train.lower_bound_hours.to_numpy())
        matrix.set_float_info('label_upper_bound', train.upper_bound_hours.to_numpy())
        model = xgb.train(params, matrix, num_boost_round=500)
        model.save_model(out / f'{arm}.ubj')
        inference = xgb.DMatrix(v, missing=np.nan)
        predictions[arm] = model.predict(inference)
        margins[arm] = model.predict(inference, output_margin=True)
        replay = xgb.Booster()
        replay.load_model(out / f'{arm}.ubj')
        np.testing.assert_array_equal(replay.predict(inference), predictions[arm])
        results[arm] = {}
        for cohort in ['original', 'expanded']:
            data = pd.read_parquet(source / f'{cohort}_predictions.parquet')
            results[arm][cohort] = compare_cohort(data, predictions[arm])
    paired = {}
    for cohort in ['original', 'expanded']:
        data = pd.read_parquet(source / f'{cohort}_predictions.parquet')
        for arm, prediction in predictions.items():
            data[f'{arm}_full_columns'] = prediction
        data['forecast_aft'] = predictions['f024']
        paired[cohort] = compare_cohort(data, predictions['dual'])
        data.to_parquet(out / f'{cohort}_predictions.parquet', index=False)
    verify_hashes(inputs)
    result = {'cohorts_vs_original_f024': results, 'dual_vs_full_column_f024': paired,
              'empty_margin_identical_to_f024': bool(np.array_equal(margins['empty'], margins['f024'])),
              'empty_max_prediction_change_hours': float(np.max(np.abs(predictions['empty']-predictions['f024']))),
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
