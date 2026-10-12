"""Full-column paired ablation of historical PBL, retaining forecast PBL."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.benchmark_gefs_72h_aft import compare_cohort, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_2021_extension_v1'
    out = base / 'gefs_history_pbl_ablation_v1'
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
    removed = [name for name in names if name.startswith('boundary_layer_height_m')]
    assert len(removed) == 13
    kept = [name for name in names if name not in removed]
    assert 'gefs_control_boundary_layer_height_m_f024' in kept and len(kept) == 91
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'inference_rows.parquet')
    params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
              'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.2,
              'eta': 0.05, 'max_depth': 4, 'min_child_weight': 8, 'subsample': 0.9,
              'colsample_bytree': 1.0, 'lambda': 3.0, 'seed': 42,
              'tree_method': 'hist', 'verbosity': 0, 'nthread': 1}
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({
        'input_sha256': inputs, 'features': {'full': names, 'no_history_pbl': kept},
        'removed': removed, 'params': params, 'rounds': 500,
        'selection': 'Paired historical PBL ablation, retaining GEFS PBL and all rows.',
        'deployment_authorized': False
    }, indent=2), encoding='utf-8')
    predictions, results = {}, {}
    for arm, columns in [('full', names), ('no_history_pbl', kept)]:
        matrix = xgb.DMatrix(train[columns].astype(float), missing=np.nan)
        matrix.set_float_info('label_lower_bound', train.lower_bound_hours.to_numpy())
        matrix.set_float_info('label_upper_bound', train.upper_bound_hours.to_numpy())
        model = xgb.train(params, matrix, num_boost_round=500)
        model.save_model(out / f'{arm}.ubj')
        inference = xgb.DMatrix(val[columns].astype(float), missing=np.nan)
        predictions[arm] = model.predict(inference)
        replay = xgb.Booster()
        replay.load_model(out / f'{arm}.ubj')
        np.testing.assert_array_equal(replay.predict(inference), predictions[arm])
        results[arm] = {}
        for cohort, filename in [('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]:
            frame = pd.read_parquet(source / filename)
            results[arm][cohort] = compare_cohort(frame, predictions[arm])
    paired = {}
    for cohort, filename in [('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]:
        frame = pd.read_parquet(source / filename)
        for arm, prediction in predictions.items():
            frame[arm] = prediction
        control = frame.copy()
        control['forecast_aft'] = predictions['full']
        paired[cohort] = compare_cohort(control, predictions['no_history_pbl'])
        frame.to_parquet(out / filename, index=False)
    verify_hashes(inputs)
    result = {'cohorts_vs_frozen_f024': results, 'paired_ablation': paired,
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
