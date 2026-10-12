"""Historical-PBL ablation on original forecast training support."""

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
    control = base / 'gefs_full_columns_v1'
    out = base / 'original_gefs_without_pbl_v1'
    if out.exists():
        raise FileExistsError(out)
    inputs = {str(Path(__file__)): sha(Path(__file__))}
    for folder in [source, control]:
        report = json.loads((folder / 'report.json').read_bytes())
        inputs.update({str(folder / n): h for n, h in report['output_sha256'].items()})
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
    verify_hashes(inputs)
    original_protocol = json.loads((source / 'protocol.json').read_bytes())
    inputs.update(original_protocol['input_sha256'])
    settings = json.loads((control / 'protocol.json').read_bytes())
    params = settings['params']
    names = original_protocol['features']
    kept = [name for name in names if not name.startswith('boundary_layer_height_m')]
    assert len(kept) == 91 and 'gefs_control_boundary_layer_height_m_f024' in kept
    train = pd.read_parquet(source / 'f024_training_rows.parquet')
    inference = pd.read_parquet(source / 'inference_rows.parquet')
    # The existing full-column control used this exact original-support matrix.
    reference = pd.read_parquet(base / 'gefs_dual_horizon_v1/training_rows.parquet')
    pd.testing.assert_frame_equal(train[names], reference[names])
    verify_hashes(inputs)
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({
        'input_sha256': inputs, 'features': kept, 'params': params, 'rounds': 500,
        'training_rows': len(train), 'training_forecast_rows': int(train.gefs_mean_temperature_c_f024.notna().sum()),
        'gate': 'Original and expanded row/block MAE improve and tail no worse than frozen f024.',
        'deployment_authorized': False
    }, indent=2), encoding='utf-8')
    matrix = xgb.DMatrix(train[kept].astype(float))
    matrix.set_float_info('label_lower_bound', train.lower_bound_hours.to_numpy())
    matrix.set_float_info('label_upper_bound', train.upper_bound_hours.to_numpy())
    model = xgb.train(params, matrix, num_boost_round=500)
    model.save_model(out / 'model.ubj')
    prediction = model.predict(xgb.DMatrix(inference[kept].astype(float)))
    cohorts, paired = {}, {}
    for cohort, filename in [('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]:
        frame = pd.read_parquet(source / filename)
        cohorts[cohort] = compare_cohort(frame, prediction)
        full = pd.read_parquet(control / f'{cohort}_predictions.parquet')
        pd.testing.assert_series_equal(frame.timestamp, full.timestamp)
        paired_frame = frame.copy()
        paired_frame['forecast_aft'] = full.f024_full_columns.to_numpy()
        paired[cohort] = compare_cohort(paired_frame, prediction)
        frame['no_history_pbl'] = prediction
        frame.to_parquet(out / filename, index=False)
    verify_hashes(inputs)
    report = {'cohorts': cohorts, 'paired_full_column_control': paired,
              'passed': all(v['passed'] for v in cohorts.values()), 'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
