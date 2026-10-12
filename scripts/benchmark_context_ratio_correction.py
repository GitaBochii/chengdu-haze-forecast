"""Forward-only conditional ratio correction from saved temporal predictions."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.features import MEASUREMENT_COLUMNS, SNAPSHOT_DERIVED_COLUMNS
from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from scripts.benchmark_observed_business_aft import ROOT, sha


def context(frame):
    names = sorted(set(MEASUREMENT_COLUMNS) | SNAPSHOT_DERIVED_COLUMNS)
    result = frame[names].astype(float).copy()
    base = frame.temporal_prediction.to_numpy(float)
    if not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Positive finite base prediction required')
    result['base_log_hours'] = np.log(base)
    return result


def main():
    source = ROOT / 'artifacts/task01_recovery_head/early_temporal_predictions_v1'
    out = ROOT / 'artifacts/task01_recovery_head/context_ratio_correction_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / 'predictions.parquet', source / 'protocol.json', Path(__file__),
             ROOT / 'src/haze_forecast/observed_aft.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    data = pd.read_parquet(paths[0])
    out.mkdir()
    protocol = {'years': [2018, 2019], 'seed': 42, 'rounds': 500,
                'inputs': list(context(data).columns), 'input_sha256': hashes,
                'target': 'interval bounds divided by historical out-of-time base prediction',
                'gate': 'row/process MAE decrease, tail24 process MAE does not increase',
                'limitations': ['Historical development years, not untouched tests',
                                'No in-sample base predictions, label midpoint imputation, or parameter search',
                                'Fixed AFT ratio likelihood is not direct hour-scale MAE optimization',
                                'Whole evaluation processes excluded from correction training']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    report = {}
    for year in (2018, 2019):
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        valid = data.loc[data.timestamp.ge(cut) & data.timestamp.lt(end)].copy()
        train = data.loc[data.timestamp.lt(cut) & data.label_available_at.lt(cut)
                         & ~data.continuity_block.isin(valid.continuity_block)].copy()
        assert len(train) and len(valid)
        assert train.model_cutoff.le(train.timestamp).all()
        assert not set(train.continuity_block) & set(valid.continuity_block)
        model = ObservedAFTSurvivalRegressor(seed=42, nthread=1, num_boost_round=500).fit_bounds(
            context(train), train.lower_bound_hours / train.temporal_prediction,
            train.upper_bound_hours / train.temporal_prediction)
        joblib.dump(model, out / f'{year}_ratio_model.joblib')
        ratios = model.predict(context(valid))
        np.testing.assert_array_equal(ratios, joblib.load(out / f'{year}_ratio_model.joblib').predict(context(valid)))
        pred = valid.temporal_prediction.to_numpy() * ratios
        assert np.isfinite(pred).all()
        train.to_parquet(out / f'{year}_training.parquet', index=False)
        valid['corrected_prediction'] = pred
        valid.to_parquet(out / f'{year}_predictions.parquet', index=False)
        control = score_predictions(valid, valid.temporal_prediction)
        candidate = score_predictions(valid, pred)
        checks = {k: candidate[k] < control[k] for k in ('exact_row_mae', 'exact_block_mae')}
        checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
        report[str(year)] = {'training_rows': len(train), 'training_blocks': int(train.continuity_block.nunique()),
                             'control': control, 'candidate': candidate, 'checks': checks,
                             'passed': all(checks.values())}
        (out / f'{year}_decision.json').write_text(json.dumps(report[str(year)], indent=2), encoding='utf-8')
        print(json.dumps({str(year): report[str(year)]}, indent=2), flush=True)
        if not all(checks.values()):
            break
    assert hashes == {str(p): sha(p) for p in paths}
    (out / 'report.json').write_text(json.dumps({'years': report, 'deployment_authorized': False,
        'output_sha256': {p.name: sha(p) for p in out.iterdir()}}, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
