"""Fixed CatBoost SurvivalAft comparison, without imputing interval labels."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from haze_forecast.cohort_scorecard import score_predictions
from scripts.benchmark_observed_business_aft import ROOT, sha


def make_model(iterations=500):
    return CatBoostRegressor(loss_function='SurvivalAft:dist=Normal;scale=1.2',
                             iterations=iterations, learning_rate=.05, depth=5,
                             l2_leaf_reg=8, random_strength=1, random_seed=42,
                             verbose=False, allow_writing_files=False, thread_count=1)


def median_hours(model, frame):
    raw = model.predict(frame.astype(float), prediction_type='RawFormulaVal')
    result = np.exp(raw)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite time prediction')
    return result


def main():
    source = ROOT / 'artifacts/task01_recovery_head/transition_earlier_recovery_v1'
    out = ROOT / 'artifacts/task01_recovery_head/catboost_interval_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source / 'protocol.json', Path(__file__)]
    paths += [source / f'{year}_{suffix}' for year in (2020, 2021)
              for suffix in ('training.parquet', 'predictions.parquet', 'aft.joblib')]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    names = json.loads(paths[0].read_bytes())['features']
    out.mkdir()
    protocol = {'input_sha256': hashes, 'features': names, 'years': [2020, 2021],
                'parameters': make_model().get_params(),
                'prediction': 'exp(raw margin), Normal noise median is zero',
                'gate': 'row/process MAE decrease, tail24 process MAE does not increase',
                'limitations': ['Historical development, no deployment',
                                'Framework and tree construction differ; not a pure structure ablation',
                                'Fixed settings adapted from existing CatBoost MAE configuration',
                                'No eval_set, early stopping, or parameter search']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scores = {}
    for year in (2020, 2021):
        train = pd.read_parquet(source / f'{year}_training.parquet')
        valid = pd.read_parquet(source / f'{year}_predictions.parquet')
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        assert train.timestamp.lt(cut).all() and train.label_available_at.lt(cut).all()
        assert valid.timestamp.ge(cut).all() and valid.label_available_at.lt(end).all()
        assert not set(train.continuity_block) & set(valid.continuity_block)
        target = train[['lower_bound_hours', 'upper_bound_hours']].to_numpy(float)
        assert np.isfinite(target).all() and (target[:, 0] > 0).all() and (target[:, 1] >= target[:, 0]).all()
        model = make_model()
        model.fit(train[names].astype(float), target)
        joblib.dump(model, out / f'{year}_model.joblib')
        pred = median_hours(model, valid[names])
        np.testing.assert_array_equal(pred, median_hours(joblib.load(out / f'{year}_model.joblib'), valid[names]))
        baseline = joblib.load(source / f'{year}_aft.joblib').predict(valid[names])
        np.testing.assert_array_equal(baseline, valid.paired_aft)
        result = valid.copy()
        result['catboost_prediction'] = pred
        result.to_parquet(out / f'{year}_predictions.parquet', index=False)
        control, candidate = score_predictions(valid, baseline), score_predictions(valid, pred)
        checks = {k: candidate[k] < control[k] for k in ('exact_row_mae', 'exact_block_mae')}
        checks['tail24_block_mae'] = candidate['tail24_block_mae'] <= control['tail24_block_mae']
        scores[str(year)] = {'control': control, 'candidate': candidate,
                             'checks': checks, 'passed': all(checks.values())}
        (out / f'{year}_decision.json').write_text(json.dumps(scores[str(year)], indent=2), encoding='utf-8')
        print(json.dumps({str(year): scores[str(year)]}, indent=2), flush=True)
        if not all(checks.values()):
            break
    assert hashes == {str(p): sha(p) for p in paths}
    (out / 'report.json').write_text(json.dumps({'years': scores, 'deployment_authorized': False,
        'output_sha256': {p.name: sha(p) for p in out.iterdir()}}, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
