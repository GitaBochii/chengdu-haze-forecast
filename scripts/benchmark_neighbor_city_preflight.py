"""Fixed four-arm, two-fold test of lagged neighboring-city information."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]
EXTRA = [f'city48_{i:02d}' for i in range(24)]
ARMS = ['native', 'masked', 'local', 'regional']
KEYS = ['exact_row_mae', 'exact_block_mae', 'tail24_block_mae']


def arm_inputs(frame, names, arm):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    result = frame[names if arm == 'native' else names+EXTRA].copy()
    masked = EXTRA if arm == 'masked' else EXTRA[3:] if arm == 'local' else []
    if masked:
        result[masked] = result[masked].where(result[masked].isna(), 0.)
    return result.astype(float)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT/'artifacts/task01_recovery_head'
    out = base/'neighbor_city_preflight_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [base/'gefs_2021_extension_v1/f024_training_rows.parquet',
             base/'neighbor_city_features_20260918/features.parquet',
             base/'neighbor_city_features_20260918/report.json',
             base/'grouped_forecast_aft_2023_20260916/protocol.json',
             ROOT/'neighbor_city_preflight_plan_20260918.md', Path(__file__).resolve(),
             ROOT/'src/haze_forecast/cohort_scorecard.py', ROOT/'src/haze_forecast/paired_interval_score.py']
    hashes = {str(p): digest(p) for p in paths}
    original = pd.read_parquet(paths[0]); extra = pd.read_parquet(paths[1])
    feature_report = json.loads(paths[2].read_text(encoding='utf-8'))
    assert digest(paths[1]) == feature_report['feature_sha256']
    assert list(feature_report['column_mapping']) == EXTRA
    assert all(feature_report['column_mapping'][c].startswith('成都_') for c in EXTRA[:3])
    pd.testing.assert_series_equal(original.timestamp, extra.timestamp)
    assert ((original.timestamp-extra.source_timestamp) == pd.Timedelta(hours=48)).all()
    data = pd.concat([original, extra[EXTRA]], axis=1)
    frozen = json.loads(paths[3].read_bytes()); names = frozen['features']
    out.mkdir()
    protocol = {'input_sha256': hashes, 'base_features': names, 'extra_features': EXTRA,
                'params': frozen['params'], 'rounds': 500, 'arms': ARMS, 'years': [2021, 2022]}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    scores = {}; paired = {}; gates = {}
    for year in (2021, 2022):
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        valid = data.loc[data.timestamp.ge(cut)&data.timestamp.lt(end)&data.label_available_at.lt(end)].copy()
        train = data.loc[data.timestamp.lt(cut)&data.label_available_at.lt(cut)&~data.continuity_block.isin(valid.continuity_block)].copy()
        assert not set(train.continuity_block)&set(valid.continuity_block)
        train[['timestamp','label_available_at','continuity_block']].to_parquet(out/f'{year}_training.parquet', index=False)
        scores[str(year)] = {}
        for arm in ARMS:
            matrix = xgb.DMatrix(arm_inputs(train, names, arm))
            matrix.set_float_info('label_lower_bound', train.lower_bound_hours)
            matrix.set_float_info('label_upper_bound', train.upper_bound_hours)
            infer = xgb.DMatrix(arm_inputs(valid, names, arm))
            model = xgb.train(frozen['params'], matrix, num_boost_round=500)
            valid[f'prediction_{arm}'] = model.predict(infer)
            model.save_model(out/f'{year}_{arm}.ubj')
            restored = xgb.Booster(); restored.load_model(out/f'{year}_{arm}.ubj')
            np.testing.assert_array_equal(restored.predict(infer), valid[f'prediction_{arm}'])
            scores[str(year)][arm] = score_predictions(valid, valid[f'prediction_{arm}'])
            print(year, arm, {k:scores[str(year)][arm][k] for k in KEYS}, flush=True)
        paired[str(year)] = {arm: paired_error_summary(valid, valid[f'prediction_{arm}'], valid.prediction_regional) for arm in ARMS[:-1]}
        candidate = scores[str(year)]['regional']
        gates[str(year)] = {}
        for arm in ARMS[:-1]:
            control = scores[str(year)][arm]; pair = paired[str(year)][arm]
            gates[str(year)][arm] = bool(candidate[KEYS[0]] < control[KEYS[0]] and
                candidate[KEYS[1]] < control[KEYS[1]] and candidate[KEYS[2]] <= control[KEYS[2]] and
                pair['all_row_delta_upper'] <= 0 and pair['all_block_delta_upper'] <= 0)
        valid.to_parquet(out/f'{year}_predictions.parquet', index=False)
    assert all(digest(Path(p)) == h for p,h in hashes.items())
    report = {'scores': scores, 'paired': paired, 'gates': gates,
              'passed': all(g for year in gates.values() for g in year.values()),
              'output_sha256': {p.name:digest(p) for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'gates': gates, 'passed': report['passed']}), flush=True)


if __name__ == '__main__':
    main()
