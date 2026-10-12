"""Rebuild members, controls, predictions and scores without benchmark helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    root = Path(__file__).resolve().parents[1]
    base = root/'artifacts/task01_recovery_head'
    out = base/'neighbor_city_preflight_20260918'
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    report = json.loads((out/'report.json').read_text(encoding='utf-8'))
    for path, expected in protocol['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    for name, expected in report['output_sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == expected
    data = pd.read_parquet(base/'gefs_2021_extension_v1/f024_training_rows.parquet')
    extra = pd.read_parquet(base/'neighbor_city_features_20260918/features.parquet').set_index('timestamp')
    total = 0; metrics = 0; pairs = 0
    for year in protocol['years']:
        v = pd.read_parquet(out/f'{year}_predictions.parquet')
        cut = pd.Timestamp(year=year, month=1, day=1, tz='Asia/Shanghai')
        end = cut.replace(year=year+1)
        eligible = data[(data.timestamp >= cut)&(data.timestamp < end)&(data.label_available_at < end)]
        pd.testing.assert_frame_equal(v[data.columns], eligible.reset_index(drop=True))
        train = data[(data.timestamp < cut)&(data.label_available_at < cut)&(~data.continuity_block.isin(eligible.continuity_block))]
        pd.testing.assert_frame_equal(pd.read_parquet(out/f'{year}_training.parquet'), train[['timestamp','label_available_at','continuity_block']].reset_index(drop=True))
        assert set(train.continuity_block).isdisjoint(v.continuity_block)
        for arm in protocol['arms']:
            x = v[protocol['base_features']].astype(float).copy()
            if arm != 'native':
                for index, name in enumerate(protocol['extra_features']):
                    values = extra.loc[v.timestamp, name].to_numpy().copy()
                    if arm == 'masked' or (arm == 'local' and index >= 3):
                        values[np.isfinite(values)] = 0.
                    np.testing.assert_array_equal(v[name], extra.loc[v.timestamp, name].to_numpy())
                    x[name] = values
            model = xgb.Booster(); model.load_model(out/f'{year}_{arm}.ubj')
            prediction = model.predict(xgb.DMatrix(x))
            np.testing.assert_array_equal(prediction, v[f'prediction_{arm}'])
            total += len(v)
            exact = v.exact_hours.notna().to_numpy()
            error = abs(prediction[exact]-v.exact_hours.to_numpy()[exact])
            by_block = {}; tail = {}
            for block, truth, e in zip(v.continuity_block.to_numpy()[exact], v.exact_hours.to_numpy()[exact], error):
                by_block.setdefault(block, []).append(e)
                if truth >= 24:
                    tail.setdefault(block, []).append(e)
            actual = [np.mean(error), np.mean([np.mean(x) for x in by_block.values()]), np.mean([np.mean(x) for x in tail.values()])]
            expected = report['scores'][str(year)][arm]
            np.testing.assert_allclose(actual, [expected[k] for k in ['exact_row_mae','exact_block_mae','tail24_block_mae']], rtol=1e-12)
            metrics += 3
        for arm in ['native','masked','local']:
            intervals = []
            for lo, hi, original, candidate in zip(v.lower_bound_hours, v.upper_bound_hours, v[f'prediction_{arm}'], v.prediction_regional):
                endpoints = [lo, hi] if np.isfinite(hi) else [lo, max(lo, original, candidate)]
                endpoints += [p for p in (original, candidate) if lo <= p <= hi]
                differences = [abs(candidate-y)-abs(original-y) for y in endpoints]
                intervals.append([min(differences), max(differences)])
            bounds = np.array(intervals)
            for index, bound in enumerate(['lower','upper']):
                block_values = {}
                for block,value in zip(v.continuity_block,bounds[:,index]):
                    block_values.setdefault(block,[]).append(value)
                actual = [bounds[:,index].mean(),np.mean([np.mean(x) for x in block_values.values()])]
                expected = report['paired'][str(year)][arm]
                np.testing.assert_allclose(actual,[expected[f'all_row_delta_{bound}'],expected[f'all_block_delta_{bound}']],rtol=1e-10,atol=1e-10)
            pairs += 1
    result = {'prediction_values_replayed': total, 'maes_independently_checked': metrics,
              'full_population_paired_comparisons_checked': pairs, 'membership_and_hashes': 'passed',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (out/'verification.json').open('x', encoding='utf-8') as target:
        json.dump(result,target,indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
