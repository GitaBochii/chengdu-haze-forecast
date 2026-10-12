"""Independent supervision, saved-model and seven-state matrix replay."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'run_length_transition_2022_v2'
    protocol = json.loads((out / 'protocol.json').read_bytes())
    report = json.loads((out / 'report.json').read_bytes())
    for p, digest in protocol['input_sha256'].items():
        assert sha(Path(p)) == digest
    for name, digest in report['output_sha256'].items():
        assert sha(out / name) == digest
    train = pd.read_parquet(base / 'gefs_temporal_calibration_data_v1/training_rows.parquet')
    grid = pd.read_parquet(base / 'observed_business_dataset_v1/features.parquet').set_index('timestamp')
    transitions = pd.read_parquet(out / 'transitions.parquet')
    rebuilt = []
    for owner, row in enumerate(train.itertuples()):
        begin = grid.index.get_loc(row.timestamp) + 1
        values = grid.aqi.iloc[begin:begin + int(row.upper_bound_hours) + 5].to_numpy()
        bad = np.flatnonzero(~np.isfinite(values))
        values = values[:bad[0]] if len(bad) else values
        clean = values <= 100
        run = 0
        for i, value in enumerate(clean):
            rebuilt.append((owner, i + 1, run, int(value)))
            run = run + 1 if value else 0
            if run == 6:
                break
    np.testing.assert_array_equal(np.asarray(rebuilt), transitions.to_numpy())
    times = train.timestamp.iloc[transitions.owner.to_numpy()].reset_index(drop=True)
    last = times + pd.to_timedelta(transitions.horizon_hours, unit='h')
    assert last.lt(pd.Timestamp('2022-01-01', tz='Asia/Shanghai')).all()
    val = pd.read_parquet(out / 'predictions.parquet')
    saved = np.load(out / 'distribution.npy')
    model = lgb.Booster(model_str=(out / 'model.txt').read_text(encoding='utf-8'))
    horizon, names = protocol['horizon'], protocol['features']
    output = []
    for start in range(0, len(val), 16):
        x = val[names].iloc[start:start + 16].to_numpy(dtype=float)
        n = len(x)
        augmented = np.column_stack((np.repeat(x, horizon * 6, axis=0),
                                     np.tile(np.repeat(np.arange(1, horizon + 1), 6), n),
                                     np.tile(np.arange(6), n * horizon)))
        q = model.predict(augmented, num_threads=1).reshape(n, horizon, 6)
        mass = np.zeros((n, 7))
        mass[:, 0] = 1
        events = []
        for hour in range(horizon):
            matrix = np.zeros((n, 7, 7))
            for state in range(6):
                matrix[:, state, 0] = 1 - q[:, hour, state]
                matrix[:, state, state + 1] = q[:, hour, state]
            matrix[:, 6, 6] = 1
            updated = np.einsum('bi,bij->bj', mass, matrix)
            if hour >= 5:
                events.append(updated[:, 6] - mass[:, 6])
            mass = updated
        output.append(np.column_stack((*events, mass[:, :6].sum(axis=1))))
    replay = np.concatenate(output)
    np.testing.assert_allclose(replay, saved, atol=1e-12, rtol=1e-10)
    cdf = saved[:, :-1].cumsum(axis=1)
    prediction = np.array([np.flatnonzero(row >= .5)[0] + 1 if (row >= .5).any() else np.nan for row in cdf])
    np.testing.assert_array_equal(prediction, val.transition_prediction)
    if report['candidate'] is not None:
        err = abs(prediction - val.exact_hours.to_numpy())
        score = report['candidate']
        np.testing.assert_allclose(err.mean(), score['exact_row_mae'])
        np.testing.assert_allclose(pd.Series(err).groupby(val.continuity_block.to_numpy()).mean().mean(), score['exact_block_mae'])
        mask = val.exact_hours.ge(24).to_numpy()
        np.testing.assert_allclose(pd.Series(err[mask]).groupby(val.continuity_block.to_numpy()[mask]).mean().mean(), score['tail24_block_mae'])
    result = {'status': 'PASS', 'verified_transitions': len(transitions), 'verified_predictions': len(val),
              'maximum_distribution_difference': float(abs(replay - saved).max()),
              'verifier_sha256': sha(Path(__file__))}
    (out / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
