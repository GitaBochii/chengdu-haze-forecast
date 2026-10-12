"""Fixed pre-2022 interval hazard pilot; no later-year selection."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.interval_hazard_em import IntervalHazardEM
from scripts.benchmark_gefs_72h_aft import verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_temporal_calibration_data_v1'
    out = base / 'interval_hazard_temporal_v1'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((source / 'report.json').read_bytes())
    protocol = json.loads((source / 'protocol.json').read_bytes())
    inputs = dict(protocol['input_sha256'])
    inputs.update({str(source / name): value for name, value in report['output_sha256'].items()})
    paths = [source / 'report.json', Path(__file__),
             ROOT / 'src/haze_forecast/interval_hazard_em.py',
             ROOT / 'src/haze_forecast/interval_hazard_likelihood.py']
    inputs.update({str(p): sha(p) for p in paths})
    verify_hashes(inputs)
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'predictions.parquet')
    names = protocol['features']
    cutoff = pd.Timestamp('2022-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert not set(train.continuity_block) & set(val.continuity_block)
    out.mkdir()
    settings = {'input_sha256': inputs, 'features': names, 'rounds': 300, 'max_em_steps': 5,
                'initialization': 'constant hazard from reciprocal training median lower bound',
                'horizon': 'maximum observed training endpoint; no validation horizon fitting',
                'selection': 'fixed pilot; no later-year scores read',
                'unresolved': 'NaN and failed gate; never drop or cap', 'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    model = IntervalHazardEM().fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    prediction = model.predict(val[names])
    np.testing.assert_allclose(prediction, joblib.load(out / 'model.joblib').predict(val[names]), equal_nan=True)
    val['interval_hazard'] = prediction
    val.to_parquet(out / 'predictions.parquet', index=False)
    allowed = val.score_allowed.to_numpy()
    resolved = bool(np.isfinite(prediction).all())
    candidate = metrics(val.loc[allowed], prediction[allowed]) if resolved else None
    control = metrics(val.loc[allowed], val.loc[allowed, 'forecast_aft'].to_numpy())
    passed = bool(resolved and candidate['row_mae'] < control['row_mae']
                  and candidate['block_mae'] < control['block_mae']
                  and candidate['tail_24h_block_mae'] <= control['tail_24h_block_mae'])
    verify_hashes(inputs)
    result = {'training_rows': len(train), 'horizon': model.horizon, 'trace': model.trace,
              'unresolved_rows': int((~np.isfinite(prediction)).sum()),
              'candidate': candidate, 'control': control, 'pilot_passed': passed,
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
