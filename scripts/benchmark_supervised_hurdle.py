"""Frozen 2023 development check for a supervised support-separated mixture."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary
from haze_forecast.supervised_hurdle_math import median, nll

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT / 'artifacts/task01_recovery_head'
    out = b / 'supervised_hurdle_2023_20260918'
    if out.exists():
        raise FileExistsError(out)
    paths = [b / 'gefs_2021_extension_v1/f024_training_rows.parquet',
             b / 'gefs_2021_extension_v1/inference_rows.parquet',
             b / 'grouped_forecast_aft_2023_20260916/expanded.parquet',
             b / 'duration_separation_20260918_v2/protocol.json',
             ROOT / 'supervised_hurdle_plan_20260918.md', Path(__file__).resolve(),
             ROOT / 'src/haze_forecast/supervised_hurdle_math.py',
             ROOT / 'src/haze_forecast/lognormal_aft_audit.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    t, v, cohort = [pd.read_parquet(p) for p in paths[:3]]
    protocol = json.loads(paths[3].read_bytes())
    assert t.timestamp.lt(pd.Timestamp('2023-01-01', tz='Asia/Shanghai')).all()
    assert t.label_available_at.lt(pd.Timestamp('2023-01-01', tz='Asia/Shanghai')).all()
    assert not set(t.continuity_block) & set(cohort.continuity_block)
    target = np.where(t.lower_bound_hours >= 24, 1., np.where(t.upper_bound_hours < 24, 0., np.nan))
    known = np.isfinite(target)
    names = [n for n in protocol['features'] if t.loc[known, n].notna().any()]
    model = HistGradientBoostingClassifier(**protocol['params'])
    with threadpool_limits(limits=2):
        model.fit(t.loc[known, names], target[known])
        pt, pv = model.predict_proba(t[names])[:, 1], model.predict_proba(v[names])[:, 1]
    q = float(target[known].mean())
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps({'input_sha256': hashes, 'features': names,
        'params': protocol['params'], 'constant_probability': q}, indent=2), encoding='utf-8')
    joblib.dump(model, out / 'gate.joblib')
    training, predictions = t.copy(), v[['timestamp']].copy()
    fits = {}
    for name, train_p, valid_p in [('input_gate', pt, pv),
                                  ('constant_gate', np.full(len(t), q), np.full(len(v), q))]:
        fit = minimize(lambda mu: nll(t.lower_bound_hours, t.upper_bound_hours, train_p, mu).mean(),
                       np.log([8., 48.]), method='L-BFGS-B', bounds=[(-5., 10.)]*2,
                       options={'maxiter': 200, 'ftol': 1e-10})
        if not fit.success or (fit.x <= -4.999).any() or (fit.x >= 9.999).any():
            raise RuntimeError(str(fit))
        fits[name] = {'mu': fit.x.tolist(), 'nll': float(fit.fun), 'iterations': fit.nit,
                      'message': str(fit.message)}
        training[name+'_probability'] = train_p
        predictions[name+'_probability'] = valid_p
        predictions[name] = median(valid_p, fit.x)
    scored = cohort.merge(predictions, on='timestamp', validate='one_to_one')
    assert len(scored) == len(cohort)
    scores = {n: score_predictions(scored, scored[n].to_numpy())
              for n in ['forecast_aft', 'input_gate', 'constant_gate']}
    paired = {n: paired_error_summary(scored, scored[n].to_numpy(), scored.input_gate.to_numpy())
              for n in ['forecast_aft', 'constant_gate']}
    passed = all(scores['input_gate']['exact_row_mae'] < scores[n]['exact_row_mae']
                 and scores['input_gate']['exact_block_mae'] < scores[n]['exact_block_mae']
                 and scores['input_gate']['tail24_block_mae'] <= scores[n]['tail24_block_mae']
                 and paired[n]['all_row_delta_upper'] <= 0
                 and paired[n]['all_block_delta_upper'] <= 0 for n in paired)
    training.to_parquet(out / 'training.parquet', index=False)
    predictions.to_parquet(out / 'predictions.parquet', index=False)
    scored.to_parquet(out / 'scored.parquet', index=False)
    for p, h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    report = {'fits': fits, 'scores': scores, 'paired': paired, 'pass': bool(passed),
              'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
