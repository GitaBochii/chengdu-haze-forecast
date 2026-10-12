"""Development-only AFT distribution selection, then frozen annual diagnostics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.benchmark_observed_business_aft import ROOT, metrics, sha

BASE = ROOT / 'artifacts/task01_recovery_head/observed_annual_update_v1'
DEV = ROOT / 'artifacts/task01_recovery_head/observed_business_aft_v1'
OUT = ROOT / 'artifacts/task01_recovery_head/aft_distribution_development_v1'


def prediction(model, frame, names, distribution, scale):
    location = model.predict(xgb.DMatrix(frame[names].astype(float), missing=np.nan))
    # Extreme noise has median log(log(2)); exp(location) alone is not its median.
    return location * (np.log(2.0) ** scale if distribution == 'extreme' else 1.0)


def fit(frame, names, distribution, scale):
    matrix = xgb.DMatrix(frame[names].astype(float), missing=np.nan)
    matrix.set_float_info('label_lower_bound', frame.lower_bound_hours)
    matrix.set_float_info('label_upper_bound', frame.upper_bound_hours)
    params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
        'aft_loss_distribution': distribution, 'aft_loss_distribution_scale': scale,
        'eta': .05, 'max_depth': 4, 'min_child_weight': 8, 'subsample': .9,
        'colsample_bytree': .8, 'lambda': 3.0, 'seed': 42,
        'tree_method': 'hist', 'verbosity': 0, 'nthread': 1}
    return xgb.train(params, matrix, num_boost_round=500)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol = json.loads((DEV / 'protocol.json').read_bytes())
    names = protocol['features']
    paths = [BASE / str(y) / n for y in (2023, 2024, 2025) for n in ('training_rows.parquet', 'predictions.parquet')]
    paths += [DEV / 'predictions.parquet', DEV / 'protocol.json', DEV / 'report.json',
              Path(__file__), ROOT / 'scripts/benchmark_observed_business_aft.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    settings = [(dist, scale) for dist in ('normal', 'logistic', 'extreme') for scale in (.6, 1.2)]
    OUT.mkdir(parents=True)
    frozen = {'input_sha256': hashes, 'settings': settings, 'features': names,
        'selection': '2023 score_allowed only; both main MAEs improve legacy and 24h tail not worse; lowest block MAE',
        'score_semantics': 'conditional median, extreme factor log(2)**scale',
        'source': 'https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html',
        'limitations': ['reused development years', 'exact-only scoring excludes censored truth',
                       'legacy training control not deployed-model replay']}
    (OUT / 'protocol.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    train = pd.read_parquet(BASE / '2023/training_rows.parquet')
    validation = pd.read_parquet(DEV / 'predictions.parquet')
    allowed = validation.score_allowed
    baseline = metrics(validation.loc[allowed], validation.loc[allowed, 'legacy_training'].to_numpy())
    scores = {}
    passing = []
    for dist, scale in settings:
        name = f'{dist}_{scale}'
        model = fit(train, names, dist, scale)
        model.save_model(OUT / f'{name}.json')
        pred = prediction(model, validation, names, dist, scale)
        assert np.isfinite(pred).all()
        if name == 'normal_1.2':
            np.testing.assert_array_equal(pred, validation.observed_bounds)
        validation[name] = pred
        result = metrics(validation.loc[allowed], pred[allowed])
        result['max_prediction'] = float(pred.max())
        result['passes_gate'] = bool(result['row_mae'] < baseline['row_mae'] and
            result['block_mae'] < baseline['block_mae'] and result['tail_24h_block_mae'] <= baseline['tail_24h_block_mae'])
        scores[name] = result
        if result['passes_gate']:
            passing.append((result['block_mae'], name, dist, scale))
    validation.to_parquet(OUT / 'development_predictions.parquet', index=False)
    winner = min(passing) if passing else None
    # Persist the development choice before opening any cross-year validation.
    (OUT / 'selection.json').write_text(json.dumps({'winner': winner, 'scores': scores}, indent=2), encoding='utf-8')
    crossyear = {}
    if winner:
        _, name, dist, scale = winner
        for year in (2024, 2025):
            train = pd.read_parquet(BASE / str(year) / 'training_rows.parquet')
            validation = pd.read_parquet(BASE / str(year) / 'predictions.parquet')
            model = fit(train, names, dist, scale)
            model.save_model(OUT / f'{year}_{name}.json')
            pred = prediction(model, validation, names, dist, scale)
            assert np.isfinite(pred).all()
            validation['candidate'] = pred
            validation.to_parquet(OUT / f'{year}_predictions.parquet', index=False)
            crossyear[str(year)] = {'candidate': metrics(validation, pred),
                'legacy': metrics(validation, validation.legacy_training.to_numpy())}
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_distribution_comparison', 'winner': winner,
        'development': scores, 'crossyear': crossyear, 'inputs_unchanged': True,
        'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'winner': winner, 'development': scores, 'crossyear': crossyear}, indent=2))


if __name__ == '__main__':
    main()
