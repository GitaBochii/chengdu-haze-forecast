"""Fixed block-origin inverse weighting of complete observed-prefix likelihoods."""

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_hazard import expand_bounds
from haze_forecast.weather_source_ablation import uniform_weather_rows
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    root = ROOT / 'artifacts/task01_recovery_head'
    base = root / 'observed_station_weather_v1'
    out = root / 'hazard_block_weight_v1'
    if out.exists():
        raise FileExistsError(out)
    train_path = root / 'observed_annual_update_v1/2023/training_rows.parquet'
    full_path = root / 'observed_business_dataset_v1/features.parquet'
    paths = [train_path, full_path, base / 'station_weather.parquet', base / 'protocol.json',
        base / '2023_predictions.parquet', base / '2023_station_no_pbl.joblib', Path(__file__),
        ROOT / 'src/haze_forecast/observed_hazard.py', ROOT / 'src/haze_forecast/discrete_survival.py',
        ROOT / 'src/haze_forecast/weather_source_ablation.py', ROOT / 'src/haze_forecast/features.py',
        ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((base / 'protocol.json').read_bytes())['features']
    t = pd.read_parquet(train_path)
    v = pd.read_parquet(base / '2023_predictions.parquet')
    full = uniform_weather_rows(pd.read_parquet(full_path), pd.read_parquet(base / 'station_weather.parquet')).set_index('timestamp')
    x = full.loc[t.timestamp, names].set_axis(t.index)
    z = full.loc[v.timestamp, names].set_axis(v.index)
    expanded, target, owner = expand_bounds(x, t.lower_bound_hours, t.upper_bound_hours)
    count = t.groupby('continuity_block').timestamp.transform('size').to_numpy()
    origin_weight = 1.0 / count
    block_sums = pd.Series(origin_weight).groupby(t.continuity_block).sum()
    np.testing.assert_allclose(block_sums, 1.0)
    sample_weight = origin_weight[owner]
    sample_weight /= sample_weight.mean()
    assert np.isfinite(sample_weight).all() and (sample_weight > 0).all()
    model = copy.deepcopy(joblib.load(base / '2023_station_no_pbl.joblib'))
    model.model = type(model.model)(**model.model.get_params())
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'weight': '1 / training origins in continuity block; expand via origin owner; mean expanded weight normalized to 1',
        'selection': 'single fixed candidate; require both main MAEs better and tail no worse than unweighted same-weather control',
        'limitations': ['block origin likelihood average not equal risk-hour mass', 'no inverse-risk-length weighting',
                       'development only; no deployment', 'all earlier source and label limits remain']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    pd.DataFrame({'timestamp': t.timestamp, 'continuity_block': t.continuity_block, 'origin_weight': origin_weight}).to_parquet(out / 'weights.parquet', index=False)
    model.model.fit(expanded, target, sample_weight=sample_weight)
    joblib.dump(model, out / 'model.joblib')
    pred = model.predict(z)
    v['block_weighted'] = pred
    allowed = v.score_allowed & np.isfinite(pred)
    candidate = metrics(v.loc[allowed], pred[allowed])
    baseline = metrics(v.loc[allowed], v.loc[allowed, 'station_no_pbl'].to_numpy())
    passed = bool(np.isfinite(pred).all() and candidate['row_mae'] < baseline['row_mae'] and
        candidate['block_mae'] < baseline['block_mae'] and candidate['tail_24h_block_mae'] <= baseline['tail_24h_block_mae'])
    v.to_parquet(out / 'predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_block_weight_development', 'candidate': candidate, 'baseline': baseline,
        'passes_gate': passed, 'unresolved': int(np.isnan(pred).sum()), 'training_blocks': len(block_sums),
        'expanded_rows': len(owner), 'input_unchanged': True,
        'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
