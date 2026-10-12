"""2025 block-start updates with explicit mature-label manifests."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor
from haze_forecast.observed_hazard import ObservedHazardRegressor
from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    root = ROOT / 'artifacts/task01_recovery_head'
    data = root / 'observed_business_dataset_v1'
    repair = root / 'ghcnh_weather_repair_v1'
    out = root / 'block_updated_hazard_2025_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [data / n for n in ('origins.parquet', 'features.parquet', 'report.json')]
    paths += [repair / 'repaired_features.parquet', repair / 'predictions.parquet',
        ROOT / 'artifacts/strict_history_backtest/predictions.parquet', Path(__file__),
        ROOT / 'scripts/benchmark_observed_business_aft.py']
    paths += [ROOT / f'src/haze_forecast/{n}.py' for n in
        ('observed_hazard', 'discrete_survival', 'observed_aft', 'training', 'app')]
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    hashes = {str(p): sha(p) for p in paths}
    names = json.loads((data / 'report.json').read_bytes())['feature_names']
    origins = pd.read_parquet(data / 'origins.parquet')
    full = pd.read_parquet(data / 'features.parquet')
    repaired = pd.read_parquet(repair / 'repaired_features.parquet')
    reference = pd.read_parquet(repair / 'predictions.parquet')
    mask = origins.timestamp.dt.year.eq(2025)
    np.testing.assert_array_equal(origins.loc[mask, 'timestamp'], reference.timestamp)
    origins.loc[mask, names] = repaired[names].to_numpy()
    starts = full.groupby('continuity_block').timestamp.min()
    page = pd.read_parquet(ROOT / 'artifacts/strict_history_backtest/predictions.parquet')
    page = page[['timestamp', 'strict_prediction_hours']]
    validation = origins.loc[mask].copy()
    protocol = {'input_sha256': hashes, 'features': names, 'year': 2025,
        'update': 'once before continuity block start; all train timestamps and label availability strictly earlier',
        'train_membership': 'informative observed origins; current and later blocks excluded',
        'models': {'hazard': 'LightGBM observed bounds, 300 trees seed42',
                   'exact_aft': 'exact observed targets, XGBoost normal scale1.2 500 rounds seed42'},
        'selection': 'none; 2025 diagnostic bottleneck, not pristine holdout',
        'limitations': ['GHCNh archive timestamps do not prove realtime availability',
            'page updates old events, new models update conservative blocks; not identical event definitions',
            'exact MAE excludes censored outcomes; no fabricated unresolved medians']}
    out.mkdir(parents=True)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    pieces, manifests = [], []
    begun = time.monotonic()
    for block, val in validation.groupby('continuity_block', sort=True):
        cutoff = starts.loc[block]
        train = origins.loc[origins.timestamp.lt(cutoff) & origins.label_available_at.lt(cutoff)
                            & origins.continuity_block.lt(block) & origins.informative].copy()
        assert len(train) and not set(train.continuity_block) & set(val.continuity_block)
        exact = train.loc[train.exact_hours.notna()]
        dest = out / str(block)
        dest.mkdir()
        train.to_parquet(dest / 'training_rows.parquet', index=False)
        model = ObservedHazardRegressor(seed=42).fit_bounds(train[names], train.lower_bound_hours, train.upper_bound_hours)
        joblib.dump(model, dest / 'hazard.joblib')
        val = val.copy()
        val['block_hazard'] = model.predict(val[names])
        aft = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(exact[names], exact.lower_bound_hours, exact.upper_bound_hours)
        joblib.dump(aft, dest / 'exact_aft.joblib')
        val['block_exact_aft'] = aft.predict(val[names])
        val.to_parquet(dest / 'predictions.parquet', index=False)
        manifests.append({'block': int(block), 'cutoff': str(cutoff), 'training_rows': len(train),
            'exact_training_rows': len(exact), 'prediction_rows': len(val),
            'last_train_timestamp': str(train.timestamp.max()),
            'last_label_available_at': str(train.label_available_at.max()),
            'hashes': {p.name: sha(p) for p in dest.iterdir() if p.is_file()}})
        pieces.append(val)
        print(json.dumps({'block_completed': int(block), 'blocks_done': len(pieces),
                          'elapsed_seconds': round(time.monotonic() - begun, 1)}), flush=True)
    predictions = pd.concat(pieces).sort_values('timestamp').reset_index(drop=True)
    predictions = predictions.merge(page, on='timestamp', how='left', validate='one_to_one')
    predictions = predictions.merge(reference[['timestamp', 'hazard_after']], on='timestamp', validate='one_to_one')
    assert len(predictions) == len(validation) and predictions.strict_prediction_hours.notna().all()
    ok = predictions.block_hazard.notna()
    scores = {n: metrics(predictions.loc[ok], predictions.loc[ok, n].to_numpy()) for n in
        ('block_hazard', 'block_exact_aft', 'strict_prediction_hours', 'hazard_after')}
    predictions.to_parquet(out / 'predictions.parquet', index=False)
    assert hashes == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_block_update', 'manifests': manifests, 'scores': scores,
        'unresolved_hazard_origins': int((~ok).sum()), 'origins': len(predictions),
        'elapsed_seconds': time.monotonic() - begun, 'inputs_unchanged': True,
        'predictions_sha256': sha(out / 'predictions.parquet')}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'scores': scores, 'origins': len(predictions), 'elapsed': report['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
