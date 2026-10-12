"""Training-only support reference for frozen 2018/2019 cohorts; no new predictor."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from haze_forecast.cohort_scorecard import score_predictions
from scripts.benchmark_observed_business_aft import ROOT, sha


def process_distances(reference, query, train_blocks, query_blocks):
    blocks = np.unique(train_blocks)
    positions = [np.flatnonzero(train_blocks == block) for block in blocks]
    result = np.empty((len(query), len(blocks)))
    for start in range(0, len(query), 128):
        stop = min(start + 128, len(query))
        distances = cdist(query[start:stop], reference, metric='sqeuclidean') / reference.shape[1]
        for j, indices in enumerate(positions):
            result[start:stop, j] = distances[:, indices].min(axis=1)
    result[np.asarray(query_blocks)[:, None] == blocks[None, :]] = np.inf
    return result, blocks


def main():
    source = ROOT / 'artifacts/task01_recovery_head/two_expert_aft_2018_2019_v1'
    out = ROOT / 'artifacts/task01_recovery_head/input_support_2018_2019_20260916'
    if out.exists():
        raise FileExistsError(out)
    names = json.loads((source / 'protocol.json').read_bytes())['features']
    files = [source / 'protocol.json', Path(__file__)]
    files += [source / f'{year}_{kind}.parquet' for year in (2018, 2019)
              for kind in ('training', 'predictions')]
    files += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in files}
    protocol = {
        'years': [2018, 2019], 'features': names, 'input_sha256': before,
        'distance': 'training-median imputed, training-IQR scaled mean square; minimum per process',
        'reference': 'training rows exclude their own process; median nearest distance per process',
        'radius': '90th percentile of training process reference, no validation-label selection',
        'purpose': 'diagnostic only, not a confidence score, prediction or routing rule',
        'deployment_authorized': False,
    }
    out.mkdir()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    reports = {}
    for year in (2018, 2019):
        train = pd.read_parquet(source / f'{year}_training.parquet')
        valid = pd.read_parquet(source / f'{year}_predictions.parquet')
        cut = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        assert train.timestamp.lt(cut).all() and train.label_available_at.lt(cut).all()
        assert not set(train.continuity_block) & set(valid.continuity_block)
        numeric = train[names].astype(float)
        med = numeric.median()
        scale = numeric.quantile(.75) - numeric.quantile(.25)
        kept = scale.index[scale.gt(0) & med.notna()].tolist()
        reference = ((train[kept].fillna(med) - med[kept]) / scale[kept]).to_numpy(float)
        query = ((valid[kept].fillna(med) - med[kept]) / scale[kept]).to_numpy(float)
        train_d, _ = process_distances(reference, reference, train.continuity_block.to_numpy(),
                                       train.continuity_block.to_numpy())
        block_reference = pd.Series(train_d.min(axis=1)).groupby(
            train.continuity_block.to_numpy()).median()
        radius = float(block_reference.quantile(.9))
        distances, _ = process_distances(reference, query, train.continuity_block.to_numpy(),
                                         valid.continuity_block.to_numpy())
        # Independently verify the vectorised distance for a fixed query/donor pair.
        first_block = np.unique(train.continuity_block)[0]
        check = np.square(reference[train.continuity_block.eq(first_block)] - query[0]).mean(axis=1).min()
        np.testing.assert_allclose(distances[0, 0], check, rtol=1e-12)
        rows = valid[['timestamp', 'continuity_block', 'lower_bound_hours', 'upper_bound_hours',
                      'exact_hours', 'temporal_prediction']].copy()
        rows['nearest_distance'] = distances.min(axis=1)
        rows['support_processes'] = (distances <= radius).sum(axis=1)
        rows['missing_fraction'] = valid[names].isna().mean(axis=1)
        rows['exact_absolute_error'] = abs(rows.temporal_prediction - rows.exact_hours)
        groups = {}
        for name, mask in [('all', np.ones(len(rows), bool)),
                           ('no_near_process', rows.support_processes.eq(0)),
                           ('has_near_process', rows.support_processes.gt(0))]:
            frame = rows.loc[mask]
            groups[name] = score_predictions(frame, frame.temporal_prediction) if len(frame) else None
        blocks = rows.groupby('continuity_block').agg(
            rows=('timestamp', 'size'), exact_rows=('exact_hours', 'count'),
            mae=('exact_absolute_error', 'mean'), nearest=('nearest_distance', 'median'),
            support_processes=('support_processes', 'median'), missing=('missing_fraction', 'mean'))
        feature_coverage = pd.DataFrame({'train_present': train[names].notna().sum(),
                                         'valid_present': valid[names].notna().sum()})
        feature_coverage.to_csv(out / f'{year}_coverage.csv')
        rows.to_parquet(out / f'{year}_origins.parquet', index=False)
        blocks.to_csv(out / f'{year}_blocks.csv')
        reports[year] = {
            'distance_features': len(kept), 'all_missing_train_features': med.index[med.isna()].tolist(),
            'radius': radius, 'groups': groups,
            'row_distance_error_spearman': float(rows.nearest_distance.corr(rows.exact_absolute_error, method='spearman')),
            'block_distance_error_spearman': float(blocks.nearest.corr(blocks.mae, method='spearman')),
        }
    assert before == {str(p): sha(p) for p in files}
    report = {'years': reports, 'inputs_unchanged': True, 'model_changed': False,
              'limitations': ['Previously inspected historical years, not untouched evaluation',
                              'Distance and imputation are conventions, not physical similarity',
                              'Related lag columns are not independent evidence',
                              'No distance threshold selected using validation outcomes',
                              'Association cannot distinguish causality or irreducible uncertainty']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
