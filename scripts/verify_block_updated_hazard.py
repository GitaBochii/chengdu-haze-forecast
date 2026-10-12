"""Independently reconstruct every block training set and replay saved models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    root = ROOT / 'artifacts/task01_recovery_head'
    out = root / 'block_updated_hazard_2025_v1'
    q = json.loads((out / 'protocol.json').read_bytes())
    r = json.loads((out / 'report.json').read_bytes())
    assert all(sha(Path(p)) == h for p, h in q['input_sha256'].items())
    assert sha(out / 'predictions.parquet') == r['predictions_sha256']
    source = pd.read_parquet(root / 'observed_business_dataset_v1/origins.parquet')
    fix = pd.read_parquet(root / 'ghcnh_weather_repair_v1/repaired_features.parquet')
    source.loc[source.timestamp.dt.year.eq(2025), q['features']] = fix[q['features']].to_numpy()
    full = pd.read_parquet(root / 'observed_business_dataset_v1/features.parquet')
    starts = full.groupby('continuity_block').timestamp.min()
    aggregate = pd.read_parquet(out / 'predictions.parquet')
    count = 0
    for manifest in r['manifests']:
        block = manifest['block']
        dest = out / str(block)
        cutoff = starts.loc[block]
        assert pd.Timestamp(manifest['cutoff']) == cutoff
        for file, digest in manifest['hashes'].items():
            assert sha(dest / file) == digest
        expected = source.loc[source.informative & source.timestamp.lt(cutoff)
                              & source.label_available_at.lt(cutoff) & source.continuity_block.lt(block)]
        training = pd.read_parquet(dest / 'training_rows.parquet')
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), training.reset_index(drop=True))
        val = pd.read_parquet(dest / 'predictions.parquet')
        expected_val = source.loc[source.timestamp.dt.year.eq(2025) & source.continuity_block.eq(block)]
        pd.testing.assert_frame_equal(val[source.columns].reset_index(drop=True), expected_val.reset_index(drop=True))
        for name, column in [('hazard', 'block_hazard'), ('exact_aft', 'block_exact_aft')]:
            model = joblib.load(dest / f'{name}.joblib')
            np.testing.assert_array_equal(model.predict(val[q['features']]), val[column])
            np.testing.assert_array_equal(aggregate.set_index('timestamp').loc[val.timestamp, column], val[column])
        count += len(val)
    assert count == len(aggregate) == int(source.timestamp.dt.year.eq(2025).sum())
    for column, recorded in r['scores'].items():
        exact = aggregate.loc[aggregate.block_hazard.notna() & aggregate.exact_hours.notna()]
        error = (exact[column] - exact.exact_hours).abs()
        tail = exact.exact_hours.ge(24)
        actual = [error.mean(), error.groupby(exact.continuity_block).mean().mean(),
                  error[tail].groupby(exact.loc[tail, 'continuity_block']).mean().mean()]
        np.testing.assert_allclose(actual, [recorded[k] for k in ('row_mae', 'block_mae', 'tail_24h_block_mae')], rtol=0, atol=1e-12)
    result = {'status': 'PASS', 'blocks': len(r['manifests']), 'replayed_models': 2 * len(r['manifests']),
        'all_origins': count, 'report_sha256': sha(out / 'report.json'), 'verifier_sha256': sha(Path(__file__)),
        'checks': ['all_training_membership', 'label_availability', 'block_disjointness', 'all_prediction_replays',
                   'all_population_coverage', 'main_and_tail_metrics', 'recorded_hashes']}
    (out / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
