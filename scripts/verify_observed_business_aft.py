"""Independent fixed-year AFT replay and boundary sensitivity audit."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    directory = Path('artifacts/task01_recovery_head/observed_business_aft_v1')
    report = json.loads((directory / 'report.json').read_bytes())
    protocol = json.loads((directory / 'protocol.json').read_bytes())
    validation = pd.read_parquet(directory / 'predictions.parquet')
    train = pd.read_parquet(directory / 'training_rows.parquet')
    for path, value in protocol['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
    for path, value in report['output_sha256'].items():
        assert hashlib.sha256((directory / path).read_bytes()).hexdigest() == value
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    assert train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
    assert train.informative.all()
    full = pd.read_parquet('artifacts/task01_recovery_head/observed_business_dataset_v1/features.parquet')
    bounds = full.groupby('continuity_block').timestamp.agg(['min', 'max'])
    end = pd.Timestamp('2024-01-01', tz='Asia/Shanghai')
    allowed_blocks = bounds.index[(bounds['min'] >= cutoff) & (bounds['max'] < end)]
    allowed = validation.continuity_block.isin(allowed_blocks) & validation.label_available_at.lt(end)
    np.testing.assert_array_equal(allowed, validation.score_allowed)
    result = {'status': 'PASS', 'all_exact_diagnostic': {}, 'excluded_blocks': {}}
    for block, rows in validation.loc[~allowed].groupby('continuity_block'):
        result['excluded_blocks'][str(block)] = {'rows': len(rows), 'start': str(rows.timestamp.min()),
            'end': str(rows.timestamp.max()), 'grid_block_start': str(bounds.loc[block, 'min']),
            'grid_block_end': str(bounds.loc[block, 'max'])}
    for arm in protocol['arms']:
        prediction = joblib.load(directory / f'{arm}.joblib').predict(validation[protocol['features']])
        np.testing.assert_array_equal(prediction, validation[arm])
        scored = validation.loc[allowed & validation.exact_hours.notna()]
        error = (scored[arm] - scored.exact_hours).abs()
        tail = scored.exact_hours.ge(24)
        censored = validation.loc[allowed & validation.exact_hours.isna() & validation.informative]
        actual = [error.mean(), error.groupby(scored.continuity_block).mean().mean(),
                  error[tail].groupby(scored.loc[tail, 'continuity_block']).mean().mean(),
                  censored[arm].lt(censored.lower_bound_hours).mean()]
        expected = [report['scores'][arm][key] for key in ('row_mae', 'block_mae', 'tail_24h_block_mae', 'below_censor_bound_fraction')]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        all_exact = validation.loc[validation.exact_hours.notna()]
        all_error = (all_exact[arm] - all_exact.exact_hours).abs()
        result['all_exact_diagnostic'][arm] = {'rows': len(all_exact),
            'row_mae': float(all_error.mean()),
            'block_mae': float(all_error.groupby(all_exact.continuity_block).mean().mean())}
    result['report_sha256'] = hashlib.sha256((directory / 'report.json').read_bytes()).hexdigest()
    result['verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory / 'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
