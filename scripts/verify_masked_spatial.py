"""Independent saved-curve, temporal-boundary and metric verification."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_regional_curves import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    out = base / 'masked_spatial_training_20260916'
    destination = out / 'independent_verification.json'
    if destination.exists():
        raise FileExistsError(destination)
    report = json.loads((out / 'report.json').read_bytes())
    for path, expected in report['input_sha256'].items():
        assert digest(Path(path)) == expected, path
    results = {}
    for year in (2022, 2023):
        folder = out / str(year)
        previous = base / f'regional_rolling_{year}_20260916'
        result = report['years'][str(year)]
        for name, expected in result['output_sha256'].items():
            assert digest(folder / name) == expected, name
        trace = json.loads((folder / 'masked_airformer_training.json').read_bytes())
        assert all(row['selection_year'] == year - 1 for row in trace)
        assert min(trace, key=lambda row: row['validation_loss'])['epoch'] == result['best_epoch']
        for filename, cutoff in (('training_windows.parquet', year-1), ('selection_windows.parquet', year)):
            timestamps = pd.read_parquet(previous / filename).timestamp
            assert (timestamps + pd.Timedelta(hours=72) < pd.Timestamp(f'{cutoff}-01-01', tz='Asia/Shanghai')).all()
        original = pd.read_parquet(previous / 'predictions.parquet')
        frame = pd.read_parquet(folder / 'predictions.parquet')
        pd.testing.assert_frame_equal(frame[original.columns], original)
        with np.load(folder / 'city_curves.npz') as curves:
            np.testing.assert_array_equal(curves['origin_ns'], pd.DatetimeIndex(frame.timestamp).asi8)
            hours = []
            for curve in curves['prediction'][:, :, 1]:
                assert np.isfinite(curve).all()
                candidates = [i+1 for i in range(67) if (curve[i:i+6] <= 100).all()]
                hours.append(candidates[0] if candidates else 68)
        np.testing.assert_array_equal(hours, frame.masked_airformer)
        scored = frame.loc[frame.score_mature]
        exact = scored.loc[scored.exact_hours.notna()]
        error = abs(exact.masked_airformer - exact.exact_hours)
        tail = exact.exact_hours.ge(24)
        calculated = [error.mean(), error.groupby(exact.continuity_block).mean().mean(),
                      error[tail].groupby(exact.continuity_block[tail]).mean().mean()]
        np.testing.assert_allclose(calculated, [result['scores'][key] for key in (
            'exact_row_mae', 'exact_block_mae', 'tail24_block_mae')], rtol=0, atol=1e-12)
        for name, reference in (('versus_original', 'airformer'), ('versus_aft', 'matched_aft')):
            a, b = scored[reference].to_numpy(), scored.masked_airformer.to_numpy()
            lo, hi = scored.lower_bound_hours.to_numpy(), scored.upper_bound_hours.to_numpy()
            truths = np.stack([lo, hi, np.clip(a, lo, hi), np.clip(b, lo, hi)])
            difference = abs(b-truths) - abs(a-truths)
            low, high = difference.min(axis=0), difference.max(axis=0)
            values = [low.mean(), high.mean(),
                      pd.Series(low).groupby(scored.continuity_block.to_numpy()).mean().mean(),
                      pd.Series(high).groupby(scored.continuity_block.to_numpy()).mean().mean()]
            np.testing.assert_allclose(values, [result[name][key] for key in (
                'all_row_delta_lower', 'all_row_delta_upper', 'all_block_delta_lower', 'all_block_delta_upper')],
                rtol=0, atol=1e-12)
        results[str(year)] = {'passed': True, 'origins': len(frame), 'scored': len(scored), 'independent_metrics': 11}
    destination.write_text(json.dumps({'folds': results, 'verifier_sha256': digest(Path(__file__).resolve()),
        'scope': 'Saved curves and scores; checkpoint replay separately performed by training runner'}, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
