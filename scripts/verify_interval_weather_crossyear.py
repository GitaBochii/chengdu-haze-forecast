"""Independent read-only acceptance of the frozen annual weather experiment."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/task01_recovery_head'
OUT = BASE / 'interval_weather_crossyear_v1'


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_metrics(frame, values):
    values = pd.Series(values, index=frame.index)
    exact = frame.exact_hours.notna()
    errors = (frame.exact_hours - values).abs().loc[exact]
    blocks = frame.loc[exact, 'continuity_block']
    tail = frame.loc[exact, 'exact_hours'].ge(24)
    censored = ~exact & frame.informative
    return {
        'exact_rows': int(exact.sum()),
        'exact_blocks': int(blocks.nunique()),
        'row_mae': float(errors.mean()),
        'block_mae': float(errors.groupby(blocks).mean().mean()),
        'tail_24h_rows': int(tail.sum()),
        'tail_24h_block_mae': float(errors.loc[tail].groupby(blocks.loc[tail]).mean().mean()),
        'informative_censored_rows': int(censored.sum()),
        'below_censor_bound_fraction': float(values.loc[censored].lt(
            frame.loc[censored, 'lower_bound_hours']).mean()),
    }


def main():
    protocol = json.loads((OUT / 'protocol.json').read_bytes())
    report = json.loads((OUT / 'report.json').read_bytes())
    for name, expected in protocol['input_sha256'].items():
        assert fingerprint(name) == expected, name
    for name, expected in report['output_sha256'].items():
        assert fingerprint(OUT / name) == expected, name
    labels = pd.read_parquet(BASE / 'interval_business_labels_v1/origins.parquet').set_index('timestamp')
    full = pd.read_parquet(BASE / 'observed_business_dataset_v1/features.parquet',
                           columns=['timestamp', 'continuity_block'])
    fields = protocol['replaced_fields']
    names = protocol['features']
    assert len(names) == len(set(names)) == 104
    assert len(fields) == 25
    stale = {}
    boundary_rows = {}
    sensitivity = {}
    for year in (2024, 2025):
        start = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        end = pd.Timestamp(f'{year + 1}-01-01', tz='Asia/Shanghai')
        crossing = set()
        for cutoff in (start, end):
            crossing |= set(full.loc[full.timestamp.lt(cutoff), 'continuity_block']) & set(
                full.loc[full.timestamp.ge(cutoff), 'continuity_block'])
        original_train = pd.read_parquet(BASE / 'original_gefs_without_pbl_crossyear_v1' /
                                         f'{year}_training_rows.parquet')
        original_x = pd.read_parquet(BASE / 'reusable_gefs_crossyear_v1' / f'{year}_features.parquet')
        origins = pd.read_parquet(BASE / 'reusable_gefs_crossyear_v1' / f'{year}_predictions.parquet')
        original_x['timestamp'] = origins.timestamp.to_numpy()
        assert original_train.timestamp.is_unique and origins.timestamp.is_unique
        assert original_train.timestamp.lt(start).all()
        assert original_train.label_available_at.lt(start).all()
        assert not set(original_train.continuity_block) & set(origins.continuity_block)
        true_labels = labels.loc[origins.timestamp]
        assert true_labels.label_available_at.lt(end).all()
        assert not set(original_train.continuity_block) & crossing
        end_cross = set(full.loc[full.timestamp.lt(end), 'continuity_block']) & set(
            full.loc[full.timestamp.ge(end), 'continuity_block'])
        assert not set(true_labels.continuity_block) & end_cross
        boundary_rows[year] = int(true_labels.continuity_block.isin(crossing).sum())
        sensitivity[year] = {}
        assert origins.timestamp.ge(start).all() and origins.timestamp.lt(end).all()
        stale[year] = int((true_labels.label_available_at.to_numpy() !=
                           origins.label_available_at.to_numpy()).sum())
        for arm in ('original', 'uniform', 'masked'):
            train = pd.read_parquet(OUT / f'{year}_{arm}_training.parquet')
            x = pd.read_parquet(OUT / f'{year}_{arm}_inference.parquet')
            for candidate, reference in ((train, original_train), (x, original_x)):
                untouched = [c for c in reference if c not in fields]
                pd.testing.assert_frame_equal(candidate[untouched], reference[untouched])
                if arm == 'original':
                    pd.testing.assert_frame_equal(candidate, reference)
                elif arm == 'masked':
                    pd.testing.assert_frame_equal(candidate[fields].isna(), reference[fields].isna())
                else:
                    assert candidate[fields].notna().all().all()
            values = joblib.load(OUT / f'{year}_{arm}.joblib').predict(x[names])
            assert np.isfinite(values).all()
            by_time = pd.Series(values, index=x.timestamp)
            for cohort in ('predictions', 'expanded', 'common'):
                saved = pd.read_parquet(OUT / f'{year}_{arm}_{cohort}.parquet')
                old = pd.read_parquet(BASE / 'reusable_gefs_crossyear_v1' / f'{year}_{cohort}.parquet')
                pd.testing.assert_frame_equal(saved[old.columns], old)
                np.testing.assert_array_equal(by_time.loc[saved.timestamp], saved.weather_candidate)
                if cohort != 'predictions':
                    for c in ('exact_hours', 'lower_bound_hours', 'upper_bound_hours', 'informative'):
                        np.testing.assert_array_equal(saved[c], labels.loc[saved.timestamp, c])
                calculated = independent_metrics(saved, saved.weather_candidate)
                for metric, value in calculated.items():
                    np.testing.assert_allclose(value, report['years'][str(year)]['scores'][arm][cohort][metric],
                                               rtol=1e-12, atol=1e-12, equal_nan=True)
                if cohort == 'common':
                    clean = saved.loc[~saved.continuity_block.isin(crossing)]
                    sensitivity[year][arm] = independent_metrics(clean, clean.weather_candidate)
                    sensitivity[year]['page'] = independent_metrics(clean, clean.strict_prediction_hours)
                    for metric, value in independent_metrics(saved, saved.strict_prediction_hours).items():
                        np.testing.assert_allclose(value, report['years'][str(year)]['page_common'][metric],
                                                   rtol=1e-12, atol=1e-12, equal_nan=True)
    print(json.dumps({'status': 'PASS', 'input_hashes': len(protocol['input_sha256']),
                      'output_hashes': len(report['output_sha256']), 'model_replays': 6,
                      'cohort_metric_sets': 18, 'stale_legacy_label_available_at': stale,
                      'validation_start_boundary_rows': boundary_rows,
                      'common_without_crossyear_blocks': sensitivity,
                      'deployment_authorized': False}, indent=2))


if __name__ == '__main__':
    main()
