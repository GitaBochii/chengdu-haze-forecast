"""Re-evaluate frozen annual weather arms without training or cohort selection."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_cohort(labels, saved, year):
    if not labels.timestamp.is_unique or not saved.timestamp.is_unique:
        raise ValueError('Duplicate origin timestamps')
    if not saved.timestamp.isin(labels.timestamp).all():
        raise ValueError('Origin absent from canonical labels')
    columns = ['timestamp', 'continuity_block', 'lower_bound_hours', 'upper_bound_hours',
               'exact_hours', 'label_available_at']
    result = labels.set_index('timestamp').loc[saved.timestamp].reset_index()[columns]
    for column in columns[1:-1]:
        np.testing.assert_allclose(result[column], saved[column], equal_nan=True)
    start = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp(f'{year + 1}-01-01', tz='Asia/Shanghai')
    if not (result.timestamp.ge(start).all() and result.timestamp.lt(end).all()
            and result.label_available_at.lt(end).all()):
        raise ValueError('Invalid annual cohort or immature label')
    stale = int((result.label_available_at.to_numpy() != saved.label_available_at.to_numpy()).sum())
    return result, stale


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'interval_weather_crossyear_v1'
    out = base / 'unified_scorecard_crossyear_v1'
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((source / 'protocol.json').read_bytes())
    old_report = json.loads((source / 'report.json').read_bytes())
    inputs = {Path(k): v for k, v in protocol['input_sha256'].items()}
    for name, value in old_report['output_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Invalid frozen filename')
        inputs[source / name] = value
    # UI changes do not participate in replay. Preserve their drift explicitly.
    ui_path = ROOT / 'src/haze_forecast/app.py'
    ui_drift = {}
    for path, expected in inputs.items():
        actual = sha(path)
        if actual != expected:
            if path == ui_path:
                ui_drift[str(path)] = {'frozen': expected, 'current': actual,
                                      'reason': 'Authorized snapshot metric attribution fix'}
            else:
                raise ValueError(f'Frozen dependency changed: {path}')
    label_path = base / 'interval_business_labels_v1/origins.parquet'
    label_report = json.loads((label_path.parent / 'report.json').read_bytes())
    if sha(label_path) != label_report['output_sha256']:
        raise ValueError('Canonical label hash mismatch')
    labels = pd.read_parquet(label_path)
    results, outputs, model_hashes, incompatible = {}, {}, {}, {}
    for year in (2024, 2025):
        results[str(year)] = {}
        legacy = pd.read_parquet(source / f'{year}_original_predictions.parquet')
        canonical = labels.set_index('timestamp').loc[legacy.timestamp]
        incompatible[str(year)] = {
            'cohort': 'predictions', 'status': 'not_scored_label_version_mismatch',
            'rows': len(legacy),
            'mismatches': {c: int((~np.isclose(canonical[c], legacy[c], equal_nan=True)).sum())
                           for c in ['lower_bound_hours', 'upper_bound_hours', 'exact_hours']},
        }
        if not any(incompatible[str(year)]['mismatches'].values()):
            raise ValueError('Legacy cohort compatibility changed; review protocol')
        reference = {}
        for arm in ('original', 'uniform', 'masked'):
            checkpoint = source / f'{year}_{arm}.joblib'
            model_hashes[f'{year}_{arm}'] = sha(checkpoint)
            inference = pd.read_parquet(source / f'{year}_{arm}_inference.parquet')
            if not inference.timestamp.is_unique:
                raise ValueError('Duplicate inference origin')
            prediction = joblib.load(checkpoint).predict(inference[protocol['features']])
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite prediction; no origin may be dropped')
            indexed = pd.Series(prediction, index=inference.timestamp)
            for key in ('expanded', 'common'):
                saved = pd.read_parquet(source / f'{year}_{arm}_{key}.parquet')
                cohort, stale = canonical_cohort(labels, saved, year)
                if key in reference:
                    pd.testing.assert_frame_equal(cohort, reference[key])
                else:
                    reference[key] = cohort
                aligned = indexed.loc[cohort.timestamp].to_numpy()
                np.testing.assert_array_equal(aligned, saved.weather_candidate)
                scores = score_predictions(cohort, aligned)
                old = old_report['years'][str(year)]['scores'][arm][key]
                for new, prior in [('exact_row_mae', 'row_mae'), ('exact_block_mae', 'block_mae'),
                                   ('tail24_block_mae', 'tail_24h_block_mae')]:
                    np.testing.assert_allclose(scores[new], old[prior], rtol=0, atol=1e-12)
                results[str(year)].setdefault(key, {})[arm] = {
                    'scores': scores, 'stale_maturity_metadata_rows': stale}
                scored = cohort.copy()
                scored['prediction'] = aligned
                outputs[f'{year}_{arm}_{key}.parquet'] = scored
    evidence = list(inputs) + [label_path, label_path.parent / 'report.json',
                               source / 'report.json', Path(__file__).resolve(),
                               ROOT / 'src/haze_forecast/cohort_scorecard.py']
    manifest = {
        'input_sha256': {str(p): sha(p) for p in evidence}, 'model_sha256': model_hashes,
        'features': protocol['features'], 'historical_ui_drift': ui_drift,
        'training_policy': protocol['training'],
        'cohort_policy': 'Expanded and common cohorts; original cohort rejected for label incompatibility',
        'incompatible_cohorts': incompatible,
        'limitations': ['Reused historical years; not untouched holdout',
                        'ERA5 offline only; no new availability evidence',
                        'Finite interval error bounds are not confidence intervals',
                        'No webpage comparison; annual cohorts are not full hourly populations',
                        '2023 additional-2021-forecast claim in source protocol was corrected by erratum'],
        'deployment_authorized': False,
    }
    out.mkdir()
    for filename, frame in outputs.items():
        frame.to_parquet(out / filename, index=False)
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for path, expected in manifest['input_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Input changed during evaluation: {path}')
    result = {'years': results, 'output_sha256': {p.name: sha(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({year: value['common'] for year, value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
