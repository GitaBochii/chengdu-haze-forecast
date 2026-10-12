"""Frozen interval hazard algorithm refit before 2023, paired development gates."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.interval_hazard_em import IntervalHazardEM
from scripts.benchmark_gefs_72h_aft import compare_cohort, verify_hashes
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    started = time.monotonic()
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_2021_extension_v1'
    pilot = base / 'interval_hazard_temporal_v1'
    out = base / 'interval_hazard_development_v1'
    if out.exists():
        raise FileExistsError(out)
    pilot_report = json.loads((pilot / 'report.json').read_bytes())
    if pilot_report.get('pilot_passed') is not True:
        raise ValueError('Earlier temporal pilot must pass before development')
    settings = json.loads((pilot / 'protocol.json').read_bytes())
    inputs = {str(Path(__file__)): sha(Path(__file__))}
    for folder in (source, pilot):
        report = json.loads((folder / 'report.json').read_bytes())
        inputs.update({str(folder / n): h for n, h in report['output_sha256'].items()})
        inputs[str(folder / 'report.json')] = sha(folder / 'report.json')
        inputs.update(json.loads((folder / 'protocol.json').read_bytes())['input_sha256'])
    verify_hashes(inputs)
    names = settings['features']
    train = pd.read_parquet(source / 'training_rows.parquet')
    val = pd.read_parquet(source / 'inference_rows.parquet')
    frames = {key: pd.read_parquet(source / filename) for key, filename in
              [('original', 'predictions.parquet'), ('expanded', 'expanded_predictions.parquet')]}
    cutoff = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp('2024-01-01', tz='Asia/Shanghai')
    if not (train.timestamp.lt(cutoff).all() and train.label_available_at.lt(cutoff).all()
            and val.timestamp.ge(cutoff).all() and val.timestamp.lt(end).all()):
        raise ValueError('Training or inference violates annual time split')
    for frame in frames.values():
        pd.testing.assert_series_equal(frame.timestamp, val.timestamp)
        if set(train.continuity_block) & set(frame.continuity_block):
            raise ValueError('Overlapping training and inference blocks')
        if not frame.loc[frame.score_allowed, 'label_available_at'].lt(end).all():
            raise ValueError('Scored labels must mature inside development year')
    reference = joblib.load(source / 'model.joblib').predict(val[names])
    for frame in frames.values():
        np.testing.assert_array_equal(reference, frame.extended_2021_aft)
    out.mkdir()
    protocol = {'input_sha256': inputs, 'features': names,
                'rounds': settings['rounds'], 'max_em_steps': settings['max_em_steps'],
                'training_rows': len(train), 'inference_rows': len(val),
                'gate': 'Original and expanded row/block improve and tail not worse versus BOTH f024 and extension AFT.',
                'unresolved': 'Any unresolved prediction fails; never drop or cap.',
                'selection': 'Frozen earlier pilot algorithm, no 2023 tuning', 'deployment_authorized': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = IntervalHazardEM(rounds=settings['rounds'], max_em_steps=settings['max_em_steps']).fit_bounds(
        train[names], train.lower_bound_hours, train.upper_bound_hours)
    joblib.dump(model, out / 'model.joblib')
    prediction = model.predict(val[names])
    np.testing.assert_allclose(prediction, joblib.load(out / 'model.joblib').predict(val[names]), equal_nan=True)
    resolved = bool(np.isfinite(prediction).all())
    cohorts = {}
    for key, frame in frames.items():
        if resolved:
            original = compare_cohort(frame, prediction)
            paired = frame.copy()
            paired['forecast_aft'] = paired.extended_2021_aft
            extension = compare_cohort(paired, prediction)
            cohorts[key] = {'against_original': original, 'against_extension': extension,
                            'passed': original['passed'] and extension['passed']}
        else:
            cohorts[key] = {'passed': False, 'reason': 'unresolved prediction; no partial scoring'}
        frame['interval_hazard'] = prediction
        frame.to_parquet(out / f'{key}_predictions.parquet', index=False)
    verify_hashes(inputs)
    report = {'cohorts': cohorts, 'passed': all(v['passed'] for v in cohorts.values()),
              'unresolved_rows': int((~np.isfinite(prediction)).sum()), 'horizon': model.horizon,
              'trace': model.trace, 'elapsed_seconds': time.monotonic() - started,
              'deployment_authorized': False,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
