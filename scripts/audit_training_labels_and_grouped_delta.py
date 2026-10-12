"""Read-only label alignment and shared-truth audit of the frozen grouped trial."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'grouped_forecast_aft_2023_20260916'
    out = base / 'training_labels_grouped_delta_20260916'
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads((source / 'protocol.json').read_bytes())
    report = json.loads((source / 'report.json').read_bytes())
    inputs = dict(protocol['input_sha256'])
    for name, digest in report['output_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Invalid artifact name')
        inputs[str(source / name)] = digest
    paths = {2023: base / 'gefs_2021_extension_v1/f024_training_rows.parquet',
             2024: base / 'interval_weather_crossyear_v1/2024_original_training.parquet',
             2025: base / 'interval_weather_crossyear_v1/2025_original_training.parquet'}
    evidence = list(paths.values()) + [Path(__file__).resolve(), source / 'report.json',
                                      ROOT / 'src/haze_forecast/paired_interval_score.py']
    for path in evidence:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError(f'Frozen input changed: {path}')
        inputs[str(path)] = digest
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen input changed: {path}')
    labels = pd.read_parquet(base / 'interval_business_labels_v1/origins.parquet')
    if not labels.timestamp.is_unique:
        raise ValueError('Duplicate canonical label timestamp')
    labels = labels.set_index('timestamp')
    training = {}
    for year, path in paths.items():
        frame = pd.read_parquet(path)
        if not frame.timestamp.is_unique or not frame.timestamp.isin(labels.index).all():
            raise ValueError('Unmatched or duplicate training origin')
        canonical = labels.loc[frame.timestamp]
        mismatches = {}
        for name in ('continuity_block', 'lower_bound_hours', 'upper_bound_hours', 'exact_hours'):
            mismatches[name] = int((~np.isclose(frame[name], canonical[name],
                                              rtol=0, atol=0, equal_nan=True)).sum())
        mismatches['label_available_at'] = int((frame.label_available_at.to_numpy()
                                              != canonical.label_available_at.to_numpy()).sum())
        if any(mismatches.values()):
            raise ValueError(f'Training label mismatch: {year}: {mismatches}')
        cutoff = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
        if not (frame.timestamp.lt(cutoff).all() and frame.label_available_at.lt(cutoff).all()):
            raise ValueError('Immature training label')
        training[str(year)] = {'rows': len(frame), 'mismatches': mismatches}
    frame = pd.read_parquet(source / 'expanded.parquet')
    paired = paired_error_summary(frame, frame.forecast_aft, frame.candidate)
    # Independent piecewise absolute-error calculation at endpoints and kinks.
    lo = frame.lower_bound_hours.to_numpy()
    hi = frame.upper_bound_hours.to_numpy()
    a, b = frame.forecast_aft.to_numpy(), frame.candidate.to_numpy()
    truth = np.stack([lo, hi, np.clip(a, lo, hi), np.clip(b, lo, hi)])
    delta = np.abs(b - truth) - np.abs(a - truth)
    low, high = delta.min(axis=0), delta.max(axis=0)
    expected = [low.mean(), high.mean(),
                pd.Series(low).groupby(frame.continuity_block.to_numpy()).mean().mean(),
                pd.Series(high).groupby(frame.continuity_block.to_numpy()).mean().mean()]
    np.testing.assert_allclose(expected, [paired[k] for k in (
        'all_row_delta_lower', 'all_row_delta_upper',
        'all_block_delta_lower', 'all_block_delta_upper')], rtol=0, atol=1e-12)
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Input changed during audit: {path}')
    result = {'input_sha256': inputs, 'training_label_alignment': training,
              'expanded_paired_delta': paired, 'independent_formula_passed': True,
              'deployment_authorized': False,
              'limitations': ['Canonical agreement is not proof of physical label correctness',
                              'Shared-truth bounds are not confidence intervals',
                              'No new training; original rejection gate unchanged']}
    out.mkdir()
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'training': training, 'paired': paired}, indent=2))


if __name__ == '__main__':
    main()
