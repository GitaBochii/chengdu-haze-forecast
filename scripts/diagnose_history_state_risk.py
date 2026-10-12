"""Describe frozen probability shifts without selecting a new model."""

import json

import numpy as np
import pandas as pd

from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.train_recovery_weather_expanded import hashes_unchanged
from scripts.train_recovery_weather_formal import sha
from scripts.verify_recovery_auxiliary import probabilities


def main():
    run = ROOT / 'artifacts/task01_recovery_head/history_state_residual_v1'
    output = ROOT / 'artifacts/task01_recovery_head/history_state_risk_diagnosis_v1'
    assert not output.exists()
    report = json.loads((run / 'report.json').read_bytes())
    verified = json.loads((run / 'independent_verification.json').read_bytes())
    assert verified['verification'] == 'PASS'
    assert hashes_unchanged(report['source_sha256'])
    arms = ('linear_covered', 'linear_all', 'nonlinear_all')
    files = [run / name for name in ('report.json', 'independent_verification.json')]
    files += [run / arm / name for arm in arms for name in ('logits.npy', 'validation.parquet')]
    files.append(ROOT / 'scripts/diagnose_history_state_risk.py')
    before = {str(path): sha(path) for path in files}
    base = pd.read_parquet(run / 'linear_covered/validation.parquet')
    base_logits = np.load(run / 'linear_covered/logits.npy', allow_pickle=False)
    base_probability = probabilities(base_logits)
    base_cdf = base_probability.cumsum(1)
    selected = base.label_verified_in_scoring_window & base.actual_hours.notna()
    results, tables = {}, []
    for arm in arms:
        frame = pd.read_parquet(run / arm / 'validation.parquet')
        pd.testing.assert_frame_equal(frame[base.columns.difference([
            'predicted_hours', 'probability_recovered_by_67h'])],
            base[base.columns.difference(['predicted_hours', 'probability_recovered_by_67h'])])
        logits = np.load(run / arm / 'logits.npy', allow_pickle=False)
        probability = probabilities(logits)
        cdf = probability.cumsum(1)
        np.testing.assert_allclose(probability.sum(1), 1., atol=1e-12)
        median = (cdf < .5).sum(1).clip(max=67) + 1
        np.testing.assert_array_equal(median, frame.predicted_hours)
        detail = frame.copy()
        detail['arm'] = arm
        detail['baseline_hours'] = base.predicted_hours
        detail['median_shift_hours'] = median - base.predicted_hours
        detail['capped_mean_hours'] = probability @ np.arange(1, 69)
        detail['baseline_capped_mean_hours'] = base_probability @ np.arange(1, 69)
        detail['tail_mass'] = probability[:, -1]
        detail['baseline_tail_mass'] = base_probability[:, -1]
        detail['mean_logit_shift'] = (logits - base_logits).mean(1)
        detail['positive_logit_fraction'] = (logits > base_logits).mean(1)
        detail['cdf_uniformly_higher'] = (cdf[:, :67] >= base_cdf[:, :67] - 1e-10).all(1)
        detail['cdf_uniformly_lower'] = (cdf[:, :67] <= base_cdf[:, :67] + 1e-10).all(1)
        for hour in (6, 12, 24, 48, 67):
            detail[f'cdf_{hour}h'] = cdf[:, hour - 1]
            detail[f'baseline_cdf_{hour}h'] = base_cdf[:, hour - 1]
        detail['capped_absolute_error'] = abs(np.minimum(frame.actual_hours, 68) - median)
        detail['baseline_capped_absolute_error'] = abs(np.minimum(base.actual_hours, 68) - base.predicted_hours)
        tables.append(detail)
        by_event = []
        for event, group in detail.loc[selected].groupby('event_id'):
            by_event.append({'event_id': int(event), 'rows': len(group),
                'mae': float(group.capped_absolute_error.mean()),
                'baseline_mae': float(group.baseline_capped_absolute_error.mean()),
                'earlier_rows': int((group.median_shift_hours < 0).sum()),
                'later_rows': int((group.median_shift_hours > 0).sum()),
                'mean_shift_hours': float((group.capped_mean_hours - group.baseline_capped_mean_hours).mean()),
                'tail_mass_shift': float((group.tail_mass - group.baseline_tail_mass).mean()),
                'uniformly_earlier_cdf_rows': int(group.cdf_uniformly_higher.sum()),
                'mean_logit_shift': float(group.mean_logit_shift.mean())})
        results[arm] = by_event
    assert before == {str(path): sha(path) for path in files}
    assert hashes_unchanged(report['source_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(tables, ignore_index=True).to_parquet(output / 'origin_probability_diagnostics.parquet', index=False)
    write_json(output / 'report.json', {'status': 'completed_probability_diagnosis',
        'source_sha256': before, 'events': results,
        'limitations': ['descriptive_2023_development_only', 'not_causal_attribution',
                        'capped_mean_not_uncensored_expected_duration', 'no_model_change']})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
