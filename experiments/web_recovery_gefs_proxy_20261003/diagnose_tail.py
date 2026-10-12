"""Read-only diagnostics of a completed run; never selects or fits models."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/20261003T145140578727Z'
FILES = ['predictions.parquet', 'features.npz', 'report.json', 'verification.json']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = {name: digest(RUN / name) for name in FILES}
    verification = json.loads((RUN / 'verification.json').read_text(encoding='utf-8'))
    assert verification['status'] == 'passed'
    assert verification['report_sha256'] == before['report.json']
    frame = pd.read_parquet(RUN / 'predictions.parquet')
    with np.load(RUN / 'features.npz', allow_pickle=False) as features:
        names = features['feature_names'].tolist()
        weather = features['weather_values']
        assert len(frame) == 701 and weather.shape == (701, 99)
        assert np.array_equal(frame['eval_row'].to_numpy(), np.arange(701))
        timestamp_ns = pd.to_datetime(frame.timestamp, utc=True).astype('int64').to_numpy()
        assert np.array_equal(timestamp_ns, features['timestamps'])
        weather_columns = [i for i, name in enumerate(names) if name.startswith('gefs_')]
        assert len(weather_columns) == 9
        weather_finite = np.isfinite(weather[:, weather_columns])
        assert np.all(weather_finite.all(axis=1) | ~weather_finite.any(axis=1))
        frame['weather_present'] = weather_finite.all(axis=1)
        frame['remaining_lead'] = weather[:, names.index('forecast_remaining_lead_hours')]
        age = weather[:, names.index('forecast_run_age_hours')]
        frame['forecast_run_at'] = pd.to_datetime(frame.timestamp, utc=True) - pd.to_timedelta(age, unit='h')
    rows = []
    summaries = {}
    for year in (2024, 2025):
        tail = frame[(frame.year == year) & (frame.exact_hours >= 24)].copy()
        assert len(tail) > 0
        present = tail[tail.weather_present]
        for block, group in tail.groupby('continuity_block', sort=True):
            rows.append({
                'year': year, 'block': int(block), 'rows': len(group),
                'start': str(group.timestamp.min()), 'end': str(group.timestamp.max()),
                'true_T_mean': float(group.exact_hours.mean()),
                'archive_mae': float(group.archive_exact_error.mean()),
                'metadata_mae': float(group.metadata_control_exact_error.mean()),
                'weather_mae': float(group.weather_values_exact_error.mean()),
                'weather_signed_mean': float(group.weather_values_signed_error.mean()),
                'metadata_signed_mean': float(group.metadata_control_signed_error.mean()),
                'delta_archive': float((group.weather_values_exact_error-group.archive_exact_error).mean()),
                'delta_metadata': float((group.weather_values_exact_error-group.metadata_control_exact_error).mean()),
                'weather_rows': int(group.weather_present.sum()),
                'unique_forecast_runs': int(group.forecast_run_at.nunique()),
                'lead_min': float(group.remaining_lead.min()),
                'lead_max': float(group.remaining_lead.max()),
            })
        groups = [row for row in rows if row['year'] == year]
        summaries[str(year)] = {
            'tail_rows': len(tail), 'tail_blocks': len(groups),
            'weather_rows': len(present),
            'known_lead_shorter_than_true_T': int((present.remaining_lead < present.exact_hours).sum()),
            'weather_underestimation_rows': int((tail.weather_values_signed_error < 0).sum()),
            'delta_archive_block_mean': float(np.mean([g['delta_archive'] for g in groups])),
            'delta_metadata_block_mean': float(np.mean([g['delta_metadata'] for g in groups])),
        }
        for arm in ('archive', 'metadata_control', 'weather_values'):
            reconstructed = float(np.mean([g[{'archive': 'archive_mae', 'metadata_control': 'metadata_mae', 'weather_values': 'weather_mae'}[arm]] for g in groups]))
            direct = float(tail.groupby('continuity_block')[arm + '_exact_error'].mean().mean())
            assert abs(reconstructed - direct) < 1e-10
    after = {name: digest(RUN / name) for name in FILES}
    assert before == after
    pd.DataFrame(rows).to_csv(HERE / 'tail_diagnostic.csv', index=False, encoding='utf-8-sig')
    payload = {'scope': 'exploratory_read_only_no_selection_no_new_accuracy',
               'source_sha256': before, 'source_unchanged': True, 'years': summaries}
    (HERE / 'tail_diagnostic.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
