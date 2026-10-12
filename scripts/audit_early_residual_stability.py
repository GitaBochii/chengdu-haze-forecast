"""Training-period conditional residual audit, not a calibrated predictor."""

import json

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    source = ROOT / 'artifacts/task01_recovery_head/early_temporal_predictions_v1/predictions.parquet'
    out = ROOT / 'artifacts/task01_recovery_head/early_residual_stability_v1'
    if out.exists():
        raise FileExistsError(out)
    digest = sha(source)
    frame = pd.read_parquet(source)
    bounds = [(0, 12), (12, 24), (24, 48), (48, float('inf'))]
    records = []
    for year, rows in frame.groupby(frame.timestamp.dt.year):
        cut = pd.Timestamp(f'{year+1}-01-01', tz='Asia/Shanghai')
        mature = rows.loc[rows.label_available_at.lt(cut)].copy()
        mature['residual_low'] = mature.lower_bound_hours - mature.temporal_prediction
        mature['residual_high'] = mature.upper_bound_hours - mature.temporal_prediction
        for low, high in [(0, float('inf')), *bounds]:
            selected = mature.loc[mature.temporal_prediction.ge(low) & mature.temporal_prediction.lt(high)]
            if selected.empty:
                continue
            endpoints = np.sort(np.r_[selected.residual_low, selected.residual_high])
            n = len(selected)
            # Endpoints delimit all minimizers; no interval label is imputed.
            a, b = float(endpoints[n - 1]), float(endpoints[n])
            exact = selected.loc[selected.exact_hours.notna()]
            records.append({'year': int(year), 'predicted_band': f'{low}-{high}',
                            'rows': n, 'processes': int(selected.continuity_block.nunique()),
                            'unmatured_year_rows_excluded': len(rows) - len(mature),
                            'optimal_shift_low': a, 'optimal_shift_high': b,
                            'exact_rows': len(exact),
                            'exact_signed_error_mean': float((exact.temporal_prediction-exact.exact_hours).mean()) if len(exact) else None})
    table = pd.DataFrame(records)
    assert sha(source) == digest
    out.mkdir()
    table.to_csv(out / 'bands.csv', index=False)
    report = {'input_sha256': digest, 'records': records,
              'limitations': ['Post-hoc diagnosis of training-period forecasts only',
                             'Rows, not process-balanced loss; process counts supplied',
                             'Shifts are unconstrained diagnostics, not nonnegative calibrated forecasts',
                             'No choice of calibration thresholds, parameters, or new model'],
              'deployment_authorized': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
