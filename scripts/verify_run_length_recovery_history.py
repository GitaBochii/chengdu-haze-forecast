"""Oracle-mechanism check using observed future AQI, never a forecast score."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.run_length_recovery import recovery_start_distribution
from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    source = base / 'gefs_temporal_calibration_data_v1/predictions.parquet'
    grid_path = base / 'observed_business_dataset_v1/features.parquet'
    out = base / 'run_length_mechanism_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [source, grid_path, Path(__file__), ROOT / 'src/haze_forecast/run_length_recovery.py',
              ROOT / 'tests/test_run_length_recovery.py', ROOT / 'src/haze_forecast/app.py']
    paths += [p for p in (ROOT / 'artifacts/cnemc_2025_deployment_candidate').iterdir() if p.is_file()]
    before = {str(p): sha(p) for p in paths}
    rows = pd.read_parquet(source)
    grid = pd.read_parquet(grid_path).set_index('timestamp')
    checked = []
    for row in rows.itertuples():
        end = row.timestamp + pd.Timedelta(hours=row.exact_hours + 5)
        future = grid.loc[row.timestamp + pd.Timedelta(hours=1):end, 'aqi']
        assert len(future) == int(row.exact_hours) + 5
        if future.isna().any():
            checked.append({'timestamp': row.timestamp, 'status': 'unknown_observations_not_imputed',
                            'mechanism_start_hour': None})
            continue
        clean = future.le(100).to_numpy(dtype=float)
        q = np.broadcast_to(clean[None, :, None], (1, len(clean), 6))
        distribution = recovery_start_distribution(q)[0]
        assert distribution[-1] == 0 and distribution.sum() == 1
        start = int(distribution[:-1].argmax()) + 1
        assert start == row.exact_hours
        checked.append({'timestamp': row.timestamp, 'status': 'oracle_mechanism_pass',
                        'mechanism_start_hour': start})
    result = pd.DataFrame(checked)
    assert before == {str(p): sha(Path(p)) for p in before}
    out.mkdir()
    result.to_parquet(out / 'checks.parquet', index=False)
    report = {'input_sha256': before, 'counts': result.status.value_counts().to_dict(),
              'purpose': 'Known future used only to verify decoder semantics; no learned probabilities',
              'accuracy_claim_authorized': False, 'deployment_authorized': False,
              'output_sha256': {'checks.parquet': sha(out / 'checks.parquet')}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['counts'], indent=2))


if __name__ == '__main__':
    main()
