"""Single development-fit block-balanced location correction, never year-tuned."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    devpath = base / 'interval_history_counts_v1/expanded_predictions.parquet'
    out = base / 'history_location_calibration_v1'
    if out.exists():
        raise FileExistsError(out)
    yearpaths = {y: base / f'interval_history_crossyear_v1/{y}_common.parquet' for y in (2024, 2025)}
    paths = [devpath, *yearpaths.values(), Path(__file__), ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    dev = pd.read_parquet(devpath)
    dev = dev.loc[dev.score_allowed & dev.exact_hours.notna()].copy()
    residual = dev.exact_hours - dev.history_counts_aft
    weight = 1 / dev.groupby('continuity_block').timestamp.transform('size')
    order = np.argsort(residual.to_numpy(), kind='stable')
    sorted_r, sorted_w = residual.to_numpy()[order], weight.to_numpy()[order]
    offset = float(sorted_r[np.searchsorted(np.cumsum(sorted_w), sorted_w.sum() / 2)])
    # Weighted median minimizes the fixed development block-balanced absolute error.
    objective = lambda c: float(np.sum(weight.to_numpy() * np.abs(residual.to_numpy() - c)))
    assert objective(offset) <= min(objective(offset - 1e-5), objective(offset + 1e-5)) + 1e-10
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'offset_hours': offset, 'calibration_rows': len(dev),
                'calibration_blocks': int(dev.continuity_block.nunique()),
                'rule': 'development residual weighted median; inverse block origin-count weights; prediction floor zero',
                'selection': 'one fixed correction; no later-year fitting or search',
                'limitations': ['2023 reused development cohort; conditional exact-label selection',
                                'candidate annual refit versus webpage event refit; weather inputs differ']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    for year, path in yearpaths.items():
        frame = pd.read_parquet(path)
        frame['calibrated'] = np.maximum(0, frame.history_aft + offset)
        scores = {arm: metrics(frame, frame[arm].to_numpy()) for arm in
                  ('calibrated', 'history_aft', 'strict_prediction_hours')}
        candidate, page = scores['calibrated'], scores['strict_prediction_hours']
        results[str(year)] = {'scores': scores, 'passes_page_gate': bool(
            candidate['row_mae'] < page['row_mae'] and candidate['block_mae'] < page['block_mae'] and
            candidate['tail_24h_block_mae'] <= page['tail_24h_block_mae'])}
        frame.to_parquet(out / f'{year}_predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'years': results, 'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'offset_hours': offset, 'years': results}, indent=2))


if __name__ == '__main__':
    main()
