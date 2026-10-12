"""Same-origin corrected-label diagnostic against actual history-page predictions."""

import json
from pathlib import Path

import pandas as pd

from scripts.benchmark_observed_business_aft import ROOT, metrics, sha


def main():
    root = ROOT / 'artifacts/task01_recovery_head'
    out = root / 'candidate_vs_history_page_v1'
    if out.exists():
        raise FileExistsError(out)
    page_path = ROOT / 'artifacts/strict_history_backtest/predictions.parquet'
    candidates = {2024: root / 'observed_hazard_crossyear_v1/2024_predictions.parquet',
                  2025: root / 'ghcnh_weather_repair_v1/predictions.parquet'}
    full_path = root / 'observed_business_dataset_v1/features.parquet'
    paths = [page_path, full_path, *candidates.values(), Path(__file__),
        ROOT / 'src/haze_forecast/app.py', ROOT / 'scripts/benchmark_observed_business_aft.py']
    before = {str(p): sha(p) for p in paths}
    page = pd.read_parquet(page_path)
    assert page.timestamp.is_unique
    columns = ['timestamp', 'strict_prediction_hours', 'train_end_timestamp', 'event_key', 'remaining_hours']
    page = page[columns].rename(columns={'remaining_hours': 'page_old_target', 'event_key': 'page_event_key'})
    full = pd.read_parquet(full_path)
    block_starts = full.groupby('continuity_block').timestamp.min()
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'candidate': '87-feature observed-prefix hazard, annually trained',
        'website_mode': 'strict history backtest; not future-prediction deployment model',
        'evaluation': 'shared timestamps with same newly observed exact recovery labels',
        'limitations': ['candidate annual refit vs page event-expanding refit',
            'candidate weather repaired 2025 vs original page input',
            'page train_end_timestamp alone does not prove label availability',
            '2024/2025 reused diagnostic; no selection or deployment']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    for year, candidate_path in candidates.items():
        frame = pd.read_parquet(candidate_path)
        frame['candidate_prediction'] = frame['hazard' if year == 2024 else 'hazard_after']
        merged = frame.merge(page, on='timestamp', how='left', validate='one_to_one', indicator=True)
        matched = merged.loc[merged['_merge'].eq('both')].copy()
        assert matched.train_end_timestamp.lt(matched.timestamp).all()
        matched['train_ends_before_block'] = matched.train_end_timestamp.lt(matched.continuity_block.map(block_starts))
        exact = matched.exact_hours.notna()
        statistics = {
            'all_candidate_origins': len(frame), 'matched_origins': len(matched),
            'unmatched_origins': len(merged) - len(matched), 'matched_exact': int(exact.sum()),
            'corrected_label_differs_from_old': int((matched.loc[exact, 'exact_hours'] != matched.loc[exact, 'page_old_target']).sum()),
            'page_train_end_not_before_new_block': int((~matched.train_ends_before_block).sum()),
            'candidate': metrics(matched, matched.candidate_prediction.to_numpy()),
            'page': metrics(matched, matched.strict_prediction_hours.to_numpy()),
        }
        safe = matched.loc[matched.train_ends_before_block]
        statistics['conservative_subset'] = {'origins': len(safe),
            'candidate': metrics(safe, safe.candidate_prediction.to_numpy()),
            'page': metrics(safe, safe.strict_prediction_hours.to_numpy())}
        matched.to_parquet(out / f'{year}_common.parquet', index=False)
        merged.loc[merged['_merge'].eq('left_only')].to_parquet(out / f'{year}_unmatched.parquet', index=False)
        results[str(year)] = statistics
    assert before == {str(p): sha(p) for p in paths}
    report = {'status': 'completed_history_page_comparison', 'years': results,
        'inputs_unchanged': True, 'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
