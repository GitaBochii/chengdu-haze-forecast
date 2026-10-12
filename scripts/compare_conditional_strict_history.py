"""Common-origin diagnostic against stored event-isolated website predictions."""

import json

import numpy as np
import pandas as pd

from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.train_recovery_weather_formal import sha
from scripts.verify_recovery_auxiliary import independent_scores


def main():
    history = ROOT / 'artifacts/strict_history_backtest/predictions.parquet'
    research = ROOT / 'artifacts/task01_recovery_head/conditional_crossyear_diagnostic_v1'
    output = ROOT / 'artifacts/task01_recovery_head/conditional_strict_history_common_v1'
    assert not output.exists()
    files = [history, history.parent / 'report.json', research / 'report.json',
             research / 'independent_verification.json',
             ROOT / 'scripts/compare_conditional_strict_history.py']
    files += [research / str(year) / f'{arm}.parquet'
              for year in (2024, 2025) for arm in ('base', 'nonlinear_all')]
    before = {str(path): sha(path) for path in files}
    assert json.loads((research / 'independent_verification.json').read_bytes())['verification'] == 'PASS'
    strict = pd.read_parquet(history)
    assert not strict.timestamp.duplicated().any()
    assert (pd.to_datetime(strict.train_end_timestamp) < strict.timestamp).all()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for year in (2024, 2025):
        base = pd.read_parquet(research / str(year) / 'base.parquet')
        candidate = pd.read_parquet(research / str(year) / 'nonlinear_all.parquet')
        columns = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
        pd.testing.assert_frame_equal(base[columns], candidate[columns])
        scored = base.label_verified_in_scoring_window & base.actual_hours.notna()
        rows = base.loc[scored].copy()
        rows['candidate_prediction_hours'] = candidate.loc[scored, 'predicted_hours']
        joined = rows.merge(strict[['timestamp', 'remaining_hours', 'strict_prediction_hours',
                                    'train_end_timestamp', 'event_key']],
                            on='timestamp', how='left', validate='one_to_one', indicator=True)
        matched = joined['_merge'].eq('both')
        joined.loc[~matched].to_parquet(output / f'{year}_unmatched.parquet', index=False)
        common = joined.loc[matched].copy()
        np.testing.assert_array_equal(common.actual_hours, common.remaining_hours)
        assert np.isfinite(common.strict_prediction_hours).all()
        assert (pd.to_datetime(common.train_end_timestamp) < common.timestamp).all()
        scores = {}
        for name, column in [('research_base', 'predicted_hours'),
                             ('conditional_nonlinear', 'candidate_prediction_hours'),
                             ('website_strict_aft', 'strict_prediction_hours')]:
            frame = common.copy()
            frame['predicted_hours'] = common[column]
            scores[name] = independent_scores(frame)
        common.to_parquet(output / f'{year}_common.parquet', index=False)
        results[str(year)] = {'research_scored_rows': len(rows), 'common_rows': len(common),
            'unmatched_rows': int((~matched).sum()), 'scores': scores}
    assert before == {str(path): sha(path) for path in files}
    result = {'status': 'completed_common_origin_diagnostic', 'years': results,
        'input_sha256': before, 'limitations': [
            'stored_strict_aft_predictions_not_retrained_or_checkpoint_replayed',
            'common_origin_subset_not_all_research_origins',
            'research_fixed_pre2023_weights_vs_aft_expanding_event_refits',
            'same_capped_labels_and_origins_not_identical_information_sets',
            'research_development_gate_failed_no_promotion',
            'not_a_test_of_current_deployment_weights_trained_through_2025']}
    write_json(output / 'report.json', result)
    print(json.dumps(result['years'], indent=2))


if __name__ == '__main__':
    main()
