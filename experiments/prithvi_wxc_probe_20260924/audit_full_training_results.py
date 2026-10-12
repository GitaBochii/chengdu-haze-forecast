"""Independently recompute capped interval metrics from completed run artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARTS = ('train', 'dev', 'retrospective_test', 'recent_audit')
ARMS = ('baseline', 'capacity_control', 'spatial_fusion')
SEEDS = (24, 42, 2026)
SCORES = ('exact_row_mae_capped_hours', 'row_error_lower_hours',
          'row_error_upper_hours', 'block_error_lower_hours', 'block_error_upper_hours')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(frame, predictions):
    p = np.asarray(predictions, dtype=np.float64)
    lo = frame.lower_bound_hours.to_numpy(dtype=np.float64)
    hi = frame.upper_bound_hours.to_numpy(dtype=np.float64)
    if (not len(frame) or p.shape != lo.shape or not np.isfinite(p).all()
            or ((p < 1) | (p > 68)).any()):
        raise ValueError('Predictions must be finite, one per row, within [1,68]')
    if (not np.isfinite(lo).all() or (lo < 1).any() or np.isnan(hi).any()
            or (hi < lo).any() or frame.continuity_block.isna().any()):
        raise ValueError('Invalid interval bounds or blocks')
    lo, hi = np.minimum(lo, 68), np.minimum(hi, 68)
    exact = frame.reason.eq('exact').to_numpy()
    if np.any(exact & (lo != hi)):
        raise ValueError('Exact labels must have equal capped bounds')
    nearest = np.clip(p, lo, hi)
    lower = np.abs(p-nearest)
    upper = np.maximum(np.abs(p-lo), np.abs(p-hi))
    block_codes, blocks = pd.factorize(frame.continuity_block)
    block_low = [lower[block_codes == i].mean() for i in range(len(blocks))]
    block_high = [upper[block_codes == i].mean() for i in range(len(blocks))]
    return dict(row_count=len(frame), block_count=len(blocks), original_exact_count=int(exact.sum()),
        exact_row_mae_capped_hours=float(np.abs(p[exact]-lo[exact]).mean()) if exact.any() else None,
        row_error_lower_hours=float(lower.mean()), row_error_upper_hours=float(upper.mean()),
        block_error_lower_hours=float(np.mean(block_low)), block_error_upper_hours=float(np.mean(block_high)))


def audit(run_dir):
    root = Path(run_dir).resolve()
    result = dict(status='fail', run_dir=str(root), errors=[], runs=[], summary={},
                  artifact_sha256={}, std_definition='population standard deviation (ddof=0)',
                  scope='Independent artifact and metric audit; not a new blind test or model rerun.')
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def record(path):
        result['artifact_sha256'][str(path.relative_to(root))] = sha(path)
    try:
        report_path = root/'report.json'
        report = json.loads(report_path.read_text('utf-8'))
        record(report_path)
        require(report['status'] == 'completed_offline_fixed_protocol', 'Final run report not complete')
        selected, excluded = (pd.read_parquet(root/name) for name in ('selected.parquet','excluded.parquet'))
        for name in ('selected.parquet', 'excluded.parquet'):
            record(root/name)
        require(not selected.timestamp.duplicated().any(), 'Duplicate selected origins')
        require(not excluded.timestamp.duplicated().any(), 'Duplicate excluded origins')
        require(not set(selected.timestamp).intersection(excluded.timestamp), 'Selected/excluded overlap')
        require(set(selected.partition) == set(PARTS), 'Incomplete partitions')
        require(selected.groupby('partition').size().to_dict() == report['counts'], 'Reported row counts differ')
        require(len(excluded) == report['excluded_count'], 'Excluded count differs')
        for key in ('continuity_block', 'episode_id', 'pair_id'):
            if key in selected:
                require(not selected.groupby(key).partition.nunique().gt(1).any(), 'Cross-partition '+key)
        expected = {(s,a) for s in SEEDS for a in ARMS}
        records = {(r['seed'], r['arm']):r for r in report['runs']}
        require(len(report['runs']) == 9 and set(records) == expected, 'Need exactly three seeds and three arms')
        result['selected_rows'], result['excluded_rows'] = len(selected), len(excluded)
    except Exception as exc:
        result['errors'].append(str(exc))
        return result
    for seed in SEEDS:
        for arm in ARMS:
            try:
                stored = records[seed,arm]
                checkpoint = root/f'{seed}_{arm}.pt'
                record(checkpoint)
                require(sha(checkpoint) == stored['checkpoint_sha256'], 'Checkpoint hash mismatch')
                pred_path = root/f'{seed}_{arm}_predictions.parquet'
                pred = pd.read_parquet(pred_path)
                record(pred_path)
                require(set(pred.columns) == set(selected.columns) | {'predicted_capped_median_hours'},
                        'Prediction metadata columns differ')
                pd.testing.assert_frame_equal(pred[selected.columns].reset_index(drop=True),
                                              selected.reset_index(drop=True), check_exact=True)
                summaries = {}
                for part in PARTS:
                    rows = pred.loc[pred.partition.eq(part)]
                    computed = metrics(rows, rows.predicted_capped_median_hours.to_numpy())
                    for name, value in computed.items():
                        published = stored['summary'][part][name]
                        require((value is None and published is None) or
                            (value is not None and published is not None and
                             np.isclose(value, published, rtol=1e-7, atol=1e-6)),
                            f'Reported metric differs: {part}/{name}')
                    summaries[part] = computed
                result['runs'].append(dict(seed=seed, arm=arm, summary=summaries))
            except Exception as exc:
                result['errors'].append(f'{seed}/{arm}: {exc}')
    if not result['errors']:
        for part in PARTS:
            result['summary'][part] = {}
            for arm in ARMS:
                values = [r['summary'][part] for r in result['runs'] if r['arm'] == arm]
                result['summary'][part][arm] = {key: {
                    'mean':float(np.mean([v[key] for v in values])) if all(v[key] is not None for v in values) else None,
                    'std':float(np.std([v[key] for v in values])) if all(v[key] is not None for v in values) else None,
                    'seed_count':len(values)} for key in SCORES}
        for name, digest in result['artifact_sha256'].items():
            if sha(root/name) != digest:
                result['errors'].append('Artifact changed during audit: '+name)
        result['status'] = 'fail' if result['errors'] else 'pass'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    output = args.run_dir/'independent_audit.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps(dict(status=result['status'], errors=result['errors'], output=str(output))))
    raise SystemExit(0 if result['status'] == 'pass' else 1)


if __name__ == '__main__':
    main()
