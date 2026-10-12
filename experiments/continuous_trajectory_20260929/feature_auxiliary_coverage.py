"""Read-only coverage and missingness diagnosis of frozen saved predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_probability import compare, labels, metrics, sha


HERE = Path(__file__).resolve().parent
RUN = HERE / 'feature_runs/20260929T102202428649Z'
CHANNELS = ['so2', 'no2', 'co', 'o3_8h']


def main():
    report = json.loads((RUN / 'report.json').read_text('utf-8'))
    assert report['status'] == 'completed'
    for filename, digest in report['source_sha256'].items():
        assert sha(Path(filename)) == digest, filename
    for filename, digest in report['artifact_sha256'].items():
        assert sha(RUN / filename) == digest, filename
    source = next(Path(p) for p in report['source_sha256'] if p.endswith('city_hourly.parquet'))
    raw = pd.read_parquet(source)
    extra = raw[CHANNELS].to_numpy(float)
    valid = np.isfinite(extra) & (extra >= 0)
    output = {
        'status': 'passed', 'report_sha256': sha(RUN / 'report.json'),
        'source_sha256': sha(source), 'auditor_sha256': sha(Path(__file__)),
        'metrics_implementation_sha256': sha(HERE / 'verify_probability.py'),
        'scope': 'Saved dev/test populations only; no fitting or selection changes. '
                 'Observed means finite and nonnegative. Completeness means all 24 past hours. '
                 'Missingness groups are descriptive, not causal or independent events.',
        'folds': {}, 'pooled_test': {}}
    pooled = {}
    for key, fold in report['folds'].items():
        task = key.split('_', 1)[1]
        meta = pd.read_parquet(RUN / f'{key}_origins.parquet')
        indices = meta.source_index.to_numpy()
        assert np.array_equal(raw.timestamp.iloc[indices].to_numpy(), meta.timestamp.to_numpy())
        target = raw.aqi.to_numpy()[indices[:, None] + np.arange(1, 25)].astype('float32')
        y = labels(target, task)
        assert np.array_equal(y, meta.label.to_numpy())
        history_valid = valid[indices[:, None] + np.arange(-23, 1)]
        full = history_valid.all(axis=(1, 2))
        probabilities = {}
        for arm in ('baseline', 'trend', 'chemistry'):
            chosen = fold['arms'][arm]['chosen']
            p = np.load(RUN / f'{key}_{arm}_{chosen["candidate"]}.npy', allow_pickle=False)
            assert p.shape == y.shape and np.isfinite(p).all()
            probabilities[arm] = (p, chosen['threshold'])
        result = {}
        for part in ('dev', 'test'):
            keep = meta.partition.eq(part).to_numpy()
            coverage = {}
            for j, channel in enumerate(CHANNELS):
                h = history_valid[keep, :, j]
                coverage[channel] = {
                    'origins': int(keep.sum()), 'complete_24h_origins': int(h.all(axis=1).sum()),
                    'complete_24h_fraction': float(h.all(axis=1).mean()),
                    'observed_hour_fraction': float(h.mean())}
            groups = {}
            for name, group in (('all_four_complete_24h', full), ('any_auxiliary_missing', ~full)):
                m = keep & group
                record = {'n': int(m.sum()), 'positive': int(y[m].sum()), 'arms': {}}
                for arm, (p, threshold) in probabilities.items():
                    score = metrics(y[m], p[m], threshold) if m.any() else None
                    record['arms'][arm] = score
                    if part == 'test' and score is not None:
                        pooled.setdefault((task, name, arm), []).append(score)
                groups[name] = record
            for arm, (p, threshold) in probabilities.items():
                overall = metrics(y[keep], p[keep], threshold)
                reference = fold['arms'][arm]['diagnostics'][part]['overall']
                compare({k: v for k, v in overall.items() if k in reference},
                        reference, (key, part, arm))
                for count in ('n', 'tp', 'fn', 'fp', 'tn'):
                    assert sum(r['arms'][arm][count] for r in groups.values()
                               if r['arms'][arm] is not None) == overall[count]
            result[part] = {'channel_coverage': coverage, 'missingness_groups': groups}
        output['folds'][key] = result
    for (task, group, arm), rows in pooled.items():
        counts = {k: sum(r[k] for r in rows) for k in ('n', 'tp', 'fn', 'fp', 'tn')}
        tp, fn, fp, tn = (counts[k] for k in ('tp', 'fn', 'fp', 'tn'))
        divide = lambda a, b: a / b if b else None
        counts.update(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                      false_positive_rate=divide(fp, fp + tn), f1=divide(2 * tp, 2 * tp + fp + fn))
        output['pooled_test'].setdefault(task, {}).setdefault(group, {})[arm] = counts
    (RUN / 'auxiliary_coverage.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', 'utf-8')
    print(json.dumps(output['pooled_test'], indent=2))
    print('PASS: auxiliary coverage, source alignment, labels, overall metrics and group count conservation')


if __name__ == '__main__':
    main()
