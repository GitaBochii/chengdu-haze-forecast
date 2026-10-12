"""Independent source, path-distribution and development-decision audit.

Never imports the experiment implementation or reads a historical test result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HOUR = 3_600_000_000_000
SOURCE = Path(__file__).resolve().parents[2] / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
SOURCE_SHA = 'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery(curves):
    a = np.asarray(curves)
    if a.ndim != 2 or a.shape[1] != 24 or not np.isfinite(a).all():
        raise ValueError('Complete finite 24-hour paths required')
    result = np.full(len(a), 20, dtype=np.int64)
    for start in range(19):
        good = np.all(a[:, start:start+6] <= 100, axis=1)
        result[(result == 20) & good] = start + 1
    return result


def grouping(dates, values):
    """Partition and observation discontinuities terminate a process."""
    result = np.full(len(dates), '', dtype='U80')
    segment = -1
    serial = 0
    active = None
    streak = 0
    epoch = pd.Timestamp('2014-01-01', tz='Asia/Shanghai').value
    valid = np.isfinite(values[:, 2])
    for i in range(len(dates)):
        partition = 'train' if dates[i].year <= 2019 else 'dev'
        broken = i == 0 or not valid[i-1] or dates[i].value-dates[i-1].value != HOUR
        broken |= i > 0 and (dates[i-1].year <= 2019) != (dates[i].year <= 2019)
        if broken:
            segment += 1
            active, streak = None, 0
        if not valid[i]:
            active, streak = None, 0
            continue
        if active is None and values[i, 2] > 100:
            serial += 1
            active = f'{partition}:s{segment}:p{serial}'
        if active is not None:
            result[i] = active
            streak = streak + 1 if values[i, 2] <= 100 else 0
            if streak == 6:
                active, streak = None, 0
        else:
            block = (dates[i].value-epoch) // (72*HOUR)
            result[i] = f'{partition}:s{segment}:c{block}'
    return result


def rebuild(raw):
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    if not dates.is_unique:
        raise ValueError('Duplicate timestamps')
    order = np.argsort(dates.asi8, kind='stable')
    raw = raw.iloc[order].reset_index(drop=True)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    values = raw[['pm2_5', 'pm10', 'aqi']].to_numpy(float)
    groups = grouping(dates, values)
    years, ns = dates.year.to_numpy(), dates.asi8
    candidate = np.arange(len(raw), dtype=np.int64)
    selected, reasons = [], []
    for i in candidate:
        if i < 24 or i+24 >= len(raw):
            reasons.append('edge'); continue
        section = ns[i-24:i+25]
        if np.any(np.diff(section) != HOUR):
            reasons.append('discontinuous'); continue
        yrs = years[i-24:i+25]
        if (years[i] <= 2019 and np.any(yrs > 2019)) or (years[i] == 2020 and np.any(yrs != 2020)):
            reasons.append('boundary'); continue
        # t concentration is not an input, but true t AQI must exist for grouping.
        if not (np.isfinite(values[i-24:i]).all() and np.isfinite(values[i+1:i+25]).all() and np.isfinite(values[i, 2])):
            reasons.append('nonfinite'); continue
        selected.append(i); reasons.append('eligible')
    idx = np.asarray(selected, dtype=np.int64)
    history = values[idx[:, None] + np.arange(-24, 0)]
    paths = values[idx[:, None] + np.arange(1, 25)]
    clock = np.column_stack([np.sin(2*np.pi*dates[idx].hour/24), np.cos(2*np.pi*dates[idx].hour/24),
                             np.sin(2*np.pi*dates[idx].month/12), np.cos(2*np.pi*dates[idx].month/12)])
    return raw, dates, dict(indices=idx, timestamps=dates.asi8[idx], history=history, paths=paths,
        features=np.hstack([history.reshape(len(idx), 72), clock]), groups=groups[idx],
        origin_aqi=values[idx, 2], labels=recovery(paths[:, :, 2])), np.asarray(reasons)


def greedy(distance, timestamps, groups, k=32):
    order = np.lexsort((timestamps, distance))
    chosen, used, skipped = [], set(), []
    for row in order:
        if groups[row] in used:
            skipped.append((int(row), 'same_group')); continue
        if any(abs(int(timestamps[row])-int(timestamps[p])) < 49*HOUR for p in chosen):
            skipped.append((int(row), 'overlap')); continue
        chosen.append(int(row)); used.add(groups[row])
        if len(chosen) == k:
            break
    return np.asarray(chosen, dtype=int) if len(chosen) == k else np.full(k, -1), skipped


def decision(cdf, theta):
    hit = np.asarray(cdf) >= theta
    return np.where(hit.any(axis=1), hit.argmax(axis=1)+1, 20)


def metrics(actual, prediction):
    a, p = np.asarray(actual), np.asarray(prediction)
    actual_yes, pred_yes = a < 20, p < 20
    tp, fn, fp, tn = [int(m.sum()) for m in (actual_yes & pred_yes, actual_yes & ~pred_yes,
                                            ~actual_yes & pred_yes, ~actual_yes & ~pred_yes)]
    ratio = lambda x, n: float(x/n) if n else None
    early = int(((p < a) & pred_yes).sum())
    return dict(origins=len(a), actual_right_censored=int((~actual_yes).sum()),
        predicted_no_recovery=int((~pred_yes).sum()), capped_min_T_20_mae_hours=float(np.abs(p-a).mean()) if len(a) else None,
        confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn), precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
        fpr=ratio(fp, fp+tn), premature_start_count=early, premature_start_fraction_all=ratio(early, len(a)),
        premature_start_fraction_predicted_recovery=ratio(early, pred_yes.sum()))


def select(rows, early_limit, key='theta', detections=True):
    allowed = [r for r in rows if r['fpr'] is not None and r['fpr'] <= .05
               and r['premature_start_fraction_all'] <= early_limit
               and (not detections or r['confusion']['tp']+r['confusion']['fp'] > 0)]
    return min(allowed, key=lambda r: (r['capped_min_T_20_mae_hours'], -(r['recall'] or 0), r['fpr'], -r[key])) if allowed else None


def close(actual, expected, label='value'):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (label, set(actual), set(expected))
        for key in expected:
            close(actual[key], expected[key], label+'/'+key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for i, (a, e) in enumerate(zip(actual, expected)):
            close(a, e, label+f'/{i}')
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def probability_metrics(labels, cdf):
    target = labels[:, None] <= np.arange(1, 20)
    brier = np.mean((cdf-target)**2, axis=0)
    return dict(brier_6=float(brier[5]), brier_12=float(brier[11]), brier_19=float(brier[18]),
                mean_brier_1_19=float(brier.mean()))


def main(run):
    run = Path(run)
    old = Path(__file__).resolve().parents[1]/'continuous_trajectory_20260929'
    report = json.loads((run/'report.json').read_text('utf-8'))
    frozen_paths = [Path(__file__), Path(__file__).with_name('test_verify_independent.py'), SOURCE, run/'report.json',
                    old/'verify_delayed_chemistry.py', old/'verify_issue_delay.py']
    frozen_paths += [run/name for name in report['artifact_sha256']]
    frozen = {str(p.resolve()): sha(p) for p in frozen_paths}
    manifest = dict(frozen_sha256=frozen, versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
                    independent_feature_dependencies=['verify_delayed_chemistry.py', 'verify_issue_delay.py'])
    (run/'audit_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    audit = dict(status='failed', verifier_sha256=sha(__file__), independent_frozen_sha256=frozen, checks=[], limits=['Development selected on 2020; not blind test', 'No independent retraining'])
    (run/'verification.json').write_text(json.dumps(dict(audit, status='running'), indent=2)+'\n', encoding='utf-8')
    try:
        verify(run, audit)
        assert all(sha(p) == h for p, h in frozen.items()), 'Independent audit code changed during verification'
        audit['status'] = 'passed'
    except Exception as exc:
        audit['error'] = repr(exc)
        raise
    finally:
        (run/'verification.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(audit, indent=2), flush=True)


def verify(run, audit):
    assert sha(SOURCE) == SOURCE_SHA
    # The very next operation after reading is the frozen date restriction.
    raw = pd.read_parquet(SOURCE)
    raw = raw.loc[(raw.timestamp.dt.year >= 2014) & (raw.timestamp.dt.year <= 2020)].copy()
    raw, dates, data, reasons = rebuild(raw)
    eligible = pd.read_parquet(run/'eligible.parquet')
    np.testing.assert_array_equal(eligible.source_index, data['indices'])
    np.testing.assert_array_equal(pd.DatetimeIndex(eligible.timestamp).asi8, data['timestamps'])
    np.testing.assert_array_equal(eligible.true_start, data['labels'])
    excluded = pd.read_parquet(run/'exclusions.parquet')
    exclusion_idx = np.flatnonzero(reasons != 'eligible')
    np.testing.assert_array_equal(excluded.source_index, exclusion_idx)
    name_map = {'boundary': 'partition_boundary', 'discontinuous': 'hour_gap'}
    np.testing.assert_array_equal(excluded.reason, [name_map.get(r, r) for r in reasons[exclusion_idx]])
    np.testing.assert_array_equal(pd.DatetimeIndex(excluded.timestamp).asi8, dates.asi8[exclusion_idx])
    train_mask = dates[data['indices']].year <= 2019
    splits = {}
    for name, mask in [('train', train_mask), ('dev', ~train_mask)]:
        expected = {k: v[mask] for k, v in data.items()}
        saved = dict(np.load(run/f'{name}.npz', allow_pickle=False))
        for key in expected:
            if key == 'groups':
                # IDs may have different names, but the equivalence relation must match exactly.
                a = pd.factorize(expected[key])[0]; b = pd.factorize(saved[key])[0]
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(expected[key], saved[key], rtol=0, atol=1e-12, err_msg=f'{name}/{key}')
        splits[name] = expected
    audit['checks'].append('independent train/dev qualification, labels, raw paths, 76 features and process equivalence')
    tr, dev = splits['train'], splits['dev']
    mean, std = tr['features'].mean(axis=0), tr['features'].std(axis=0)
    std[std == 0] = 1
    ztr, zdev = (tr['features']-mean)/std, (dev['features']-mean)/std
    stand = dict(np.load(run/'standardization.npz'))
    for key, value in dict(mean=mean, std=std, z_train=ztr, z_dev=zdev).items():
        np.testing.assert_allclose(value, stand[key], rtol=1e-12, atol=1e-12)
    pred = dict(np.load(run/'prediction.npz'))
    n, neighbors = len(dev['indices']), pred['neighbors']
    assert neighbors.shape == (n, 32) and np.all((neighbors >= 0) & (neighbors < len(tr['indices'])))
    for row in neighbors:
        assert len(set(tr['groups'][row])) == 32
        assert np.all(np.diff(np.sort(tr['timestamps'][row])) >= 49*HOUR)
    labels = tr['labels'][neighbors]
    pmf = np.stack([(labels == s).mean(axis=1) for s in range(1, 21)], axis=1)
    cdf = np.cumsum(pmf[:, :19], axis=1)
    np.testing.assert_array_equal(pred['pmf'], pmf)
    np.testing.assert_allclose(pred['cdf'], cdf, rtol=0, atol=1e-15)
    assert np.all(pmf >= 0) and np.allclose(pmf.sum(axis=1), 1) and np.all(np.diff(cdf, axis=1) >= 0)
    np.testing.assert_array_equal(pmf[:, 19], (labels == 20).mean(axis=1))
    np.testing.assert_allclose(pred['path_mean'], tr['paths'][neighbors].mean(axis=1), rtol=1e-12, atol=1e-12)
    onset = np.any(tr['paths'][neighbors, :, 2] > 100, axis=2).mean(axis=1)
    np.testing.assert_array_equal(pred['onset_probability'], onset)
    sample = np.random.default_rng(42).choice(n, size=min(100, n), replace=False)
    skips_saved = pd.read_parquet(run/'neighbor_skips.parquet')
    for query in sample:
        distance = np.mean((ztr-zdev[query])**2, axis=1)
        expected, skipped = greedy(distance, tr['timestamps'], tr['groups'])
        np.testing.assert_array_equal(expected, neighbors[query])
        recorded = skips_saved.loc[skips_saved.dev_row == query]
        assert list(zip(recorded.train_row, recorded.reason)) == [(r, 'overlap_49h' if reason == 'overlap' else reason) for r, reason in skipped]
    audit.update(queries_sampled=len(sample), source_rows_restricted=len(raw), training_origins=len(tr['indices']), development_origins=n)
    audit['checks'].append('all neighbors training-only, unique processes, 49-hour separation, unaltered path-derived distributions; seed42 independent distances and greedy neighbors')
    verify_reports(run, audit, tr, dev, pred, cdf)


def verify_reports(run, audit, tr, dev, pred, cdf):
    report = json.loads((run/'report.json').read_text('utf-8'))
    assert report['status'] == 'completed_pending_verification'
    assert report['train_years'] == list(range(2014, 2020)) and report['dev_year'] == 2020 and report['k'] == 32
    assert report['selected_on_development'] and not report['blind_test'] and not report['deployed']
    for path, expected in report['frozen_sha256'].items():
        assert sha(path) == expected, path
    for path, expected in report['artifact_sha256'].items():
        assert sha(run/path) == expected, path
    for path, expected in report['models'].items():
        assert sha(run/path) == expected, path
    np.testing.assert_array_equal(np.load(run/'neighbor_paths.npy'), tr['paths'][pred['neighbors']])
    np.testing.assert_array_equal(pred['selected_group_count'], np.full(len(dev['indices']), 32))
    med = np.argmax(np.cumsum(pred['pmf'], axis=1) >= .5, axis=1)+1
    np.testing.assert_array_equal(pred['median_start'], med)
    ucdf = (tr['labels'][:, None] <= np.arange(1, 20)).mean(axis=0)
    uonset = float((tr['paths'][:, :, 2] > 100).any(axis=1).mean())
    np.testing.assert_array_equal(pred['unconditional_cdf'], ucdf)
    assert pred['unconditional_onset'] == uonset
    t, g, polluted = dev['labels'], dev['groups'], dev['origin_aqi'] > 100
    raw_start = recovery(pred['raw_curve'])
    raw = metric_schema(t[polluted], raw_start[polluted], g[polluted])
    buffer_rows = [dict(delta=d, **metric_schema(t[polluted], recovery(pred['raw_curve'][polluted]+d), g[polluted])) for d in range(102)]
    close(buffer_rows, json.loads((run/'buffer_grid.json').read_text()), 'buffer_grid')
    allowed = [r for r in buffer_rows if r['fpr'] is not None and r['fpr'] <= .05 and r['early_fraction'] <= raw['early_fraction']]
    buffer = min(allowed, key=lambda r: (r['mae'], -(r['recall'] or 0), r['fpr'], -r['delta']))
    assert report['selected_buffer'] == buffer['delta']
    np.testing.assert_array_equal(pred['conservative_curve'], pred['raw_curve'].astype(float)+buffer['delta'])
    conservative_start = recovery(pred['conservative_curve'])
    reference = metric_schema(t[polluted], conservative_start[polluted], g[polluted])
    grid = [dict(theta=h, **metric_schema(t[polluted], decision(cdf[polluted], h), g[polluted]))
            for h in [round(i*.05, 2) for i in range(1, 21)]+[1.000001]]
    close(grid, json.loads((run/'grid.json').read_text()), 'theta_grid')
    allowed = [r for r in grid if r['fpr'] is not None and r['fpr'] <= .05 and r['early_fraction'] <= raw['early_fraction'] and r['confusion']['tp'] > 0]
    selected = min(allowed, key=lambda r: (r['mae'], -(r['recall'] or 0), r['fpr'], -r['theta'])) if allowed else None
    theta = selected['theta'] if selected else 1.000001
    assert report['selected_theta'] == (theta if selected else None)
    np.testing.assert_array_equal(pred['pred_start'], decision(cdf, theta))
    emp = metric_schema(t[polluted], pred['pred_start'][polluted], g[polluted])
    prob = probability_schema(t[polluted], cdf[polluted], g[polluted])
    unconditional = probability_schema(t[polluted], np.broadcast_to(ucdf, cdf[polluted].shape), g[polluted])
    clean = ~polluted
    truth_onset = np.any(dev['paths'][:, :, 2] > 100, axis=1)
    onset = pred['onset_probability']
    clean_negative = clean & ~truth_onset
    ce = (onset[clean]-truth_onset[clean])**2
    clean_metrics = dict(n=int(clean.sum()), negative_n=int(clean_negative.sum()), fpr=float(np.mean(onset[clean_negative] >= .5)) if clean_negative.any() else None,
        brier=float(ce.mean()), unconditional_brier=float(((uonset-truth_onset[clean])**2).mean()),
        group_equal_brier=group_average(ce, g[clean]), recovery=metric_schema(t[clean], pred['pred_start'][clean], g[clean]))
    rates = []
    for key in np.unique(g[clean]):
        m = clean & (g == key); neg = m & ~truth_onset; pos = m & truth_onset; hat = onset >= .5
        rates.append(dict(fpr=float(hat[neg].mean()) if neg.any() else None, recall=float(hat[pos].mean()) if pos.any() else None,
                          precision=float(truth_onset[m & hat].mean()) if (m & hat).any() else None))
    clean_metrics['group_equal'] = equal_rates(rates, ('fpr', 'recall', 'precision'))
    concentration = {}
    for condition, mask in [('all', np.ones(len(t), bool)), ('polluted', polluted), ('clean', clean)]:
        error = np.abs(pred['path_mean'][mask]-dev['paths'][mask])
        concentration[condition] = {name: dict(n=int(mask.sum()), hour_cells=int(mask.sum()*24), mae=float(error[:, :, j].mean()),
            lead_mae=error[:, :, j].mean(axis=0).tolist(), group_equal_mae=group_average(error[:, :, j].mean(axis=1), g[mask]),
            units='AQI' if name == 'aqi' else 'ug/m3') for j, name in enumerate(('pm2_5', 'pm10', 'aqi'))}
    expected_metrics = dict(raw=raw, raw_matched=raw, conservative_all_polluted=reference, conservative_matched=reference, empirical_matched=emp, probability=prob,
                            unconditional_probability=unconditional, clean=clean_metrics, concentration=concentration)
    audit['independently_recomputed_metrics'] = detached_metrics(expected_metrics)
    # Additional process-equal integrated Brier is retained in this audit; the
    # already-running frozen predictor did not include this reporting field.
    for section in ('probability', 'unconditional_probability'):
        if 'group_equal_brier_mean_1_19' not in report['metrics'][section]:
            expected_metrics[section] = {k: v for k, v in expected_metrics[section].items() if k != 'group_equal_brier_mean_1_19'}
    close(expected_metrics, report['metrics'], 'all_metrics')
    gates = dict(coverage_100=True, actual_detection=selected is not None, mae_10pct=emp['mae'] <= reference['mae']*.9,
        recall_5pp=emp['recall'] is not None and reference['recall'] is not None and emp['recall'] >= reference['recall']+.05,
        fpr_5pct=emp['fpr'] is not None and emp['fpr'] <= .05, year_mae_not_worse=emp['mae'] <= reference['mae'],
        process_mae_not_worse=emp['group_equal_mae'] <= reference['group_equal_mae'], brier_6=prob['brier_6'] <= unconditional['brier_6'],
        brier_12=prob['brier_12'] <= unconditional['brier_12'], brier_19=prob['brier_19'] <= unconditional['brier_19'],
        clean_fpr_5pct=clean_metrics['fpr'] is not None and clean_metrics['fpr'] <= .05, clean_brier=clean_metrics['brier'] <= clean_metrics['unconditional_brier'])
    close(gates, report['gates'], 'gates')
    close([k for k, v in gates.items() if not v], report['failure_reasons'], 'failure_reasons')
    assert report['preliminary_development_pass'] == all(gates.values())
    rows = pd.read_parquet(run/'row_metrics.parquet')
    for key, arr in dict(dev_row=np.arange(len(t)), polluted=polluted, covered=np.ones(len(t), bool), true_start=t,
                         pred_start=pred['pred_start'], raw_start=raw_start, conservative_start=conservative_start,
                         onset_probability=onset, true_onset=truth_onset, absolute_error=np.abs(pred['pred_start']-t),
                         reference_absolute_error=np.abs(conservative_start-t)).items():
        np.testing.assert_array_equal(rows[key], arr)
    np.testing.assert_array_equal(pd.DatetimeIndex(rows.timestamp).asi8, dev['timestamps'])
    # Reuse only independently reconstructed group equivalence, not saved row IDs.
    rows['group'] = g
    expected_groups = rows.groupby(['polluted', 'group'], as_index=False).agg(n=('dev_row', 'size'), mae=('absolute_error', 'mean'), reference_mae=('reference_absolute_error', 'mean'))
    saved_groups = pd.read_parquet(run/'group_metrics.parquet')
    # Process names are implementation-independent; compare in first appearance order.
    key_map = dict(zip(pd.read_parquet(run/'row_metrics.parquet')['group'], g))
    saved_groups['group'] = saved_groups['group'].map(key_map)
    pd.testing.assert_frame_equal(expected_groups.sort_values(['polluted', 'group']).reset_index(drop=True), saved_groups.sort_values(['polluted', 'group']).reset_index(drop=True))
    reliability = []
    for s in (6, 12, 19):
        p, y = cdf[polluted, s-1], t[polluted] <= s
        for b in range(10):
            m = (p >= b/10) & (p <= 1 if b == 9 else p < (b+1)/10)
            reliability.append(dict(horizon=s, bin=b, n=int(m.sum()), mean_probability=float(p[m].mean()) if m.any() else None,
                                    observed_fraction=float(y[m].mean()) if m.any() else None))
    pd.testing.assert_frame_equal(pd.DataFrame(reliability), pd.read_parquet(run/'reliability.parquet'))
    counts = dict(train=len(tr['indices']), dev=len(t), covered=len(t), no_prediction=0, polluted=int(polluted.sum()), clean=int(clean.sum()), exclusions=len(pd.read_parquet(run/'exclusions.parquet')))
    close(counts, report['counts'], 'counts')
    audit['checks'].append('all row/process/clean/concentration/Brier/reliability metrics; all 21 theta and 102 buffer rows; development choice and gates')
    verify_chemistry(run, audit, tr, dev, pred)


def group_average(values, groups):
    return float(np.mean([np.mean(values[groups == key]) for key in np.unique(groups)]))


def metric_schema(t, p, g, include_groups=True):
    m = metrics(t, p)
    result = dict(n=len(t), actual_right_censored=m['actual_right_censored'], predicted_recovery=int((p < 20).sum()),
        mae=m['capped_min_T_20_mae_hours'], group_equal_mae=group_average(np.abs(t-p), g) if len(t) else None,
        groups=len(set(g)), confusion=m['confusion'], recall=m['recall'], precision=m['precision'], fpr=m['fpr'],
        early_fraction=m['premature_start_fraction_all'], early_count=m['premature_start_count'])
    if include_groups:
        rates = [metric_schema(t[g == key], p[g == key], g[g == key], False) for key in np.unique(g)]
        result['group_equal'] = equal_rates(rates, ('mae', 'recall', 'precision', 'fpr', 'early_fraction'))
    return result


def equal_rates(rates, keys):
    result = {}
    for key in keys:
        valid = [row[key] for row in rates if row[key] is not None]
        result[key] = dict(value=float(np.mean(valid)) if valid else None, valid_groups=len(valid), total_groups=len(rates))
    return result


def detached_metrics(values):
    return json.loads(json.dumps(values))


def probability_schema(t, cdf, g):
    e = (cdf-(t[:, None] <= np.arange(1, 20)))**2
    return dict(n=len(t), brier_6=float(e[:, 5].mean()), brier_12=float(e[:, 11].mean()), brier_19=float(e[:, 18].mean()),
        brier_mean_1_19=float(e.mean()), group_equal_brier_6=group_average(e[:, 5], g),
        group_equal_brier_12=group_average(e[:, 11], g), group_equal_brier_19=group_average(e[:, 18], g),
        group_equal_brier_mean_1_19=group_average(e.mean(axis=1), g))


def verify_chemistry(run, audit, tr, dev, pred):
    import lightgbm as lgb
    old = Path(__file__).resolve().parents[1]/'continuous_trajectory_20260929'
    sys.path.insert(0, str(old))
    # Existing independent verifier, not the model's feature builder.
    from verify_delayed_chemistry import reconstruct
    raw = pd.read_parquet(SOURCE)
    raw = raw.loc[(raw.timestamp.dt.year >= 2014) & (raw.timestamp.dt.year <= 2020)].copy().sort_values('timestamp').reset_index(drop=True)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    ix = np.r_[tr['indices'], dev['indices']]
    x, _ = reconstruct(raw, dates, ix, 2019, 1)
    np.testing.assert_allclose(x, np.load(run/'chemistry_features.npy'), rtol=0, atol=0)
    for lead in range(24):
        model = lgb.Booster(model_str=(run/f'chemistry_aqi_h{lead+1:02}.txt').read_text('utf-8'))
        assert model.num_feature() == 323 and model.current_iteration() == 300
        assert model.params['objective'] == 'regression_l1' and model.params['num_leaves'] == 24
        assert np.isclose(float(model.params['learning_rate']), .04) and model.params['min_data_in_leaf'] == 30
        restored = model.predict(x[len(tr['indices']):], num_threads=4).clip(min=0).astype(np.float32)
        np.testing.assert_array_equal(restored, pred['raw_curve'][:, lead])
    audit['checks'].append('independent 323 chemistry feature reconstruction and all 24 saved model reinferences')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
