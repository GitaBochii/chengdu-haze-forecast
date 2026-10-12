"""Fixed AQI transition evaluation of saved forecasts, without model selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transition_times(aqi):
    aqi = np.asarray(aqi)
    if aqi.ndim != 2 or aqi.shape[1] != 24 or not np.isfinite(aqi).all() or (aqi < 0).any():
        raise ValueError('Expected finite nonnegative N by 24 AQI')
    polluted = aqi > 100
    onset = np.where(polluted.any(axis=1), polluted.argmax(axis=1) + 1, 25)
    stable = np.lib.stride_tricks.sliding_window_view(~polluted, 6, axis=1).all(axis=2)
    recovery = np.where(stable.any(axis=1), stable.argmax(axis=1) + 1, 20)
    return onset, recovery


def confusion(truth, pred):
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    if truth.shape != pred.shape:
        raise ValueError('Shapes differ')
    tp, fn = int((truth & pred).sum()), int((truth & ~pred).sum())
    fp, tn = int((~truth & pred).sum()), int((~truth & ~pred).sum())
    return dict(n=int(truth.size), tp=tp, fn=fn, fp=fp, tn=tn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def mean_or_none(x):
    return float(np.mean(x)) if len(x) else None


def timing(truth, pred, cap, days):
    result = confusion(truth < cap, pred < cap)
    errors = np.abs(truth - pred)
    joint = (truth < cap) & (pred < cap)
    result.update(capped_time_mae_h=mean_or_none(errors),
                  capped_time_bias_h=mean_or_none(pred-truth),
                  day_equal_capped_time_mae_h=mean_or_none([errors[days == d].mean() for d in np.unique(days)]),
                  joint_detection_n=int(joint.sum()),
                  joint_detection_timing_mae_h=mean_or_none(errors[joint]))
    return result


def evaluate(y, p, origin, days):
    yo, yr = transition_times(y)
    po, pr = transition_times(p)
    origin, days = np.asarray(origin), np.asarray(days)
    if y.shape != p.shape or len(y) != len(origin) or len(y) != len(days):
        raise ValueError('Evaluation shapes differ')
    dirty = origin > 100
    count_error = (p > 100).sum(axis=1) - (y > 100).sum(axis=1)
    return dict(n=len(y), hourly_state=confusion(y > 100, p > 100),
                polluted_hours=dict(mae_h=mean_or_none(np.abs(count_error)), bias_h=mean_or_none(count_error)),
                onset=timing(yo[~dirty], po[~dirty], 25, days[~dirty]),
                recovery=timing(yr[dirty], pr[dirty], 20, days[dirty]))


def main(run):
    original = json.loads((run/'report.json').read_text('utf-8'))
    previous_audit = json.loads((run/'independent_audit.json').read_text('utf-8'))
    assert previous_audit['status'] == 'passed_independent_source_and_metric_audit'
    assert previous_audit['report_sha256'] == sha(run/'report.json')
    for name, digest in original['artifact_sha256'].items():
        assert sha(run/name) == digest, name
    meta = pd.read_parquet(run/'evaluation_origins.parquet')
    target = np.load(run/'targets.npy')[:, :, 2]
    assert len(meta) == len(target)
    out = dict(status='completed_fixed_transition_audit',
               scope='24-hour AQI proxy; recovery starts 1-19, cap20; onset cap25; not full haze dissipation',
               report_sha256=sha(run/'report.json'),
               evaluator_sha256=sha(__file__),
               protocol_sha256=sha(Path(__file__).with_name('EVENT_PROTOCOL.md')), results={})
    days = meta.timestamp.dt.strftime('%Y-%m-%d').to_numpy()
    for arm in ('persistence', 'daily_repeat', 'pollution_only', 'all_conditions'):
        pred = np.load(run/(arm+'.npy'))[:, :, 2]
        out['results'][arm] = {}
        for part in ('dev', 'test', 'recent_audit'):
            m = meta.partition.eq(part).to_numpy()
            out['results'][arm][part] = evaluate(target[m], pred[m], meta.origin_aqi.to_numpy()[m], days[m])
    (run/'event_metrics.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
