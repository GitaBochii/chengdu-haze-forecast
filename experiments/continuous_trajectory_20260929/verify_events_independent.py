"""Independent loop-based verification; never imports the primary evaluator."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RUN = BASE / 'runs' / '20260929T091246660694Z'
ARMS = ('persistence', 'daily_repeat', 'pollution_only', 'all_conditions')


def mean(values):
    return sum(values) / len(values) if len(values) else None


def transition(row, kind):
    if len(row) != 24 or not all(math.isfinite(float(x)) for x in row):
        raise ValueError('24 finite AQI values required')
    if kind == 'onset':
        for hour, value in enumerate(row, 1):
            if value > 100:
                return hour
        return 25
    if kind != 'recovery':
        raise ValueError(kind)
    for start in range(19):
        clean = True
        for offset in range(6):
            if row[start + offset] > 100:
                clean = False
                break
        if clean:
            return start + 1
    return 20


def confusion(observed, predicted):
    counts = dict(n=0, tp=0, fn=0, fp=0, tn=0)
    assert len(observed) == len(predicted)
    for actual, forecast in zip(observed, predicted):
        counts['n'] += 1
        counts['tp' if actual and forecast else 'fn' if actual else 'fp' if forecast else 'tn'] += 1
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    counts.update(precision=tp / (tp + fp) if tp + fp else None,
                  recall=tp / (tp + fn) if tp + fn else None,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    return counts


def event_scores(actual, predicted, dates, kind):
    cap = 25 if kind == 'onset' else 20
    a = [transition(row, kind) for row in actual]
    p = [transition(row, kind) for row in predicted]
    result = confusion([x < cap for x in a], [x < cap for x in p])
    differences = [y - x for x, y in zip(a, p)]
    day_errors = defaultdict(list)
    for day, error in zip(dates, differences):
        day_errors[day].append(abs(error))
    joint = [abs(y - x) for x, y in zip(a, p) if x < cap and y < cap]
    result.update(capped_time_mae_h=mean([abs(x) for x in differences]),
                  capped_time_bias_h=mean(differences),
                  day_equal_capped_time_mae_h=mean([mean(x) for x in day_errors.values()]),
                  joint_detection_n=len(joint), joint_detection_timing_mae_h=mean(joint))
    return result


def evaluate(actual, predicted, origins):
    observed_state = [bool(value > 100) for row in actual for value in row]
    predicted_state = [bool(value > 100) for row in predicted for value in row]
    differences = [sum(value > 100 for value in p) - sum(value > 100 for value in a)
                   for a, p in zip(actual.tolist(), predicted.tolist())]
    result = dict(n=len(actual), hourly_state=confusion(observed_state, predicted_state),
                  polluted_hours=dict(mae_h=mean([abs(x) for x in differences]), bias_h=mean(differences)))
    for kind in ('onset', 'recovery'):
        indices = [i for i, value in enumerate(origins.origin_aqi)
                   if (value <= 100 if kind == 'onset' else value > 100)]
        dates = [origins.timestamp.iloc[i].date().isoformat() for i in indices]
        result[kind] = event_scores(actual[indices], predicted[indices], dates, kind)
    return result


def semantic_tests():
    cases = 0
    def check(condition):
        nonlocal cases
        assert condition
        cases += 1
    check(transition([100] * 24, 'onset') == 25)
    check(transition([100] * 24, 'recovery') == 1)
    check(transition([101] + [100] * 23, 'onset') == 1)
    check(transition([100] * 23 + [101], 'onset') == 24)
    check(transition([101] * 18 + [100] * 6, 'recovery') == 19)
    check(transition([101] * 19 + [100] * 5, 'recovery') == 20)
    check(transition([100] * 5 + [101] + [100] * 18, 'recovery') == 7)
    check(transition([101] * 24, 'recovery') == 20)
    check(confusion([], []) == dict(n=0, tp=0, fn=0, fp=0, tn=0, precision=None, recall=None, f1=None))
    check(confusion([True, True, False, False], [True, False, True, False]) ==
          dict(n=4, tp=1, fn=1, fp=1, tn=1, precision=0.5, recall=0.5, f1=0.5))
    check(confusion([True], [False])['f1'] == 0)
    check(confusion([False], [False])['f1'] is None)
    empty = event_scores([], [], [], 'onset')
    check(empty['capped_time_mae_h'] is None and empty['joint_detection_timing_mae_h'] is None)
    a = [[101] * 24, [100] * 24, [100] * 24]
    p = [[100] * 24, [100] * 24, [100] * 24]
    scored = event_scores(a, p, ['a', 'b', 'b'], 'onset')
    check(scored['capped_time_mae_h'] == 8 and scored['day_equal_capped_time_mae_h'] == 12)
    recovered = event_scores([[101] * 24], [[100] * 24], ['a'], 'recovery')
    check(recovered['fp'] == 1 and recovered['capped_time_mae_h'] == 19 and recovered['joint_detection_n'] == 0)
    return cases


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(expected, actual, path, evidence):
    if isinstance(expected, dict):
        assert set(expected) == set(actual), path
        for key in expected:
            compare(expected[key], actual[key], path + '.' + key, evidence)
        return
    if expected is None:
        passed = actual is None
        error = None
    else:
        error = abs(expected - actual)
        passed = error <= 1e-10 if isinstance(expected, float) else expected == actual
    evidence.append(dict(path=path, expected=expected, recomputed=actual,
                         absolute_error=error, passed=bool(passed)))


def main():
    tests = semantic_tests()
    paths = [RUN / name for name in ['event_metrics.json', 'report.json', 'targets.npy',
             'evaluation_origins.parquet'] + [arm + '.npy' for arm in ARMS]]
    paths += [BASE / 'EVENT_PROTOCOL.md', BASE / 'event_audit.py']
    hashes = {str(p.relative_to(BASE)): digest(p) for p in paths}
    expected = json.loads((RUN / 'event_metrics.json').read_text(encoding='utf-8'))
    assert expected['report_sha256'] == digest(RUN / 'report.json')
    assert expected['protocol_sha256'] == digest(BASE / 'EVENT_PROTOCOL.md')
    assert expected['evaluator_sha256'] == digest(BASE / 'event_audit.py')
    report = json.loads((RUN / 'report.json').read_text(encoding='utf-8'))
    assert report['trajectory_target_order'] == ['pm2_5', 'pm10', 'aqi']
    origins = pd.read_parquet(RUN / 'evaluation_origins.parquet')
    targets = np.load(RUN / 'targets.npy', allow_pickle=False)
    assert targets.shape == (len(origins), 24, 3) and np.isfinite(targets).all()
    assert (origins.polluted == (origins.origin_aqi > 100)).all()
    results, coverage = {}, {}
    for partition, year in [('dev', 2024), ('test', 2025), ('recent_audit', 2026)]:
        subset = origins[origins.partition == partition]
        assert (subset.timestamp.dt.year == year).all()
        coverage[partition] = dict(year=year, n=len(subset), days=subset.timestamp.dt.date.nunique(),
                                   first=str(subset.timestamp.min()), last=str(subset.timestamp.max()))
    for arm in ARMS:
        predicted = np.load(RUN / (arm + '.npy'), allow_pickle=False)
        assert predicted.shape == targets.shape and np.isfinite(predicted).all()
        results[arm] = {}
        for partition in coverage:
            mask = (origins.partition == partition).to_numpy()
            results[arm][partition] = evaluate(targets[mask, :, 2], predicted[mask, :, 2], origins.loc[mask])
    comparisons = []
    compare(expected['results'], results, 'results', comparisons)
    unchanged = all(digest(path) == hashes[str(path.relative_to(BASE))] for path in paths)
    passed = all(row['passed'] for row in comparisons) and unchanged
    output = dict(status='passed' if passed else 'failed', independent_implementation=True,
                  primary_evaluator_imported=False, semantic_tests_passed=tests,
                  numeric_absolute_tolerance=1e-10, leaves_compared=len(comparisons),
                  failures=[row for row in comparisons if not row['passed']],
                  max_absolute_error=max(row['absolute_error'] or 0 for row in comparisons),
                  existing_inputs_unchanged=unchanged, input_sha256=hashes,
                  verifier_sha256=digest(Path(__file__)), coverage=coverage,
                  limitations=['AQI transition proxies, not particle-primary haze labels.',
                               'Recovery cap20 includes censoring beyond start hour19; not full dissipation time.',
                               'Onset cap25 encodes no onset within24h; not an observed onset.',
                               'Joint-detection timing is selected; overlapping origins are not independent events.',
                               'Retrospective audit only; no tuning, significance claim, or deployment.'],
                  recomputed_results=results, comparisons=comparisons)
    (RUN / 'event_independent_audit.json').write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: output[key] for key in ['status', 'semantic_tests_passed', 'leaves_compared',
                     'max_absolute_error', 'existing_inputs_unchanged', 'failures']}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
