"""Audit within-event recovery-clock consistency without changing labels."""

import json

import numpy as np
import pandas as pd

from scripts.build_strict_history_backtest import ROOT, SOURCES
from scripts.train_recovery_weather_formal import sha


def main():
    output = ROOT / 'artifacts/task01_recovery_head/aft_label_clock_audit_v1'
    assert not output.exists()
    hashes = {str(path): sha(path) for path in SOURCES.values()}
    reports, details = {}, []
    for source, path in SOURCES.items():
        frame = pd.read_parquet(path)
        frame['implied_recovery_ns'] = pd.DatetimeIndex(frame.timestamp).asi8 + np.rint(
            frame.remaining_hours.to_numpy() * 3600 * 10**9).astype(np.int64)
        groups = frame.groupby('event_id').implied_recovery_ns.agg(['nunique', 'min', 'max'])
        inconsistent = groups.index[groups['nunique'] > 1]
        affected = frame.event_id.isin(inconsistent)
        reports[source] = {'rows': len(frame), 'events': len(groups),
            'inconsistent_events': len(inconsistent), 'rows_in_inconsistent_events': int(affected.sum()),
            'max_implied_endpoint_spread_hours': float((groups['max'] - groups['min']).max() / (3600 * 10**9))}
        selected = frame.loc[affected, ['timestamp', 'event_id', 'remaining_hours', 'implied_recovery_ns']].copy()
        selected['source'] = source
        selected['latest_implied_endpoint_ns'] = selected.event_id.map(groups['max'])
        selected['endpoint_spread_from_latest_hours'] = (
            selected.latest_implied_endpoint_ns - selected.implied_recovery_ns) / (3600 * 10**9)
        details.append(selected)
    assert hashes == {str(path): sha(path) for path in SOURCES.values()}
    output.mkdir(parents=True)
    pd.concat(details, ignore_index=True).to_parquet(output / 'inconsistent_event_rows.parquet', index=False)
    result = {'status': 'completed_label_clock_diagnostic', 'sources': reports, 'input_sha256': hashes,
        'script_sha256': sha(ROOT / 'scripts/audit_aft_label_clock.py'),
        'limitations': ['endpoint_spread_is_diagnostic_not_corrected_ground_truth',
                       'missing_hours_can_make_true_recovery_unknown', 'no_labels_or_models_modified']}
    (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
