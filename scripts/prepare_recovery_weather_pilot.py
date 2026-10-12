"""Build and verify four weather input tensors, without fitting a forecast model."""

import hashlib
import json

import numpy as np
import pandas as pd

from haze_forecast.recovery_weather_inputs import (
    WEATHER_AHEAD_HOURS,
    WEATHER_FEATURES,
    WeatherScaler,
    build_weather_inputs,
)
from scripts.benchmark_recovery_selection import ROOT
from scripts.benchmark_regional_air_models import write_json


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    sources = [base / 'future_weather_sequence_fields_v1/report.json',
               base / 'future_weather_autumn_fields_v1/report_repaired.json']
    output = base / 'future_weather_inputs_v1'
    output.mkdir(exist_ok=False)
    rows, hashes = [], {}
    for source in sources:
        payload = source.read_bytes()
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
        report = json.loads(payload)
        if not report['source_unchanged']:
            raise ValueError('source was changed during extraction')
        for record in report['records']:
            if record['status'] != 'pass' or set(record['fields']) != set(WEATHER_FEATURES):
                raise ValueError('incomplete pilot record')
            for field in record['fields'].values():
                path = (ROOT / field['path']).resolve()
                if not path.is_relative_to(source.parent.resolve()):
                    raise ValueError('GRIB outside source directory')
                if hashlib.sha256(path.read_bytes()).hexdigest() != field['sha256']:
                    raise ValueError('GRIB checksum changed')
            rows.append({**{key: record['probe'][key] for key in ('issue_utc', 'run_utc', 'valid_utc')},
                         'available_at_utc': pd.Timestamp(record['head']['last_modified']).isoformat(),
                         **{name: field['value'] for name, field in record['fields'].items()}})
    frame = pd.DataFrame(rows)
    origins = sorted(pd.to_datetime(frame.issue_utc, utc=True).unique())
    values, mask = build_weather_inputs(frame, origins)
    if values.shape != (4, 24, 4) or not mask.all():
        raise ValueError('expected four complete audited origin sequences')
    training = np.array([origin.year == 2022 for origin in origins])
    scaler = WeatherScaler.fit(values, mask, training)
    changed = values.copy()
    changed[~training] += 1e6
    check = WeatherScaler.fit(changed, mask, training)
    np.testing.assert_array_equal(scaler.mean, check.mean)
    np.testing.assert_array_equal(scaler.scale, check.scale)
    normalized = scaler.transform(values, mask)
    np.savez_compressed(output / 'weather_inputs.npz', values=values, mask=mask,
                        normalized=normalized, training_rows=training,
                        origin_ns=np.array([origin.value for origin in origins]),
                        ahead_hours=np.array(WEATHER_AHEAD_HOURS),
                        mean=scaler.mean, scale=scaler.scale)
    with np.load(output / 'weather_inputs.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['values'], values)
        np.testing.assert_array_equal(saved['mask'], mask)
        np.testing.assert_array_equal(saved['normalized'], normalized)
    report = {'status': 'pass', 'shape': list(values.shape), 'observed_cells': int(mask.sum()),
              'features': list(WEATHER_FEATURES), 'ahead_hours': list(WEATHER_AHEAD_HOURS),
              'origins_utc': [origin.isoformat() for origin in origins],
              'scaler_training_year': 2022, 'validation_perturbation_invariance': True,
              'mean': scaler.mean.tolist(), 'scale': scaler.scale.tolist(), 'sources': hashes,
              'limitations': ['interface_smoke_test_only', 'four_origins_not_training_corpus',
                              'archive_proxy_not_first_publication', 'no_model_training_or_metrics']}
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
