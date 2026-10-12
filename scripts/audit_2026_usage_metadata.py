"""Extract historical evaluation membership without reading score values."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'artifacts/task01_recovery_head/usage_2026_metadata_20260918'
    if out.exists():
        raise FileExistsError(out)
    records = []
    for path in sorted((ROOT / 'artifacts/regional_air_models').glob('*/report.json')):
        report = json.loads(path.read_bytes())
        if 'configuration' not in report:
            continue
        years = {name: list(value.get('years', {})) for name, value in report.get('models', {}).items()}
        records.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'training_last_target': report.get('training_last_target'),
                        'validation_last_target': report.get('validation_last_target'),
                        'partitions': report.get('partitions'), 'model_year_keys': years,
                        'limitations_2026': [x for x in report.get('limitations', []) if '2026' in x]})
    assert records and any('2026' in keys for r in records for keys in r['model_year_keys'].values())
    for r in records:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256']
    sources = [Path(__file__).resolve(), ROOT / 'scripts/benchmark_regional_air_models.py',
               ROOT / 'scripts/train_regional_rolling.py', ROOT / 'src/haze_forecast/regional_air_experiment.py']
    result = {'reports': records, 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'conclusion': '2026 evaluation results already exist; not a pristine holdout. '
                            'This metadata audit does not replay training or prove every past use.'}
    out.mkdir()
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({Path(r['path']).parent.name: r['model_year_keys'] for r in records}, indent=2))


if __name__ == '__main__':
    main()
