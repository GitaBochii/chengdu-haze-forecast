import hashlib
import importlib
import json
from datetime import timedelta

import pandas as pd
import pytest


def api():
    from importlib.util import find_spec

    assert find_spec('scripts.build_gefs_72h_features') is not None, 'builder missing'
    return importlib.import_module('scripts.build_gefs_72h_features')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def collection(tmp_path, request):
    base = tmp_path / 'artifacts/task01_recovery_head'
    folder = base / 'gefs_72h_collection_v1'
    source = base / 'reusable_gefs_snapshots_v1/snapshots.parquet'
    source.parent.mkdir(parents=True)
    count = getattr(request, 'param', 2)
    runs = pd.date_range('2022-01-01', periods=count, tz='UTC')
    pd.DataFrame({'run_at': runs}).to_parquet(source)
    for name in ['scripts/extract_gefs_72h_batch.py', 'scripts/audit_gefs_72h_collection.py',
                 'src/haze_forecast/grib.py', 'src/haze_forecast/forecast_sources.py',
                 'tools/wgrib2/v3.1.3/wgrib2.exe',
                 'src/haze_forecast/forecast_snapshot_reuse.py',
                 'gefs_72h_comparison_protocol_20260906.md']:
        put(tmp_path / name, name)
    protocol = {'runs': [r.isoformat() for r in runs], 'source_sha256': sha(source),
                'collector_sha256': sha(tmp_path / 'scripts/extract_gefs_72h_batch.py'),
                'decoder_sha256': sha(tmp_path / 'tools/wgrib2/v3.1.3/wgrib2.exe')}
    put(folder / 'protocol.json', protocol)
    verified = []
    for run in runs:
        destination = folder / run.strftime('%Y%m%d%H')
        products = {}
        hashes = {}
        for product in ['geavg', 'gespr', 'gec00']:
            fields = ['boundary_layer_height_m'] if product == 'gec00' else [
                'temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms']
            items = {}
            put(destination / f'{product}.idx', 'index')
            hashes[f'{product}.idx'] = sha(destination / f'{product}.idx')
            for field in fields:
                filename = f'{product}_{field}.grb2'
                put(destination / filename, [product, field])
                hashes[filename] = sha(destination / filename)
                items[field] = {'value': 8.07 if product == 'geavg' else 0.69,
                                'sha256': hashes[filename],
                                'last_modified': (run.to_pydatetime() + timedelta(hours=4)).isoformat()}
            products[product] = {'fields': items}
        record = {'run_at': run.isoformat(),
                  'valid_at': (run.to_pydatetime() + timedelta(hours=72)).isoformat(),
                  'archive_available_at_proxy': (run.to_pydatetime() + timedelta(hours=4)).isoformat(),
                  'products': products, 'output_sha256': hashes}
        put(destination / 'complete.json', record)
        verified.append({'run_at': run.isoformat(),
                         'complete_sha256': sha(destination / 'complete.json')})
    audit = {'verified_runs': count, 'verified_grib_messages': 9 * count, 'planned_runs': count,
             'protocol_sha256': sha(folder / 'protocol.json'), 'verified': verified,
             'audit_script_sha256': sha(tmp_path / 'scripts/audit_gefs_72h_collection.py'),
             'helper_sha256': {n: sha(tmp_path / 'src/haze_forecast' / n)
                               for n in ['grib.py', 'forecast_sources.py']}}
    put(folder / 'independent_audit.json', audit)
    return tmp_path, folder


def test_main_rejects_incomplete_plan_without_creating_output(collection):
    root, _ = collection
    with pytest.raises(ValueError, match='292'):
        api().build(root)
    assert not (root / 'artifacts/task01_recovery_head/gefs_72h_features_v1').exists()


@pytest.mark.parametrize('change', ['marker', 'raw', 'audit', 'missing'])
def test_rejects_stale_or_missing_evidence(collection, change):
    root, folder = collection
    if change == 'marker':
        with (folder / '2022010100/complete.json').open('a') as handle:
            handle.write(' ')
    elif change == 'raw':
        put(folder / '2022010100/geavg_temperature_c.grb2', 'changed')
    elif change == 'missing':
        (folder / 'independent_audit.json').unlink()
    else:
        audit = json.loads((folder / 'independent_audit.json').read_bytes())
        audit['verified'] = audit['verified'][:1]
        put(folder / 'independent_audit.json', audit)
    with pytest.raises((ValueError, FileNotFoundError)):
        api().validated_snapshots(root, expected_runs=2)


def test_available_boundary_values_and_missing_rows(collection):
    root, _ = collection
    module = api()
    snapshots, _ = module.validated_snapshots(root, expected_runs=2)
    origins = pd.to_datetime(['2022-01-01T04:00:00Z', '2014-01-01T00:00:00Z',
                             '2022-01-01T03:59:59Z', '2022-01-05T00:00:00Z'])
    result = module.match_origins(snapshots, origins)
    assert result.timestamp.tolist() == origins.tolist()
    assert len(result) == 4
    assert result.loc[0, 'gefs_mean_temperature_c_f072'] == 8.07
    assert result.loc[0, 'gefs_spread_temperature_c_f072'] == 0.69
    assert result.loc[0, 'forecast_known_age_hours'] == 0
    assert result.loc[0, 'forecast_remaining_lead_hours'] == 68
    assert result.loc[[1, 2, 3], module.FIELDS].isna().all().all()


def test_future_snapshot_does_not_change_past_and_invalid_origins_rejected(collection):
    root, _ = collection
    module = api()
    snapshots, _ = module.validated_snapshots(root, expected_runs=2)
    origins = pd.to_datetime(['2022-01-01T04:00:00Z'])
    pd.testing.assert_frame_equal(module.match_origins(snapshots.iloc[:1], origins),
                                  module.match_origins(snapshots, origins))
    for invalid in [pd.to_datetime(['2022-01-01']), origins.append(origins)]:
        with pytest.raises(ValueError, match='unique timezone-aware'):
            module.match_origins(snapshots, invalid)


@pytest.mark.parametrize('collection', [292], indirect=True)
def test_full_build_keeps_7890_origins_and_never_authorizes_training(collection):
    root, _ = collection
    base = root / 'artifacts/task01_recovery_head'
    origin_path = base / 'interval_business_labels_v1/origins.parquet'
    origin_path.parent.mkdir()
    origins = pd.date_range('2014-01-01', periods=7890, tz='Asia/Shanghai', freq='12h')
    pd.DataFrame({'timestamp': origins}).to_parquet(origin_path)
    api().build(root)
    output = base / 'gefs_72h_features_v1'
    report = json.loads((output / 'report.json').read_bytes())
    features = pd.read_parquet(output / 'features.parquet')
    assert features.timestamp.tolist() == origins.tolist()
    assert report['snapshots'] == 292 and report['origins'] == 7890
    assert report['formal_model_evaluation_allowed'] is False
    for name, value in report['output_sha256'].items():
        assert sha(output / name) == value


@pytest.mark.parametrize('collection', [292], indirect=True)
def test_main_stale_audit_fails_before_output_creation(collection):
    root, folder = collection
    audit = json.loads((folder / 'independent_audit.json').read_bytes())
    audit['protocol_sha256'] = 'stale'
    put(folder / 'independent_audit.json', audit)
    with pytest.raises(ValueError, match='stale audit protocol'):
        api().build(root)
    assert not (root / 'artifacts/task01_recovery_head/gefs_72h_features_v1').exists()
