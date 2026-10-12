import numpy as np
import pandas as pd
import pytest

from haze_forecast.operational_gefs_benchmark import (
    publication_audited_features,
    safe_event_year_split,
)
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES


def test_split_uses_local_year_and_confirmation_not_last_sample():
    frame = pd.DataFrame({'timestamp': pd.to_datetime(['2023-01-01T00:00Z',
                                                      '2023-12-31T15:00Z',
                                                      '2023-12-31T18:00Z']),
                          'event_key': ['old', 'crossing', 'new'], 'remaining_hours': [2., 2., 1.]})
    train, test = safe_event_year_split(frame, 2024)
    assert train.event_key.tolist() == ['old']
    assert test.event_key.tolist() == ['new']


def test_raw_key_complete_but_unverified_features_rejected():
    frame = pd.DataFrame({'timestamp': pd.to_datetime(['2024-01-01'], utc=True), 'event_id': [1],
                          **{field: [np.nan] for field in OPERATIONAL_GEFS_FEATURES}})
    with pytest.raises(ValueError, match='availability'):
        publication_audited_features(frame)
    frame['gefs_availability_policy'] = 'archive_timestamp_proxy'
    frame['gefs_availability_audited'] = True
    frame['gefs_extraction_verified'] = False
    with pytest.raises(ValueError, match='extraction'):
        publication_audited_features(frame)
    frame['gefs_extraction_verified'] = True
    assert len(publication_audited_features(frame)) == 1
    with pytest.raises(ValueError, match='duplicate'):
        publication_audited_features(pd.concat([frame, frame]))


@pytest.mark.parametrize('mutate_input', [False, True])
def test_benchmark_freezes_2023_before_scoring_only_selected_confirmation(tmp_path, monkeypatch, mutate_input):
    import json

    from scripts import benchmark_operational_gefs_ensemble as runner
    processed = tmp_path / 'data' / 'processed'
    processed.mkdir(parents=True)
    deployed = tmp_path / 'artifacts' / 'cnemc_2025_deployment_candidate'
    deployed.mkdir(parents=True)
    (deployed / 'model_metadata.json').write_text(json.dumps({'feature_names': ['value']}))
    for year, name in [(2014, 'cnemc_chengdu_2014_training_features'), (2015, 'training_features'),
                       *[(y, f'cnemc_chengdu_{y}_training_features') for y in range(2022, 2026)]]:
        frame = pd.DataFrame({'timestamp': pd.to_datetime([f'{year}-01-02'], utc=True),
                              'event_id': [1], 'particle_primary': [True], 'remaining_hours': [1.],
                              'value': [1.], **{c: [np.nan] for c in runner.PBL_COLUMNS},
                              **{c: [np.nan] for c in OPERATIONAL_GEFS_FEATURES}})
        frame.to_parquet(processed / f'{name}.parquet', index=False)
    output = tmp_path / 'results' / 'report.json'
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'PROCESSED', processed)
    monkeypatch.setattr(runner, '_attach_operational_gefs', lambda frame, year, directory: frame)
    monkeypatch.setattr('sys.argv', ['benchmark', '--output', str(output)])
    calls = []
    def score(train, test, features, record_path):
        calls.append(record_path.stem)
        if mutate_input and len(calls) == 1:
            with (processed / 'training_features.parquet').open('ab') as stream:
                stream.write(b'changed after load')
        if not record_path.stem.endswith('2023'):
            selected = json.loads(output.with_name('report_selection_2023.json').read_text())
            assert selected['selected_variant'] == 'ensemble_6'
            assert record_path.stem.startswith(('aft_', 'ensemble_6_'))
        value = 1. if record_path.stem.startswith('ensemble_6_') and not record_path.stem.startswith('ensemble_6_24') else 2.
        return {'row_mae': value, 'event_equal_mae': value}
    monkeypatch.setattr(runner, '_score', score)
    if mutate_input:
        with pytest.raises(RuntimeError, match='inputs changed'):
            runner.main()
        assert not output.exists()
        return
    runner.main()
    assert len(calls) == 8
    report = json.loads(output.read_text())
    assert set(report['scores']['2024']) == {'aft', 'ensemble_6'}
    assert report['decision']['status'] == 'research_candidate_only'
    with pytest.raises(FileExistsError):
        runner.main()
