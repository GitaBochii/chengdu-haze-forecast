import json

import numpy as np
import pandas as pd
import pytest

from scripts import benchmark_weather_source_ablation as runner
from scripts.benchmark_operational_gefs_ensemble import PBL_COLUMNS


@pytest.mark.parametrize('mutate', [False, True])
def test_selection_is_frozen_before_confirmation_and_input_changes_rejected(tmp_path, monkeypatch, mutate):
    processed = tmp_path / 'data/processed'
    processed.mkdir(parents=True)
    deployed = tmp_path / 'artifacts/cnemc_2025_deployment_candidate'
    deployed.mkdir(parents=True)
    (deployed / 'model_metadata.json').write_text(json.dumps({'feature_names': ['aqi', 'temperature_c']}))
    sources = [(2014, 'cnemc_chengdu_2014_training_features'), (2015, 'training_features'),
               *[(y, f'cnemc_chengdu_{y}_training_features') for y in range(2022, 2026)]]
    for year, name in sources:
        frame = pd.DataFrame({'timestamp': pd.to_datetime([f'{year}-01-02'], utc=True),
                              'event_id': [1], 'remaining_hours': [2.], 'particle_primary': [True],
                              'aqi': [120.], 'temperature_c': [10.],
                              **{c: [np.nan] for c in PBL_COLUMNS}})
        frame.to_parquet(processed / f'{name}.parquet', index=False)
    weather = tmp_path / 'weather.parquet'
    pd.DataFrame({'unused': [1]}).to_parquet(weather)
    output = tmp_path / 'result'
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'uniform_weather_rows', lambda rows, _: rows.copy())
    monkeypatch.setattr('sys.argv', ['benchmark', '--output', str(output), '--weather', str(weather)])
    calls = []

    def score(train, test, features, destination):
        calls.append(destination.stem)
        if destination.stem.endswith(('2024', '2025')):
            frozen = json.loads((output / 'selection.json').read_text())
            assert frozen['selected_arm'] == 'uniform_four'
            assert destination.stem.startswith(('baseline_', 'uniform_four_'))
        if mutate and len(calls) == 1:
            with weather.open('ab') as stream:
                stream.write(b'changed')
        value = 1. if destination.stem.startswith('uniform_four') else 2.
        return {'row_mae': value, 'event_equal_mae': value}

    monkeypatch.setattr(runner, '_score', score)
    if mutate:
        with pytest.raises(RuntimeError, match='inputs changed'):
            runner.main()
        assert not (output / 'report.json').exists()
        return
    runner.main()
    assert len(calls) == 12
    result = json.loads((output / 'report.json').read_text())
    assert result['decision']['status'] == 'research_candidate_only'
    assert result['decision']['deployment_artifact_changed'] is False
    with pytest.raises(FileExistsError):
        runner.main()
