import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from scripts import benchmark_recovery_selection as runner
from scripts.benchmark_recovery_head import RecoveryWindows


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.risk = torch.nn.Linear(5, 67)
        self.curve = torch.nn.Linear(5, 2)

    def forward(self, x, active):
        values = x[:, -1]
        return self.curve(values)[:, None].expand(-1, 72, -1, -1), self.risk(values.mean(1))


def tiny_data():
    rng = np.random.default_rng(42)
    times = pd.date_range('2022-01-01', periods=180, freq='h', tz='Asia/Shanghai')
    panel = {'timestamps': times, 'inputs': rng.normal(size=(180, 2, 5)).astype('float32'),
             'targets': np.full((180, 2, 2), 150., dtype='float32'),
             'scaler': SimpleNamespace(mean=np.zeros(5), scale=np.ones(5))}
    aqi = np.full(180, 150.)
    aqi[5:11] = 80.
    labels = pd.DataFrame({'event_id': [1., 2.], 'remaining_hours': [3., 80.]},
                          index=times[[2, 100]])
    dataset = RecoveryWindows(panel, np.array([2, 100]), 2, 72, aqi, np.ones(180, dtype=bool))
    args = argparse.Namespace(seed=42, epochs=2, batch_size=2, learning_rate=.001,
                              auxiliary_weight=.1, horizon=72, validation_year=2022)
    return panel, aqi, labels, dataset, args


def test_train_same_candidate_pool_ties_keep_earlier_and_freeze_pair(tmp_path, monkeypatch):
    panel, aqi, labels, dataset, args = tiny_data()
    # Freeze just the selection metric so both rules exercise exact ties across real training epochs.
    original = runner.selection_metrics
    fixed = original(runner.duration_records(dataset.origins, panel, labels, aqi,
                                             np.array([3., 68.]), 72))
    monkeypatch.setattr(runner, 'selection_metrics', lambda records: fixed)
    result = runner.train_selection_pair(TinyNetwork(), dataset, dataset, panel, labels,
                                         aqi, args, tmp_path, torch.device('cpu'))
    assert len(result['trace']) == 2
    for rule in ('legacy', 'balanced'):
        assert result['selections'][rule]['epoch'] == 1
        assert result['selections'][rule]['candidate_epochs'] == [1, 2]
    evaluated = runner.evaluate_selection_pair(TinyNetwork(), dataset, panel, labels, aqi,
                                              args, tmp_path, torch.device('cpu'), 2023,
                                              result['frozen'])
    assert evaluated['paired_rows'] is True
    assert evaluated['legacy'] == evaluated['balanced']
    left = pd.read_parquet(tmp_path / 'legacy_2023_recovery.parquet')
    right = pd.read_parquet(tmp_path / 'balanced_2023_recovery.parquet')
    pd.testing.assert_frame_equal(left, right)
    with np.load(tmp_path / 'legacy_2023_distribution.npz') as saved:
        np.testing.assert_array_equal(saved['origin_ns'], panel['timestamps'][dataset.origins].asi8)
        assert saved['probability'].shape == (2, 68)
    with (tmp_path / 'legacy_state.pt').open('ab') as stream:
        stream.write(b'mutated')
    with pytest.raises(RuntimeError, match='frozen'):
        runner.evaluate_selection_pair(TinyNetwork(), dataset, panel, labels, aqi,
                                       args, tmp_path, torch.device('cpu'), 2024, result['frozen'])


def test_rules_can_select_different_epochs_from_identical_trajectory(tmp_path, monkeypatch):
    panel, aqi, labels, dataset, args = tiny_data()
    initial = runner.selection_metrics(runner.duration_records(
        dataset.origins, panel, labels, aqi, np.array([3., 68.]), 72))
    scores = iter([{**initial, 'row_mae': 3., 'event_equal_mae': 3., 'balanced_mae': 8.},
                   {**initial, 'row_mae': 4., 'event_equal_mae': 4., 'balanced_mae': 5.}])
    monkeypatch.setattr(runner, 'selection_metrics', lambda records: next(scores))
    result = runner.train_selection_pair(TinyNetwork(), dataset, dataset, panel, labels,
                                         aqi, args, tmp_path, torch.device('cpu'))
    assert result['selections']['legacy']['epoch'] == 1
    assert result['selections']['balanced']['epoch'] == 2
    assert result['selections']['legacy']['candidate_epochs'] == [1, 2]
    assert result['selections']['balanced']['candidate_epochs'] == [1, 2]


@pytest.mark.parametrize('accepted', [False, True])
def test_development_gate_controls_final_and_freezes_rule(tmp_path, monkeypatch, accepted):
    args = argparse.Namespace(output=str(tmp_path / 'run'), data='unused', seed=42, epochs=2,
                              batch_size=2, learning_rate=.001, auxiliary_weight=.1,
                              history=24, horizon=72, device='cpu')
    calls = []

    def fold(args, validation_year, test_years, output, device):
        calls.append((validation_year, tuple(test_years)))
        if validation_year == 2024:
            assert (output.parent / 'rule_selection.json').exists()
        return {'years': {str(year): {'legacy': {}, 'balanced': {}} for year in test_years}}

    monkeypatch.setattr(runner, 'run_fold', fold)
    monkeypatch.setattr(runner, 'hashes', lambda path: {'stable': 'hash'})
    monkeypatch.setattr(runner, 'development_gate', lambda pairs: {'accepted': accepted})
    result = runner.run_experiment(args)
    assert calls[:2] == [(2022, (2023,)), (2023, (2024,))]
    assert len(calls) == (3 if accepted else 2)
    assert result['final_ran'] is accepted
    assert result['development_gate']['accepted'] is accepted
    report = json.loads((tmp_path / 'run/report.json').read_text('utf-8'))
    assert report['inputs_unchanged'] and report['deployment_unchanged']
    with pytest.raises(FileExistsError):
        runner.run_experiment(args)


def test_failed_run_still_writes_report(tmp_path, monkeypatch):
    args = argparse.Namespace(output=str(tmp_path / 'run'), data='unused', seed=42, epochs=2,
                              batch_size=2, learning_rate=.001, auxiliary_weight=.1,
                              history=24, horizon=72, device='cpu')
    monkeypatch.setattr(runner, 'hashes', lambda path: {'stable': 'hash'})
    monkeypatch.setattr(runner, 'run_fold', lambda *args: (_ for _ in ()).throw(ValueError('test failure')))
    with pytest.raises(ValueError, match='test failure'):
        runner.run_experiment(args)
    report = json.loads((tmp_path / 'run/report.json').read_text('utf-8'))
    assert report['decision'] == 'error'
    assert report['development_gate'] is None
    assert report['final_ran'] is False
