import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from scripts.train_recovery_weather_formal import (
    FormalWeatherWindows,
    fit_weather,
    fixed_orders,
    load_ready_manifest,
    select_key,
    train_arm,
    validated_distribution,
)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_validation_rejects_nonfinite_logits_before_median(bad):
    logits = torch.zeros(2, 67)
    logits[0, 3] = bad
    with pytest.raises(ValueError, match='finite logits'):
        validated_distribution(logits)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0., -1., 2.])
def test_validation_rejects_invalid_probability(monkeypatch, bad):
    import scripts.train_recovery_weather_formal as runner

    monkeypatch.setattr(runner, 'hazard_distribution', lambda logits: torch.full((len(logits), 68), bad))
    with pytest.raises(ValueError, match='distribution'):
        validated_distribution(torch.zeros(2, 67))


def test_validated_distribution_is_normalized():
    probability = validated_distribution(torch.zeros(2, 67))
    torch.testing.assert_close(probability.sum(1), torch.ones(2))


def test_weather_scaler_training_only_and_complete_required():
    times = pd.to_datetime(['2022-02-01T00:00:00Z', '2023-02-01T00:00:00Z'])
    rows = []
    for issue in times:
        for ahead in range(3, 73, 3):
            rows.append({'issue_utc': issue, 'run_utc': issue - pd.Timedelta(hours=12),
                         'available_at_utc': issue - pd.Timedelta(hours=6),
                         'valid_utc': issue + pd.Timedelta(hours=ahead),
                         'temperature_c': 10 if issue.year == 2022 else 1000,
                         'relative_humidity_pct': 20, 'wind_u_ms': 2, 'wind_v_ms': 3})
    frame = pd.DataFrame(rows)
    values, mask, scaler = fit_weather(frame, times)
    assert scaler.mean.tolist() == [10., 20., 2., 3.]
    assert values[0].sum() == 0
    assert values[1, 0, 0] == 990
    assert mask.all()
    with pytest.raises(ValueError, match='complete'):
        fit_weather(frame.iloc[:-1], times)


def test_orders_are_identical_complete_and_seeded():
    first = fixed_orders(19, 3, 42)
    np.testing.assert_array_equal(first, fixed_orders(19, 3, 42))
    assert not np.array_equal(first[0], first[1])
    for row in first:
        np.testing.assert_array_equal(np.sort(row), np.arange(19))
    assert select_key({'event_equal_mae': 2., 'row_mae': 3.}) == (2., 3.)


def test_readiness_gate_rejects_partial_collection(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('issue_utc,url\n2022-01-01T00:00:00Z,a\n')
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'status': 'ready_for_formal_training',
                               'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                               'passed_objects': 224, 'failed_objects': 0, 'pending_objects': 4702,
                               'complete_origins': 8}))
    with pytest.raises(ValueError, match='full collection'):
        load_ready_manifest(gate, manifest, tmp_path / 'results', {})


def test_unrequested_windows_receive_explicit_all_missing_weather(monkeypatch):
    from scripts.benchmark_recovery_head import RecoveryWindows

    monkeypatch.setattr(RecoveryWindows, '__getitem__', lambda self, item: (torch.tensor(item),))
    dataset = object.__new__(FormalWeatherWindows)
    dataset.origins = np.array([10, 20])
    dataset.weather_by_origin = {10: (torch.ones(24, 4), torch.ones(24, 4, dtype=torch.bool))}
    dataset.missing_weather = (torch.zeros(24, 4), torch.zeros(24, 4, dtype=torch.bool))
    assert dataset[0][-1].all()
    assert not dataset[1][-1].any()
    assert dataset[1][-2].sum() == 0
    assert dataset[1][0] == 1


class TinyWindows(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return (torch.ones(1, 1, 1), torch.ones(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool),
                torch.ones(1, 2, dtype=torch.bool), torch.tensor([1., 0.]),
                torch.tensor([True, False]), torch.tensor(index % 2 == 0),
                torch.zeros(24, 4), torch.zeros(24, 4, dtype=torch.bool))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, x, active):
        return self.value.expand(len(x), 1, 1, 1), self.value.expand(len(x), 2)


def test_training_full_epochs_tie_earliest_and_keeps_noneligible(tmp_path):
    args = SimpleNamespace(epochs=2, batch_size=3, learning_rate=.001, auxiliary_weight=.1)
    calls = []

    def validation(model):
        calls.append(float(model.value.detach()))
        return {'event_equal_mae': 2., 'row_mae': 3., 'tail_event_equal_mae': 4.}

    report = train_arm(TinyModel(), TinyWindows(), fixed_orders(4, 2, 42),
                       args, tmp_path, torch.device('cpu'), validation)
    assert len(calls) == 2
    assert report['selection']['epoch'] == 1
    assert report['selection']['candidate_epochs'] == [1, 2]
    trace = json.loads((tmp_path / 'training_trace.json').read_text())
    assert all(row['training_windows'] == 4 for row in trace)
    assert all(row['informative_survival_origins'] == 2 for row in trace)
    assert all(row['concentration_elements'] == 4 for row in trace)
    assert calls[0] != calls[1]
