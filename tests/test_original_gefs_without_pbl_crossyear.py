import pandas as pd
import pytest

from scripts.benchmark_original_gefs_without_pbl_crossyear import (
    candidate_features,
    crossyear_gate,
    require_development,
    validate_split,
)


def test_features_remove_only_history_pbl():
    old = [f'x{i}' for i in range(90)] + ['gefs_control_boundary_layer_height_m_f024']
    old += [f'boundary_layer_height_m_{i}' for i in range(13)]
    kept = old[:91]
    assert candidate_features(old, kept) == kept
    with pytest.raises(ValueError):
        candidate_features(old, kept[:-1])


@pytest.mark.parametrize('passed', [False, None, 1])
def test_development_fails_closed(passed):
    with pytest.raises(ValueError):
        require_development({'passed': passed})


def test_pass_is_not_deployment_permission():
    result = crossyear_gate({str(y): {'common': {
        'no_history_pbl': {'row_mae': 1, 'block_mae': 1, 'tail_24h_block_mae': 1},
        'forecast_aft': {'row_mae': 2, 'block_mae': 2, 'tail_24h_block_mae': 1},
        'strict_prediction_hours': {'row_mae': 2, 'block_mae': 2, 'tail_24h_block_mae': 2},
    }} for y in (2024, 2025)})
    assert result['passed']
    assert result['deployment_authorized'] is False


@pytest.mark.parametrize('metric,value', [('row_mae', 2), ('block_mae', 2),
                                          ('tail_24h_block_mae', 1.1),
                                          ('row_mae', float('nan'))])
def test_gate_rejects_one_year_failure(metric, value):
    years = {str(y): {'common': {
        'no_history_pbl': {'row_mae': 1, 'block_mae': 1, 'tail_24h_block_mae': 1},
        'forecast_aft': {'row_mae': 2, 'block_mae': 2, 'tail_24h_block_mae': 1},
        'strict_prediction_hours': {'row_mae': 2, 'block_mae': 2, 'tail_24h_block_mae': 2},
    }} for y in (2024, 2025)}
    years['2025']['common']['no_history_pbl'][metric] = value
    assert crossyear_gate(years)['passed'] is False


@pytest.mark.parametrize('kind', ['timestamp', 'label_available_at', 'block'])
def test_split_rejects_leakage(kind):
    cutoff = pd.Timestamp('2024-01-01', tz='Asia/Shanghai')
    train = pd.DataFrame({'timestamp': [cutoff - pd.Timedelta(days=1)],
                          'label_available_at': [cutoff - pd.Timedelta(hours=1)],
                          'continuity_block': [1]})
    val = pd.DataFrame({'timestamp': [cutoff], 'continuity_block': [2]})
    validate_split(train, val, 2024)
    if kind == 'block':
        train['continuity_block'] = 2
    else:
        train[kind] = cutoff
    with pytest.raises(ValueError):
        validate_split(train, val, 2024)
