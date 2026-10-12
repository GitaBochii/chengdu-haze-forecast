from scripts.benchmark_recovery_conditional import paired_gate


def folds(row, event, tail):
    return {str(y): {'years': {str(y + 1): {'legacy': {
        'row_mae': row, 'event_equal_mae': event, 'tail_event_equal_mae': tail,
        'tail_events': ids}}}} for y, ids in ((2022, [1, 2]), (2023, [3]))}


def test_gate_uses_formal_legacy_arm_and_rejects_tail_regression():
    base = folds(10, 11, 12)
    candidate = folds(9, 10, 13)
    for fold in candidate.values():
        for year in fold['years'].values():
            year['balanced'] = {'row_mae': 0, 'event_equal_mae': 0, 'tail_event_equal_mae': 0}
    assert not paired_gate(base, candidate)['accepted']
    assert paired_gate(base, folds(9, 10, 11))['accepted']
