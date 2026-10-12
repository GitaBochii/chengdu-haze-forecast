import pytest
from experiments.prithvi_wxc_probe_20260924.audit_pilot_run import validate_runs


def test_requires_all_fixed_runs_and_full_budget():
    runs = [dict(seed=seed,arm=arm,training_nll=[1.]*100) for seed in (24,42,2026)
            for arm in ('baseline','capacity_control','spatial_fusion')]
    validate_runs(runs)
    for bad in (runs[:-1],runs[:-1]+[runs[0]], [dict(r,training_nll=[1.]*99) for r in runs]):
        with pytest.raises(ValueError):
            validate_runs(bad)
