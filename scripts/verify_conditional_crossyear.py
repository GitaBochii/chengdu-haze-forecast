"""Independently check frozen cross-year probabilities and scoring identities."""

import json

import numpy as np
import pandas as pd
import torch

from haze_forecast.history_state_residual import HistoryStateResidual
from scripts.benchmark_recovery_head import ROOT
from scripts.benchmark_regional_air_models import write_json
from scripts.train_recovery_weather_expanded import hashes_unchanged
from scripts.train_recovery_weather_formal import sha
from scripts.verify_conditional_recovery_v2 import conditional_log_mass, score_log_mass


def main():
    if not __debug__:
        raise RuntimeError('verification requires assertions')
    run = ROOT / 'artifacts/task01_recovery_head/conditional_crossyear_diagnostic_v1'
    source = ROOT / 'artifacts/task01_recovery_head/conditional_recovery_v1'
    report = json.loads((run / 'report.json').read_bytes())
    assert report['status'] == 'completed_frozen_diagnostic_not_promotion'
    assert report['evaluated_years'] == [2024, 2025]
    assert report['inputs_before'] == report['inputs_after']
    assert report['model_digest_before'] == report['model_digest_after']
    assert hashes_unchanged(report['source_sha256'])
    output = {}
    for year in (2024, 2025):
        folder = run / str(year)
        annual = report['years'][str(year)]
        for name, value in annual['output_sha256'].items():
            assert sha(folder / name) == value
        data = np.load(folder / 'states.npz', allow_pickle=False)
        identity = pd.read_parquet(folder / 'identity.parquet')
        np.testing.assert_array_equal(pd.DatetimeIndex(identity.timestamp).asi8, data['origin_ns'])
        times = pd.DatetimeIndex(identity.timestamp)
        assert (times.year == year).all() and not times.has_duplicates
        assert (times.asi8 + 72 * 3600 * 10**9 < pd.Timestamp(f'{year+1}-01-01', tz=times.tz).value).all()
        state = torch.from_numpy(data['state']).cuda()
        logits = data['logits']
        base_log = conditional_log_mass(logits, np.zeros_like(logits))
        annual_result = {}
        for arm in ('base', 'linear_all', 'nonlinear_all'):
            if arm == 'base':
                log_mass = base_log
            else:
                model = HistoryStateResidual(nonlinear=arm == 'nonlinear_all').cuda().eval()
                model.load_state_dict(torch.load(source / arm / 'selected_state.pt', weights_only=True))
                with torch.no_grad():
                    delta = model(state).cpu().numpy()
                log_mass = conditional_log_mass(logits, delta)
            saved = np.load(folder / f'{arm}_log_probability.npy', allow_pickle=False)
            np.testing.assert_allclose(log_mass, saved, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(log_mass[:, -1], base_log[:, -1])
            median, _, metrics = score_log_mass(log_mass, identity)
            frame = pd.read_parquet(folder / f'{arm}.parquet')
            columns = ['timestamp', 'event_id', 'actual_hours', 'label_verified_in_scoring_window']
            pd.testing.assert_frame_equal(frame[columns], identity[columns])
            np.testing.assert_array_equal(median, frame.predicted_hours)
            for key, value in metrics.items():
                if key.endswith('mae'):
                    np.testing.assert_allclose(value, annual['arms'][arm][key], atol=1e-12)
                else:
                    assert value == annual['arms'][arm][key]
            annual_result[arm] = {'metrics': metrics,
                'independent_log_probability_max_difference': float(abs(log_mass - saved).max())}
        output[str(year)] = annual_result
    assert hashes_unchanged(report['source_sha256'])
    result = {'verification': 'PASS', 'years': output,
        'report_sha256': sha(run / 'report.json'),
        'verifier_sha256': sha(ROOT / 'scripts/verify_conditional_crossyear.py'),
        'limitations': ['backbone_state_cache_not_independently_recomputed',
                       'diagnostic_not_promotion', 'previously_inspected_years']}
    write_json(run / 'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
