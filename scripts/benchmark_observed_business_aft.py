"""Fixed 2023 comparison on the complete causal business population."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from haze_forecast.observed_aft import ObservedAFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/task01_recovery_head/observed_business_dataset_v1'
OUT = ROOT / 'artifacts/task01_recovery_head/observed_business_aft_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(frame, prediction):
    exact = frame.exact_hours.notna()
    error = (frame.loc[exact, 'exact_hours'] - prediction[exact]).abs()
    block = frame.loc[exact, 'continuity_block']
    tail = frame.loc[exact, 'exact_hours'].ge(24)
    censor = ~exact & frame.informative
    return {
        'exact_rows': int(exact.sum()), 'exact_blocks': int(block.nunique()),
        'row_mae': float(error.mean()), 'block_mae': float(error.groupby(block).mean().mean()),
        'tail_24h_rows': int(tail.sum()),
        'tail_24h_block_mae': float(error[tail].groupby(block[tail]).mean().mean()),
        'informative_censored_rows': int(censor.sum()),
        'below_censor_bound_fraction': float((prediction[censor] < frame.loc[censor, 'lower_bound_hours']).mean()),
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    report = json.loads((DATA / 'report.json').read_bytes())
    names = report['feature_names']
    full = pd.read_parquet(DATA / 'features.parquet')
    origins = pd.read_parquet(DATA / 'origins.parquet')
    legacy_dir = ROOT / 'artifacts/task01_recovery_head/observed_aft_development_v1'
    legacy_report = json.loads((legacy_dir / 'report.json').read_bytes())
    legacy_path = legacy_dir / 'legacy_full.joblib'
    assert sha(legacy_path) == legacy_report['scores']['legacy_full']['checkpoint_sha256']
    paths = [DATA / n for n in ('features.parquet', 'origins.parquet', 'report.json')]
    paths += [legacy_path, legacy_dir / 'protocol.json', legacy_dir / 'report.json',
              Path(__file__), ROOT / 'src/haze_forecast/observed_aft.py',
              ROOT / 'src/haze_forecast/training.py', ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*joblib'))
    before = {str(p): sha(p) for p in paths}
    start = pd.Timestamp('2023-01-01', tz='Asia/Shanghai')
    end = pd.Timestamp('2024-01-01', tz='Asia/Shanghai')
    def cross_blocks(cutoff):
        return set(full.loc[full.timestamp < cutoff, 'continuity_block']) & set(
            full.loc[full.timestamp >= cutoff, 'continuity_block'])
    cross_start, cross_end = cross_blocks(start), cross_blocks(end)
    train_mask = (origins.timestamp < start) & (origins.label_available_at < start)
    train_mask &= ~origins.continuity_block.isin(cross_start) & origins.informative
    validation_mask = origins.timestamp.ge(start) & origins.timestamp.lt(end)
    # Score only labels available inside the development year; keep all origin rows for coverage.
    validation = origins.loc[validation_mask].copy().reset_index(drop=True)
    validation['score_allowed'] = validation.label_available_at.lt(end) & ~validation.continuity_block.isin(cross_start | cross_end)
    train = origins.loc[train_mask].copy().reset_index(drop=True)
    assert len(train) and validation.score_allowed.any()
    assert not set(train.continuity_block) & set(validation.continuity_block)
    protocol = {
        'input_sha256': before, 'features': names, 'seed': 42, 'rounds': 500,
        'train_start': '2014', 'train_cutoff': str(start), 'validation_end': str(end),
        'cross_start_blocks': sorted(cross_start), 'cross_end_blocks': sorted(cross_end),
        'arms': ['legacy_training', 'exact_only', 'observed_bounds'],
        'legacy_control': 'frozen pre-2023 legacy_full model; all arms receive identical rebuilt features',
        'selection': 'none; fixed diagnostic',
        'candidate_gate': 'both row and continuity-block MAE lower than legacy; 24h tail block MAE not worse',
        'limitations': ['exact subset cannot establish full-population MAE', '2023 is reused development data',
                       'not a deployed-model comparison', 'O3 source semantics differ',
                       'legacy training uses old labels and population; mechanistic control only'],
    }
    OUT.mkdir(parents=True)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    train.to_parquet(OUT / 'training_rows.parquet', index=False)
    validation.to_parquet(OUT / 'validation_origins.parquet', index=False)
    results = {}
    for arm in protocol['arms']:
        if arm == 'legacy_training':
            model = joblib.load(legacy_path)
            selected = None
        else:
            selected = train if arm == 'observed_bounds' else train.loc[train.exact_hours.notna()]
            model = ObservedAFTSurvivalRegressor(seed=42, nthread=1).fit_bounds(
                selected[names], selected.lower_bound_hours, selected.upper_bound_hours)
        joblib.dump(model, OUT / f'{arm}.joblib')
        prediction = model.predict(validation[names])
        assert np.isfinite(prediction).all()
        validation[arm] = prediction
        scored = validation.loc[validation.score_allowed]
        results[arm] = metrics(scored, prediction[validation.score_allowed])
        results[arm]['train_rows'] = legacy_report['scores']['legacy_full']['train_rows'] if selected is None else len(selected)
        results[arm]['model_sha256'] = sha(OUT / f'{arm}.joblib')
    baseline = results['legacy_training']
    for arm in ('exact_only', 'observed_bounds'):
        result = results[arm]
        result['passes_development_gate'] = bool(result['row_mae'] < baseline['row_mae'] and
            result['block_mae'] < baseline['block_mae'] and
            result['tail_24h_block_mae'] <= baseline['tail_24h_block_mae'])
    validation.to_parquet(OUT / 'predictions.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    result = {'status': 'completed_development_comparison', 'scores': results,
              'all_validation_origins': len(validation), 'score_allowed_origins': int(validation.score_allowed.sum()),
              'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
