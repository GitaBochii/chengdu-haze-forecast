"""Fixed 24-hour classification diagnostic using only identifiable interval labels."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import ndtr
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

from scripts.benchmark_observed_business_aft import ROOT, sha


def main():
    base = ROOT / 'artifacts/task01_recovery_head'
    dev = base / 'interval_history_counts_v1'
    train_path = base / 'interval_business_aft_v1/training_rows.parquet'
    history_path = base / 'trajectory_reset_audit_v1/origins.parquet'
    out = base / 'long_duration_classifier_v1'
    if out.exists():
        raise FileExistsError(out)
    paths = [train_path, history_path, dev / 'expanded_predictions.parquet', dev / 'protocol.json',
             Path(__file__), ROOT / 'src/haze_forecast/app.py']
    paths += list((ROOT / 'artifacts/cnemc_2025_deployment_candidate').glob('*.joblib'))
    before = {str(p): sha(p) for p in paths}
    names = json.loads((dev / 'protocol.json').read_bytes())['features']
    train = pd.read_parquet(train_path)
    history = pd.read_parquet(history_path).set_index('timestamp')
    train[names[87:]] = history.loc[train.timestamp, names[87:]].to_numpy()
    identifiable = train.lower_bound_hours.ge(24) | train.upper_bound_hours.lt(24)
    selected = train.loc[identifiable]
    target = selected.lower_bound_hours.ge(24).astype(int)
    val = pd.read_parquet(dev / 'expanded_predictions.parquet')
    val[names[87:]] = history.loc[val.timestamp, names[87:]].to_numpy()
    assert selected.label_available_at.lt(pd.Timestamp('2023-01-01', tz='Asia/Shanghai')).all()
    assert not set(selected.continuity_block) & set(val.continuity_block)
    out.mkdir(parents=True)
    protocol = {'input_sha256': before, 'features': names, 'threshold_hours': 24,
                'training_rows': len(selected), 'excluded_ambiguous_rows': int((~identifiable).sum()),
                'positive_rows': int(target.sum()), 'selection': 'fixed classifier, no tuning or routing to regressors',
                'settings': '300 trees, learning_rate .04, 15 leaves, min_child_samples40, lambda2, seed42',
                'limitations': ['2023 reused development data', 'identifiable subset may be selected',
                                'classification quality does not prove duration MAE improvement']}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    model = LGBMClassifier(n_estimators=300, learning_rate=.04, num_leaves=15,
                          min_child_samples=40, reg_lambda=2, random_state=42, n_jobs=1, verbosity=-1)
    model.fit(selected[names].astype(float), target)
    joblib.dump(model, out / 'classifier.joblib')
    prob = model.predict_proba(val[names].astype(float))[:, 1]
    np.testing.assert_array_equal(prob, joblib.load(out / 'classifier.joblib').predict_proba(val[names].astype(float))[:, 1])
    val['classifier_probability'] = prob
    # Existing normal AFT uses scale 1.2 and its prediction is exp(location).
    val['aft_probability'] = ndtr((np.log(val.history_counts_aft) - np.log(24)) / 1.2)
    usable = val.score_allowed & (val.lower_bound_hours.ge(24) | val.upper_bound_hours.lt(24))
    scored = val.loc[usable].copy()
    y = scored.lower_bound_hours.ge(24).astype(int)
    weights = 1 / scored.groupby('continuity_block').timestamp.transform('size')
    scores = {}
    for arm in ('classifier_probability', 'aft_probability'):
        p = scored[arm]
        scores[arm] = {'auc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'brier': float(brier_score_loss(y, p)),
            'block_balanced_brier': float(np.average((p-y)**2, weights=weights)),
            'confusion_at_fixed_half': confusion_matrix(y, p.ge(.5), labels=[0, 1]).tolist()}
    val.to_parquet(out / 'predictions.parquet', index=False)
    selected[['timestamp', 'continuity_block', 'lower_bound_hours', 'upper_bound_hours', 'label_available_at']].to_parquet(out / 'training_membership.parquet', index=False)
    assert before == {str(p): sha(p) for p in paths}
    report = {'scores': scores, 'scored_rows': len(scored), 'positive_rows': int(y.sum()),
              'scored_blocks': int(scored.continuity_block.nunique()), 'inputs_unchanged': True,
              'output_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
