"""Independent seasonal decision, fallback and pooled gate audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verify_probability import sha
from verify_forward import check, score
from verify_followup_screens import BASE, HERE, pooled, verify_hashes


def choose(y, candidates, budget):
    grid = [dict(candidate=name, threshold=i/100, **score(y, p, i/100))
            for name, p in candidates.items() for i in range(1, 102)]
    feasible = [row for row in grid if row['false_positive_rate'] <= budget]
    # All names have arm prefixes. The frozen protocol specifies first input
    # candidate after the recall, precision, FPR and threshold objectives tie.
    objective = lambda r: (r['recall'] or 0, r['precision'] or 0, -r['false_positive_rate'], r['threshold'])
    best = max(feasible, key=objective)
    ties = [r['candidate'] for r in feasible if objective(r) == objective(best)]
    return best, grid, ties


def decision_score(y, predictions):
    result = score(y, predictions.astype(float), .5)
    result.pop('brier')
    result.pop('average_precision')
    return result


def main():
    run = HERE/'season_runs/20260929T103011739567Z'
    report = json.loads((run/'report.json').read_text('utf-8'))
    assert report['status'] == 'completed' and report['deployed'] is False
    hashes = {}
    verify_hashes(run/'report.json', hashes, set())
    base = json.loads((BASE/'report.json').read_text('utf-8'))
    prior = json.loads((BASE/'independent_audit.json').read_text('utf-8'))
    assert prior['status'] == 'passed' and prior['report_sha256'] == sha(BASE/'report.json')
    paths = {Path(p).parent.parent.name: Path(p).parent for p in report['hashes'] if Path(p).name == 'report.json'}
    audit = dict(status='passed', report_sha256=sha(run/'report.json'), verifier_sha256=sha(Path(__file__)),
                 helper_hashes={name: sha(HERE/name) for name in ('verify_probability.py', 'verify_forward.py', 'verify_followup_screens.py')},
                 hashes_checked=hashes, baseline_independent_audit_sha256=sha(BASE/'independent_audit.json'),
                 scope='Independent dev-only seasonal grid reconstruction, exact fallback rules, full decisions and test metrics; baseline source-verified metadata reused.',
                 limitations='Retrospective overlapping origins, not a new blind holdout or independent-event inference. No retraining.', strategies={})
    assert set(report['strategies']) == {'baseline_season', 'full_season'}
    for strategy, record in report['strategies'].items():
        assert set(record['folds']) == set(base['folds'])
        folds, scores = {}, {}
        for key, fold in record['folds'].items():
            meta = pd.read_parquet(BASE/f'{key}_origins.parquet')
            y = meta.label.to_numpy(bool)
            dev, test = (meta.partition.eq(part).to_numpy() for part in ('dev', 'test'))
            assert (dev | test).all() and not (dev & test).any()
            candidates = {}
            arms = ('baseline',) if strategy == 'baseline_season' else ('baseline', 'trend', 'chemistry', 'spatial')
            for arm in arms:
                for weight in ('unweighted', 'balanced'):
                    path = (BASE/f'{key}_{weight}.npy' if arm == 'baseline' else
                            paths['spatial_runs']/f'{key}_{weight}.npy' if arm == 'spatial' else
                            paths['feature_runs']/f'{key}_{arm}_{weight}.npy')
                    p = np.load(path, allow_pickle=False)
                    assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                    candidates[f'{arm}_{weight}'] = p
            budget = .1 if key.endswith('_onset') else .05
            best, global_grid, global_ties = choose(y[dev], {n:p[dev] for n,p in candidates.items()}, budget)
            check(best, fold['global_choice'], f'{strategy}/{key}/global')
            decisions = np.zeros(len(y), bool)
            seasons = {}
            for season in ('DJF', 'MAM', 'JJA', 'SON'):
                group = meta.season.eq(season).to_numpy()
                subset = dev & group
                positives, negatives = int(y[subset].sum()), int((~y[subset]).sum())
                fallback = positives < 20 or negatives < 50
                if fallback:
                    chosen, grid, ties = best, [], global_ties
                else:
                    chosen, grid, ties = choose(y[subset], {n:p[subset] for n,p in candidates.items()}, budget)
                decisions[group] = candidates[chosen['candidate']][group] >= chosen['threshold']
                m = test & group
                reconstructed = dict(fallback=fallback, dev_positive=positives, dev_negative=negatives,
                                     chosen=chosen, development_grid=grid,
                                     test=decision_score(y[m], decisions[m]) if m.any() else None)
                check(reconstructed, fold['seasons'][season], f'{strategy}/{key}/{season}')
                seasons[season] = dict(fallback=fallback, dev_positive=positives, dev_negative=negatives,
                                       chosen=chosen, grid_rows_verified=len(grid), top_tied_candidates=ties)
            saved = np.load(run/f'{strategy}_{key}_decisions.npy', allow_pickle=False)
            assert saved.dtype == np.dtype(bool)
            np.testing.assert_array_equal(decisions, saved)
            test_score = decision_score(y[test], decisions[test])
            check(test_score, fold['test'], f'{strategy}/{key}/test')
            scores[key] = test_score
            folds[key] = dict(global_choice=best, global_grid_rows_recomputed=len(global_grid),
                              global_tied_candidates=global_ties, seasons=seasons, test=test_score,
                              saved_decisions_verified=len(decisions))
        totals = {}
        for task in ('onset', 'recovery'):
            keys = [f'{year}_{task}' for year in (2021, 2022, 2023)]
            baseline = pooled([prior['folds'][key]['test']['overall'] for key in keys])
            new = pooled([scores[key] for key in keys])
            years = sum(scores[key]['recall'] > prior['folds'][key]['test']['overall']['recall'] for key in keys)
            gate = bool(new['recall'] >= baseline['recall']+.05 and new['false_positive_rate'] <= baseline['false_positive_rate']
                        and new['false_positive_rate'] <= (.1 if task == 'onset' else .05) and years >= 2)
            totals[task] = dict(baseline=baseline, new=new, years_recall_improved=years, meaningful_gate=gate)
        check(totals, record['pooled'], f'{strategy}/pooled')
        audit['strategies'][strategy] = dict(folds=folds, pooled=totals)
        print(json.dumps(dict(strategy=strategy, pooled=totals)))
    (run/'independent_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'PASS: {run / "independent_audit.json"}')


if __name__ == '__main__':
    main()
