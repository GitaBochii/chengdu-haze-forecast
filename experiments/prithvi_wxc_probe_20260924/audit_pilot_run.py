"""Reload every fixed pilot run; compare original and optimized model paths."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from haze_forecast.regional_recovery_network import RecoveryPM25GNN
from haze_forecast.recovery_survival import hazard_median
from experiments.prithvi_wxc_probe_20260924.fusion_head import SpatialRecoveryFusion
from experiments.prithvi_wxc_probe_20260924.run_pilot_training import (
    HERE, ROOT, CANDIDATE_SHA, sha, make_matched_models, forward)
from experiments.prithvi_wxc_probe_20260924.pilot_training_utils import transform_spatial, error_summary


def validate_runs(runs):
    expected = {(seed,arm) for seed in (24,42,2026)
                for arm in ('baseline','capacity_control','spatial_fusion')}
    if (len(runs) != 9 or {(r['seed'],r['arm']) for r in runs} != expected
            or any(len(r['training_nll']) != 100 or not np.isfinite(r['training_nll']).all() for r in runs)):
        raise ValueError('Require all nine fixed runs and 100 finite losses each')


def audit_run(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(HERE/'pilot_training'):
        raise ValueError('Run must be inside the experiment output directory')
    report = json.loads((folder/'report.json').read_text('utf-8'))
    if report['status'] != 'offline_multisample_pilot_completed':
        raise ValueError('Incomplete training report')
    validate_runs(report['runs'])
    provenance = report['provenance']
    if provenance['candidate_sha256'] != CANDIDATE_SHA or sha(HERE/'pilot_candidate_audit.json') != CANDIDATE_SHA:
        raise ValueError('Candidate population changed')
    for base,key in ((HERE,'code_sha256'),(ROOT,'shared_code_sha256')):
        if any(sha(base/name) != expected for name,expected in provenance[key].items()):
            raise ValueError('Training source code changed')
    data = torch.load(folder/'preprocessing.pt',weights_only=True,map_location='cpu')
    if data['provenance'] != provenance:
        raise ValueError('Preprocessing provenance differs from report')
    rows = json.loads((HERE/'pilot_candidate_audit.json').read_text('utf-8'))['rows']
    if rows != data['candidates']:
        raise ValueError('Preprocessing candidate rows changed')
    x,active,z = data['x'],data['active'],data['spatial_features']
    torch.testing.assert_close(transform_spatial(data['embeddings'],data['pca']),z,rtol=0,atol=0)
    torch.set_num_threads(2)
    verified = []
    for record in report['runs']:
        seed,arm = record['seed'],record['arm']
        path = folder/f'{seed}_{arm}.pt'
        if sha(path) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        saved = torch.load(path,weights_only=True,map_location='cpu')
        if (saved['seed'] != seed or saved['arm'] != arm or saved['updates'] != 100
                or saved['provenance'] != provenance):
            raise ValueError('Checkpoint metadata changed')
        args = (17,data['coords'].numpy(),data['scaler_mean'][-2:].float(),data['scaler_scale'][-2:].float())
        native = RecoveryPM25GNN(*args)
        if arm != 'baseline':
            native = SpatialRecoveryFusion(native,4,hidden_dim=16,freeze_baseline=False)
        native.load_state_dict(saved['state_dict'],strict=True)
        fast = make_matched_models(seed,data['coords'].numpy(),data['scaler_mean'].numpy(),data['scaler_scale'].numpy())[arm]
        fast.load_state_dict(saved['state_dict'],strict=True)
        native.eval()
        fast.eval()
        with torch.no_grad():
            original = forward(native,arm,x,active,z)
            actual = forward(fast,arm,x,active,z)
        torch.testing.assert_close(actual,original,rtol=0,atol=0)
        median = hazard_median(actual).tolist()
        expected_predictions = [dict(**r,predicted_capped_median_hours=p) for r,p in zip(rows,median)]
        if record['predictions'] != expected_predictions:
            raise ValueError('Recomputed predictions differ from report')
        summary = {part:error_summary([r for r in rows if r['partition']==part],
                   [p for r,p in zip(rows,median) if r['partition']==part]) for part in ('train','dev','test')}
        if summary != record['summary']:
            raise ValueError('Recomputed metrics differ from report')
        verified.append(dict(seed=seed,arm=arm,rows=len(rows),native_exact_logits=True))
    audit = dict(status='all_nine_runs_reproduced', runs=verified,
                 report_sha256=sha(folder/'report.json'), preprocessing_sha256=sha(folder/'preprocessing.pt'),
                 audit_code_sha256=sha(Path(__file__)), deployment_verified=False)
    (folder/'audit.json').write_text(json.dumps(audit,indent=2),'utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    print(json.dumps(audit_run(parser.parse_args().run_directory),indent=2))
