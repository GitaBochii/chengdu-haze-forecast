"""Read-only access to one audited retrospective multimodal research run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
from functools import cached_property
from pathlib import Path

import pandas as pd

RUN_ID = '20260926T054740428470Z'
EXPERIMENT = 'experiments/prithvi_wxc_probe_20260924'
AUDIT_SHA = 'e87505ad6c44a35d3d363533d3eccb97e4e2eea9381865d9d62f68861697101b'
SEEDS = (24, 42, 2026)
ARMS = {'baseline': '污染物序列基线', 'capacity_control': '等容量对照',
        'spatial_fusion': 'Prithvi 气象空间融合'}
NOTICE = ('仅供 2020 年历史离线研究：25 个起点，非实时预报。气象输入为再分析，'
          '2020 年曾用于基础模型验证；不能视为未见年份盲测。'
          '预测及误差按 68 小时封顶计算，68 表示至少 68 小时，不是精确消散时刻。')


class MultimodalInputError(ValueError):
    """Invalid public parameters, safe to return as HTTP 400."""


def _check(path, expected):
    try:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    except OSError as error:
        raise RuntimeError('Multimodal research artifact unavailable') from error
    if digest != expected:
        raise RuntimeError('Multimodal research integrity check failed')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class MultimodalResearch:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.experiment = self.root / EXPERIMENT
        self.run_dir = self.experiment / 'pilot_training' / RUN_ID

    def _verified(self):
        # A pinned audit is the trust anchor, not a mutable adjacent manifest.
        _check(self.run_dir / 'audit.json', AUDIT_SHA)
        audit = _read(self.run_dir / 'audit.json')
        _check(self.run_dir / 'report.json', audit['report_sha256'])
        _check(self.run_dir / 'preprocessing.pt', audit['preprocessing_sha256'])
        report = _read(self.run_dir / 'report.json')
        expected = {(s,a) for s in SEEDS for a in ARMS}
        if (audit['status'] != 'all_nine_runs_reproduced'
                or len(audit['runs']) != 9 or len(report['runs']) != 9
                or {(r['seed'],r['arm']) for r in audit['runs']} != expected
                or any(r['rows'] != 25 or not r['native_exact_logits'] for r in audit['runs'])
                or {(r['seed'],r['arm']) for r in report['runs']} != expected):
            raise RuntimeError('Incomplete multimodal research run')
        manifest = self.experiment / 'pilot_candidate_audit.json'
        _check(manifest, report['provenance']['candidate_sha256'])
        rows = _read(manifest)['rows']
        identities = {(r['timestamp'],r['partition']) for r in rows}
        if len(rows) != 25 or len(identities) != 25:
            raise RuntimeError('Incomplete candidate manifest')
        for run in report['runs']:
            _check(self.run_dir / f"{run['seed']}_{run['arm']}.pt", run['checkpoint_sha256'])
            predictions = run['predictions']
            if (len(predictions) != 25
                    or {(r['timestamp'],r['partition']) for r in predictions} != identities):
                raise RuntimeError('Incomplete prediction archive')
            for original, prediction in zip(rows, predictions, strict=True):
                if any(prediction[k] != v for k,v in original.items()):
                    raise RuntimeError('Prediction labels differ from frozen candidates')
                value = prediction['predicted_capped_median_hours']
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 68:
                    raise RuntimeError('Invalid archived prediction')
        return report, rows

    def status(self):
        _, rows = self._verified()
        return dict(available=True, run_id=RUN_ID, origins=len(rows),
                    partitions=dict(Counter(r['partition'] for r in rows)),
                    arms=[dict(value=k,label=v) for k,v in ARMS.items()], seeds=list(SEEDS),
                    reinfer_available=importlib.util.find_spec('torch') is not None)

    def cases(self):
        _, rows = self._verified()
        return {'records': [dict(timestamp=r['timestamp'],partition=r['partition']) for r in rows]}

    def _select(self, timestamp, seed, arm):
        if type(seed) is not int or seed not in SEEDS:
            raise MultimodalInputError('请选择固定实验种子 24、42 或 2026')
        if not isinstance(arm, str) or arm not in ARMS:
            raise MultimodalInputError('请选择有效的实验模型')
        if not isinstance(timestamp, str) or len(timestamp) > 64:
            raise MultimodalInputError('请选择已归档的历史起点')
        report, rows = self._verified()
        matches = [i for i,r in enumerate(rows) if r['timestamp'] == timestamp]
        if not matches:
            raise MultimodalInputError('请选择已归档的历史起点')
        run = next(r for r in report['runs'] if r['seed'] == seed and r['arm'] == arm)
        return report, rows, run, matches[0]

    @cached_property
    def _city(self):
        return pd.read_parquet(self.root / 'data/processed/regional_air_models_full_history/city_hourly.parquet')

    def _response(self, report, row, seed, arm, prediction, mode):
        city_path = self.root / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
        _check(city_path, report['provenance']['source_sha256']['city_hourly.parquet'])
        origin = pd.Timestamp(row['timestamp'])
        city = self._city.set_index('timestamp')
        if not city.index.is_unique:
            raise RuntimeError('Duplicate city observations')
        hours = pd.date_range(origin - pd.Timedelta(hours=23), origin, freq='h')
        history = city.reindex(hours)
        trend = [dict(timestamp=t.isoformat(), **{
            k: float(r[k]) if pd.notna(r[k]) and math.isfinite(float(r[k])) else None
            for k in ('aqi','pm2_5','pm10')}) for t,r in history.iterrows()]
        low, high = row['lower_bound_hours'], row['upper_bound_hours']
        capped_low, capped_high = min(low,68), min(high,68)
        return dict(timestamp=row['timestamp'], partition=row['partition'], seed=seed,arm=arm,
                    prediction_hours=prediction, capped_tail=prediction == 68,
                    lower_bound_hours=low, upper_bound_hours=high if math.isfinite(high) else None,
                    reason=row['reason'], error_lower_hours=max(capped_low-prediction,prediction-capped_high,0),
                    error_upper_hours=max(abs(prediction-capped_low),abs(prediction-capped_high)),
                    model_label=ARMS[arm], mode=mode, notice=NOTICE, eligible=True,trend=trend)

    def replay(self, timestamp, seed=24, arm='spatial_fusion'):
        report, rows, run, index = self._select(timestamp,seed,arm)
        return self._response(report,rows[index],seed,arm,
                              run['predictions'][index]['predicted_capped_median_hours'], 'multimodal_archive')

    def reinfer(self, timestamp, seed=24, arm='spatial_fusion'):
        report, rows, run, index = self._select(timestamp,seed,arm)
        provenance = report['provenance']
        for name, digest in provenance['shared_code_sha256'].items():
            _check(self.root / name,digest)
        _check(self.experiment / 'fusion_head.py', provenance['code_sha256']['fusion_head.py'])
        import torch
        from haze_forecast.regional_recovery_network import RecoveryPM25GNN
        from haze_forecast.recovery_survival import hazard_median
        from experiments.prithvi_wxc_probe_20260924.fusion_head import SpatialRecoveryFusion

        data = torch.load(self.run_dir / 'preprocessing.pt', weights_only=True,map_location='cpu')
        saved = torch.load(self.run_dir / f'{seed}_{arm}.pt',weights_only=True,map_location='cpu')
        if (data['provenance'] != provenance or saved['provenance'] != provenance
                or data['candidates'] != rows or saved['seed'] != seed or saved['arm'] != arm):
            raise RuntimeError('Stored model provenance mismatch')
        baseline = RecoveryPM25GNN(17,data['coords'].numpy(),data['scaler_mean'][-2:],data['scaler_scale'][-2:])
        model = baseline if arm == 'baseline' else SpatialRecoveryFusion(baseline,4,hidden_dim=16,freeze_baseline=False)
        model.load_state_dict(saved['state_dict'],strict=True)
        model.eval()
        x, active = data['x'][index:index+1], data['active'][index:index+1]
        with torch.inference_mode():
            if arm == 'baseline':
                logits = model(x,active)[1]
            else:
                z = data['spatial_features'][index:index+1]
                logits = model(x,active,z if arm == 'spatial_fusion' else torch.zeros_like(z),
                               torch.ones(1,dtype=torch.bool))[1]
            if logits.shape != (1,67) or not torch.isfinite(logits).all():
                raise RuntimeError('Invalid local model output')
            prediction = int(hazard_median(logits)[0])
        if prediction != run['predictions'][index]['predicted_capped_median_hours']:
            raise RuntimeError('Local inference differs from audited archive')
        return self._response(report,rows[index],seed,arm,prediction,'multimodal_reinference')
