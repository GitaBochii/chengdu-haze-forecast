"""Export selected historical AQI cases without training or future feedback."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/continuous_trajectory_20260929'
CURVES = EXPERIMENT / 'full_curve_runs/20260930T033524196780Z'
CALIBRATION = EXPERIMENT / 'recovery_calibration_runs/20260930T121425927928Z'
RAW = ROOT / 'data/processed/regional_air_models_full_history/city_hourly.parquet'
RAW_SHA = 'b16c1941da64f5ef19bc5856cd1215851c4130ffb80b18ca0c16af10f118ab0e'
REPORT_SHA = 'f00346cd05555e20916569c70097f2157c4f89cb879aa7ffaba90dc10b99340e'
CALIBRATION_SHA = '1626e1169ff8e98c5f138f30ff11ebba1825b7010765d52ced304af23d806cce'
TITLES = {'rebound': '短暂转好后反弹', 'sustained_recovery': '连续六小时恢复',
          'missed_rebound': '预测恢复，实测反弹'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(values, hours=None):
    values = np.asarray(values, dtype=float)
    hours = np.arange(1, len(values) + 1) if hours is None else np.asarray(hours)
    if len(hours) != len(values):
        raise ValueError('Aligned hours required')
    recovery = None
    rebounds = []
    start = None
    for i, value in enumerate(values):
        adjacent = i == 0 or hours[i] - hours[i - 1] == 1
        if not adjacent or not np.isfinite(value):
            start = None
        if not np.isfinite(value):
            continue
        if value <= 100:
            if start is None:
                start = i
            if i - start + 1 >= 6 and recovery is None:
                recovery = int(hours[start])
        else:
            if start is not None and adjacent and 1 <= i - start <= 5:
                rebounds.append({'start': int(hours[start]), 'end': int(hours[i-1]),
                                 'rebound_hour': int(hours[i])})
            start = None
    return {'recovery_start': recovery, 'short_rebounds': rebounds}


def category(prediction, actual):
    pred, truth = analyze(prediction), analyze(actual)
    if pred['recovery_start'] is None and truth['recovery_start'] is None and pred['short_rebounds'] and truth['short_rebounds']:
        return 'rebound'
    if pred['recovery_start'] is not None and truth['recovery_start'] is not None:
        return 'sustained_recovery'
    if pred['recovery_start'] is not None and not pred['short_rebounds'] and truth['recovery_start'] is None and truth['short_rebounds']:
        return 'missed_rebound'
    return None


def main():
    if digest(RAW) != RAW_SHA or digest(CURVES / 'report.json') != REPORT_SHA or digest(CALIBRATION / 'report.json') != CALIBRATION_SHA:
        raise ValueError('Frozen source or report changed')
    frozen = json.loads((CURVES / 'report.json').read_text('utf-8'))
    calibrated = json.loads((CALIBRATION / 'report.json').read_text('utf-8'))
    if frozen['status'] != 'completed_fixed_protocol':
        raise ValueError('Frozen run incomplete')
    selected = json.loads((CALIBRATION / 'selected.json').read_text('utf-8'))
    if digest(CALIBRATION / 'selected.json') != calibrated['artifact_sha256']['selected.json']:
        raise ValueError('Calibration selection changed')
    hashes = {'city_hourly.parquet': RAW_SHA, 'full_curve_report.json': REPORT_SHA,
              'calibration_report.json': digest(CALIBRATION / 'report.json'),
              'calibration_selected.json': digest(CALIBRATION / 'selected.json')}
    raw = pd.read_parquet(RAW)
    dates = pd.DatetimeIndex(raw.timestamp).tz_convert('Asia/Shanghai')
    if not dates.is_unique or not np.all(np.diff(dates.asi8) == 3600 * 10**9):
        raise ValueError('Unique hourly source required')
    eligible = []
    counts = {key: 0 for key in TITLES}
    considered = 0
    for year in (2021, 2022, 2023):
        names = [f'{year}_origins.parquet', f'{year}_targets.npy', f'{year}_lag1_chemistry.npy']
        for name in names:
            value = digest(CURVES / name)
            if value != frozen['artifact_sha256'][name]:
                raise ValueError(f'Frozen artifact changed: {name}')
            hashes[name] = value
        origins = pd.read_parquet(CURVES / names[0])
        target = np.load(CURVES / names[1])
        prediction = np.load(CURVES / names[2])
        if prediction.shape != target.shape or prediction.shape != (len(origins), 24):
            raise ValueError('24-hour arrays are not aligned')
        for j, origin in origins.iterrows():
            if origin.partition != 'test' or origin.origin_aqi <= 100:
                continue
            considered += 1
            index = int(origin.source_index)
            if dates[index] != origin.timestamp:
                raise ValueError('Origin timestamp mismatch')
            # Independently reconstruct future targets from the raw source rows.
            observed = raw.iloc[index+1:index+25].aqi.to_numpy(dtype=np.float32)
            if not np.array_equal(observed, target[j]):
                raise ValueError('Source future target mismatch')
            history = raw.iloc[index-24:index]
            values = history[['aqi', 'pm2_5', 'pm10']].to_numpy(float)
            p = prediction[j]
            if not np.isfinite(values).all() or (values < 0).any() or not np.isfinite(p).all():
                continue
            # A fixed, disclosed display range keeps the plot readable.
            if min(observed.min(), p.min()) < 20 or max(observed.max(), p.max()) > 250:
                continue
            group = category(p, observed)
            if group is None:
                continue
            counts[group] += 1
            eligible.append((origin.timestamp, group, year, index, j, p.copy(), observed.copy()))
    eligible.sort(key=lambda item: item[0])
    cases = []
    audit = []
    used = set()
    for timestamp, group, year, index, j, p, observed in eligible:
        if group in used:
            continue
        used.add(group)
        buffer = int(selected[str(year)]['lag1_chemistry'])
        hist = []
        for pos in range(index-24, index):
            row = raw.iloc[pos]
            hist.append({'timestamp': dates[pos].isoformat(), **{name: float(row[name]) for name in ('aqi', 'pm2_5', 'pm10')}})
        actual = []
        forecast = []
        for h in range(1, 25):
            row = raw.iloc[index+h]
            stamp = dates[index+h].isoformat()
            actual.append({'hour': h, 'timestamp': stamp, **{name: float(row[name]) for name in ('aqi', 'pm2_5', 'pm10')}})
            forecast.append({'hour': h, 'timestamp': stamp, 'aqi': float(p[h-1]), 'buffered_aqi': float(p[h-1]) + buffer})
        cases.append({'id': group, 'title': TITLES[group], 'category': group,
                      'timestamp': timestamp.isoformat(), 'train_end_year': year-2,
                      'dev_year': year-1, 'test_year': year, 'buffer_aqi': buffer,
                      'history': hist, 'forecast': forecast, 'actual': actual})
        audit.append({'id': group, 'timestamp': timestamp.isoformat(), 'source_index': index,
                      'frozen_row_index': j, 'prediction': analyze(p), 'actual': analyze(observed)})
    if used != set(TITLES):
        raise ValueError(f'Missing categories: {set(TITLES) - used}; counts={counts}')
    cases.sort(key=lambda c: list(TITLES).index(c['category']))
    payload = {'version': 1,
               'notice': '精选历史案例，不代表随机样本或总体精度；非实时预警。实测为预测之后的历史记录。',
               'criterion': '仅按AQI判断：连续六小时AQI不高于100；低值仅持续一至五小时后超过100视为短暂转好后反弹。',
               'model_scope': '冻结的历史24小时AQI预测；输入观测为预测时刻前24小时至前1小时。缓冲值仅用于保守判定，不是污染浓度预测。',
               'case_selection': {'rule': '2021至2023年测试集污染起点；历史输入完整；预测与未来实测AQI均在20至250范围内；失败类要求预测无短暂反弹；每类按预测时刻排序取首例。',
                                  'considered_polluted_origins': considered, 'eligible_counts': counts},
               'source_hashes': hashes, 'cases': cases}
    destination = ROOT / 'project-showcase/rebound-cases.json'
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    output = ROOT / 'artifacts/rebound_showcase'
    output.mkdir(parents=True, exist_ok=True)
    evidence = {'status': 'passed_source_targets_and_frozen_hashes', 'export_sha256': digest(destination),
                'source_hashes': hashes, 'selection': payload['case_selection'], 'cases': audit,
                'exporter_sha256': digest(Path(__file__))}
    (output / 'export-audit.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    main()
