"""Align selected frozen cross-year experiments to canonical cohorts; no model selection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.cohort_scorecard import score_predictions
from haze_forecast.paired_interval_score import paired_error_summary

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = ROOT/'artifacts/task01_recovery_head'
    out = b/'crossyear_tradeoffs_20260918'
    if out.exists():
        raise FileExistsError(out)
    labelpath=b/'interval_business_labels_v1/origins.parquet'
    labels=pd.read_parquet(labelpath).set_index('timestamp')
    hashes={str(labelpath):hashlib.sha256(labelpath.read_bytes()).hexdigest()}
    results={}
    for year in (2024,2025):
        path=b/f'direct_history_crossyear_20260918/{year}/scored_predictions.parquet'
        raw=pd.read_parquet(path)
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        print('neural columns',year,raw.columns.tolist(),flush=True)
        cohort=labels.loc[raw.timestamp].reset_index()
        candidates={}
        for arm in ('original','uniform','masked'):
            path=b/f'interval_weather_crossyear_v1/{year}_{arm}_expanded.parquet'
            weather=pd.read_parquet(path).set_index('timestamp')
            assert weather.index.is_unique and set(weather.index)==set(raw.timestamp)
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            candidates[arm]=weather.loc[cohort.timestamp,'weather_candidate'].to_numpy()
        for arm in ('snapshot','history'):
            candidates[arm]=raw.set_index('timestamp').loc[cohort.timestamp,arm].to_numpy()
        scores={n:score_predictions(cohort,p) for n,p in candidates.items()}
        paired={n:paired_error_summary(cohort,candidates['original'],p) for n,p in candidates.items() if n!='original'}
        results[str(year)]={'scores':scores,'paired':paired}
    hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    out.mkdir()
    (out/'report.json').write_text(json.dumps({'years':results,'input_sha256':hashes,
        'limits':'ERA5 arms are offline only. Reused years, no new blind test. No metric preference selected.'},indent=2,allow_nan=False),encoding='utf-8')
    for year,r in results.items():
        print(year,{n:[s[k] for k in ('exact_row_mae','exact_block_mae','tail24_block_mae')] for n,s in r['scores'].items()})


if __name__=='__main__':
    main()
